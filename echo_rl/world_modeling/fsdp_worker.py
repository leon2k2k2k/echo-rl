from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

import ray
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
from skyrl.train.dataset.replay_buffer import Experience

from echo_rl.world_modeling.loss import compute_world_model_loss, get_scheduled_coeff
from echo_rl.world_modeling.nextlat import (
    NextLatDynamicsModel,
    NextLatRLConfig,
    build_nextlat_token_mask,
    compute_nextlat_mse_loss,
)


class EchoFSDPPolicyWorkerBase(FSDPPolicyWorkerBase):
    """FSDP policy worker that adds ECHO's auxiliary world-modeling loss."""

    def _nextlat_dynamics(self):
        for module in (
            self.model,
            getattr(self.model, "model", None),
            getattr(self.model, "module", None),
            getattr(self.model, "_fsdp_wrapped_module", None),
            getattr(getattr(self.model, "model", None), "module", None),
            getattr(getattr(self.model, "model", None), "_fsdp_wrapped_module", None),
        ):
            if module is not None and hasattr(module, "nextlat_dynamics"):
                return module.nextlat_dynamics
        return None

    def initialize_aux_policy_modules(self, wrapped_model, model_config) -> None:
        nextlat_coeff = float(getattr(self.cfg.algorithm, "nextlat_coeff", 0.0) or 0.0)
        if nextlat_coeff <= 0:
            return
        nextlat_cfg = NextLatRLConfig(
            coeff=nextlat_coeff,
            mtp_horizon=int(getattr(self.cfg.algorithm, "nextlat_mtp_horizon", 1)),
            proj_factor=float(getattr(self.cfg.algorithm, "nextlat_proj_factor", 1.0)),
            bias=bool(getattr(self.cfg.algorithm, "nextlat_bias", False)),
            norm_eps=float(getattr(self.cfg.algorithm, "nextlat_norm_eps", 1e-5)),
        )
        wrapped_model.model.nextlat_dynamics = NextLatDynamicsModel(model_config.hidden_size, nextlat_cfg)

    def needs_aux_policy_hidden_states(self, loss_config: Any) -> bool:
        return float(getattr(loss_config, "nextlat_coeff", 0.0) or 0.0) > 0

    def get_aux_policy_loss_context(self, data: TrainingInputBatch) -> Dict[str, Any]:
        world_loss_mask = data.get("world_loss_mask")
        token_denominator = None
        if world_loss_mask is not None:
            token_denominator = float(world_loss_mask.sum().item() + 1e-3)
        return {
            "world_loss_weight": 1.0 / max(len(data), 1),
            "world_token_normalization_denominator": token_denominator,
        }

    def compute_aux_policy_loss(
        self,
        *,
        action_log_probs: torch.Tensor,
        model_output: Any,
        sequences: torch.Tensor,
        num_actions: int,
        attention_mask: Optional[torch.Tensor],
        policy_loss: torch.Tensor,
        experience: Experience,
        loss_config: Any,
        microbatch_weight: float,
        grad_sum_correction_factor: float,
        aux_policy_loss_context: Optional[Dict[str, Any]] = None,
    ):
        extras = experience.extras or {}
        metadata = experience.metadata or {}
        global_step = int(metadata.get("world_model_schedule_step", metadata.get("global_step", 0)))
        total_training_steps = int(metadata.get("total_training_steps", 0))
        world_model_coeff = get_scheduled_coeff(
            initial=loss_config.world_model_coeff,
            final=loss_config.world_model_coeff_end,
            step=global_step,
            total_steps=total_training_steps,
            schedule=loss_config.world_model_coeff_schedule,
            transition_step=loss_config.world_model_coeff_transition_step,
            steps=loss_config.world_model_coeff_steps,
            values=loss_config.world_model_coeff_values,
        )
        loss_config = replace(loss_config, world_model_coeff=world_model_coeff)
        context = aux_policy_loss_context or {}
        world_loss_scaled, world_metrics = compute_world_model_loss(
            action_log_probs,
            extras.get("world_loss_mask"),
            config=loss_config,
            warning_mask=extras.get("world_warning_mask"),
            env_mask=extras.get("world_env_mask"),
            full_observation_count=extras.get("world_full_observation_count"),
            token_normalization_denominator=context.get("world_token_normalization_denominator"),
        )
        nextlat_scaled = None
        nextlat_metrics: Dict[str, Any] = {}
        nextlat_coeff = float(getattr(loss_config, "nextlat_coeff", 0.0) or 0.0)
        nextlat_mask = extras.get("nextlat_loss_mask")
        if nextlat_coeff > 0:
            if nextlat_mask is None:
                raise RuntimeError("nextlat_coeff > 0 requires nextlat_loss_mask in the training batch.")
            dynamics_model = self._nextlat_dynamics()
            if dynamics_model is None:
                raise RuntimeError("nextlat_coeff > 0 but nextlat_dynamics was not initialized.")
            if getattr(self.model, "sequence_parallel_size", 1) != 1:
                raise RuntimeError("NextLat MSE currently requires sequence_parallel_size=1.")
            nextlat_cfg = NextLatRLConfig(
                coeff=nextlat_coeff,
                mtp_horizon=int(getattr(loss_config, "nextlat_mtp_horizon", 1)),
                proj_factor=float(getattr(loss_config, "nextlat_proj_factor", 1.0)),
                bias=bool(getattr(loss_config, "nextlat_bias", False)),
                norm_eps=float(getattr(loss_config, "nextlat_norm_eps", 1e-5)),
            )
            nextlat_token_mask = build_nextlat_token_mask(
                nextlat_mask,
                sequence_length=sequences.shape[1],
                num_actions=int(num_actions),
                device=action_log_probs.device,
            )
            nextlat_scaled, nextlat_metrics = compute_nextlat_mse_loss(
                dynamics_model,
                model_output=model_output,
                nextlat_token_mask=nextlat_token_mask,
                config=nextlat_cfg,
            )

        if world_loss_scaled is None and nextlat_scaled is None:
            coeff_tensor = torch.tensor(float(world_model_coeff), device=action_log_probs.device)
            return None, {
                "status/world_model_coeff": coeff_tensor,
                "status/nextlat_coeff": torch.tensor(float(nextlat_coeff), device=action_log_probs.device),
            }

        aux_loss = action_log_probs.sum() * 0.0
        if loss_config.loss_reduction in ("sequence_mean", "seq_mean_token_sum_norm"):
            if world_loss_scaled is not None:
                aux_loss = aux_loss + world_loss_scaled * context.get("world_loss_weight", microbatch_weight)
        else:
            if world_loss_scaled is not None:
                aux_loss = aux_loss + world_loss_scaled
        if nextlat_scaled is not None:
            aux_loss = aux_loss + nextlat_scaled * microbatch_weight

        world_metrics = dict(world_metrics)
        world_metrics.update(nextlat_metrics)
        world_metrics["status/world_model_coeff"] = torch.tensor(
            float(world_model_coeff), device=action_log_probs.device
        )
        world_metrics["status/nextlat_coeff"] = torch.tensor(float(nextlat_coeff), device=action_log_probs.device)
        if world_metrics.get("world_loss_scaled") is not None:
            world_metrics["world_policy_loss_ratio"] = (
                world_metrics["world_loss_scaled"].detach().abs() / (policy_loss.detach().abs() + 1e-8)
            )
        if world_metrics.get("nextlat_loss_scaled") is not None:
            world_metrics["nextlat_policy_loss_ratio"] = (
                world_metrics["nextlat_loss_scaled"].detach().abs() / (policy_loss.detach().abs() + 1e-8)
            )
        dynamics_model = self._nextlat_dynamics()
        if dynamics_model is not None:
            param_count = sum(p.numel() for p in dynamics_model.parameters() if p.requires_grad)
            param_norm = action_log_probs.sum() * 0.0
            grad_norm = action_log_probs.sum() * 0.0
            has_grad = False
            for param in dynamics_model.parameters():
                if param.requires_grad:
                    param_norm = param_norm + param.detach().float().pow(2).sum().to(action_log_probs.device)
                    if param.grad is not None:
                        has_grad = True
                        grad_norm = grad_norm + param.grad.detach().float().pow(2).sum().to(action_log_probs.device)
            world_metrics["nextlat_trainable_params"] = torch.tensor(
                float(param_count), device=action_log_probs.device
            )
            world_metrics["nextlat_param_norm"] = param_norm.sqrt()
            world_metrics["nextlat_grad_norm_pre_backward"] = grad_norm.sqrt() if has_grad else grad_norm
        return aux_loss, world_metrics


PolicyWorker = ray.remote(num_gpus=1)(EchoFSDPPolicyWorkerBase)
