from __future__ import annotations

import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import ray
import torch
from loguru import logger
from transformers import AutoConfig

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

    def init_model(self, model_path, num_training_steps: int = None):
        self._defer_nextlat_init = True
        try:
            super().init_model(model_path, num_training_steps=num_training_steps)
        finally:
            self._defer_nextlat_init = False
        self._initialize_local_nextlat_dynamics(model_path)

    @staticmethod
    def _policy_metric_logging_enabled() -> bool:
        return os.environ.get("ECHO_LOG_POLICY_TRAIN_METRICS", "1") != "0"

    @staticmethod
    def _scalar_for_log(value: Any) -> Optional[float]:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
            if value.numel() != 1:
                value = value.float().mean()
            return float(value.float().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _tensor_sum_for_log(value: Any) -> Optional[float]:
        if value is None or not hasattr(value, "detach"):
            return None
        return float(value.detach().float().sum().cpu().item())

    def _rank_for_log(self) -> int:
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return int(torch.distributed.get_rank())
        except Exception:
            pass
        return int(getattr(self, "_rank", -1))

    def _policy_rank_log_path(self) -> Optional[Path]:
        log_dir = os.environ.get("ECHO_POLICY_RANK_LOG_DIR")
        if not log_dir:
            output_dir = os.environ.get("OUTPUT_DIR")
            if not output_dir:
                skyrl_log_file = os.environ.get("SKYRL_LOG_FILE")
                if skyrl_log_file:
                    output_dir = str(Path(skyrl_log_file).resolve().parent.parent)
            if output_dir:
                log_dir = str(Path(output_dir) / "policy_rank_logs")
        if not log_dir:
            return None
        rank = self._rank_for_log()
        return Path(log_dir) / f"rank_{rank}_pid_{os.getpid()}.log"

    def _policy_rank_log(self, event: str, **fields: Any) -> None:
        path = self._policy_rank_log_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            parts = [
                time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                f"event={event}",
                f"pid={os.getpid()}",
                f"rank={self._rank_for_log()}",
            ]
            parts.extend(f"{key}={value!r}" for key, value in fields.items())
            with path.open("a", encoding="utf-8") as f:
                f.write(" ".join(parts) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass

    def _batch_summary_for_log(self, data: TrainingInputBatch) -> Dict[str, Any]:
        sequences = data.get("sequences")
        seq_len = int(sequences.shape[1]) if sequences is not None and hasattr(sequences, "shape") else None
        micro_batch_size = int(getattr(self.cfg, "micro_train_batch_size_per_gpu", 1) or 1)
        return {
            "batch": len(data),
            "micro_batch_size": micro_batch_size,
            "micro_batches": math.ceil(len(data) / micro_batch_size) if micro_batch_size > 0 else None,
            "seq_len": seq_len,
            "response_length": (data.metadata or {}).get("response_length"),
            "loss_tokens": self._tensor_sum_for_log(data.get("loss_mask")),
            "world_tokens": self._tensor_sum_for_log(data.get("world_loss_mask")),
            "world_warning_tokens": self._tensor_sum_for_log(data.get("world_warning_mask")),
            "world_env_tokens": self._tensor_sum_for_log(data.get("world_env_mask")),
            "nextlat_tokens": self._tensor_sum_for_log(data.get("nextlat_loss_mask")),
        }

    def forward_backward(
        self,
        data: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        if not self._policy_metric_logging_enabled():
            return super().forward_backward(data, loss_fn=loss_fn, loss_fn_config=loss_fn_config)

        started = time.monotonic()
        rank = self._rank_for_log()
        summary = self._batch_summary_for_log(data)
        self._policy_rank_log(
            "forward_backward_start",
            loss_fn=loss_fn or self.cfg.algorithm.policy_loss_type,
            **summary,
        )
        logger.info(
            "[policy-train] forward_backward_start "
            f"rank={rank} loss_fn={loss_fn or self.cfg.algorithm.policy_loss_type} "
            f"batch={summary['batch']} micro_batch_size={summary['micro_batch_size']} "
            f"micro_batches={summary['micro_batches']} seq_len={summary['seq_len']} "
            f"response_length={summary['response_length']} loss_tokens={summary['loss_tokens']} "
            f"world_tokens={summary['world_tokens']} warning_tokens={summary['world_warning_tokens']} "
            f"env_tokens={summary['world_env_tokens']} nextlat_tokens={summary['nextlat_tokens']}"
        )
        try:
            status = super().forward_backward(data, loss_fn=loss_fn, loss_fn_config=loss_fn_config)
        except Exception as exc:
            elapsed = time.monotonic() - started
            self._policy_rank_log("forward_backward_failed", sec=round(elapsed, 2), error=repr(exc))
            logger.exception(f"[policy-train] forward_backward_failed rank={rank} sec={elapsed:.2f}")
            raise

        elapsed = time.monotonic() - started
        tokens = summary["loss_tokens"] or summary["world_tokens"] or 0.0
        tokens_per_sec = (tokens / elapsed) if elapsed > 0 and tokens else 0.0
        self._policy_rank_log(
            "forward_backward_done",
            sec=round(elapsed, 2),
            tokens_per_sec=round(tokens_per_sec, 2),
            final_loss=status.get("final_loss"),
            policy_loss=status.get("policy_loss"),
            policy_entropy=status.get("policy_entropy"),
            world_loss_scaled=status.get("loss_metrics/world_loss_scaled"),
            world_ce=status.get("loss_metrics/world_ce_selected_per_token"),
            world_policy_ratio=status.get("loss_metrics/world_policy_loss_ratio"),
            nextlat_loss_scaled=status.get("loss_metrics/nextlat_loss_scaled"),
        )
        logger.info(
            "[policy-train] forward_backward_done "
            f"rank={rank} sec={elapsed:.2f} tokens_per_sec={tokens_per_sec:.2f} "
            f"final_loss={status.get('final_loss')} policy_loss={status.get('policy_loss')} "
            f"policy_entropy={status.get('policy_entropy')} world_loss_scaled={status.get('loss_metrics/world_loss_scaled')} "
            f"world_ce={status.get('loss_metrics/world_ce_selected_per_token')} "
            f"world_policy_ratio={status.get('loss_metrics/world_policy_loss_ratio')} "
            f"nextlat_loss_scaled={status.get('loss_metrics/nextlat_loss_scaled')}"
        )
        return status

    def optim_step(self) -> float:
        if not self._policy_metric_logging_enabled():
            return super().optim_step()

        started = time.monotonic()
        rank = self._rank_for_log()
        self._policy_rank_log("optim_step_start")
        logger.info(f"[policy-train] optim_step_start rank={rank}")
        try:
            grad_norm = super().optim_step()
        except Exception as exc:
            elapsed = time.monotonic() - started
            self._policy_rank_log("optim_step_failed", sec=round(elapsed, 2), error=repr(exc))
            logger.exception(f"[policy-train] optim_step_failed rank={rank} sec={elapsed:.2f}")
            raise

        elapsed = time.monotonic() - started
        self._policy_rank_log("optim_step_done", sec=round(elapsed, 2), grad_norm=grad_norm, lr=self.get_lr())
        logger.info(
            "[policy-train] optim_step_done "
            f"rank={rank} sec={elapsed:.2f} grad_norm={grad_norm} lr={self.get_lr()}"
        )
        return grad_norm

    def _nextlat_dynamics(self):
        local_dynamics = getattr(self, "_local_nextlat_dynamics", None)
        if local_dynamics is not None:
            return local_dynamics
        owner = self._nextlat_dynamics_owner()
        return getattr(owner, "nextlat_dynamics", None) if owner is not None else None

    def _nextlat_dynamics_owner(self):
        for module in (
            self.model,
            getattr(self.model, "model", None),
            getattr(self.model, "module", None),
            getattr(self.model, "_fsdp_wrapped_module", None),
            getattr(getattr(self.model, "model", None), "module", None),
            getattr(getattr(self.model, "model", None), "_fsdp_wrapped_module", None),
        ):
            if module is not None and hasattr(module, "nextlat_dynamics"):
                return module
        return None

    async def broadcast_to_inference_engines(self, *args, **kwargs):
        owner = self._nextlat_dynamics_owner()
        dynamics_model = getattr(owner, "nextlat_dynamics", None) if owner is not None else None
        if dynamics_model is None:
            return await super().broadcast_to_inference_engines(*args, **kwargs)

        # vLLM serves the base Qwen model and does not have ECHO's auxiliary
        # NextLat head. Keep the head trainable/checkpointable on the policy
        # worker, but exclude it from sampler weight sync.
        delattr(owner, "nextlat_dynamics")
        self._policy_rank_log("nextlat_sync_exclude_start")
        try:
            return await super().broadcast_to_inference_engines(*args, **kwargs)
        finally:
            owner.nextlat_dynamics = dynamics_model
            self._policy_rank_log("nextlat_sync_exclude_done")

    def _build_nextlat_config(self) -> Optional[NextLatRLConfig]:
        nextlat_coeff = float(getattr(self.cfg.algorithm, "nextlat_coeff", 0.0) or 0.0)
        if nextlat_coeff <= 0:
            return None
        return NextLatRLConfig(
            coeff=nextlat_coeff,
            mtp_horizon=int(getattr(self.cfg.algorithm, "nextlat_mtp_horizon", 1)),
            proj_factor=float(getattr(self.cfg.algorithm, "nextlat_proj_factor", 1.0)),
            bias=bool(getattr(self.cfg.algorithm, "nextlat_bias", False)),
            norm_eps=float(getattr(self.cfg.algorithm, "nextlat_norm_eps", 1e-5)),
        )

    def _initialize_local_nextlat_dynamics(self, model_path) -> None:
        nextlat_cfg = self._build_nextlat_config()
        if nextlat_cfg is None:
            return
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        dynamics_model = NextLatDynamicsModel(model_config.hidden_size, nextlat_cfg)
        base_param = next(self.model.parameters())
        if hasattr(base_param, "to_local"):
            base_param = base_param.to_local()
        dtype = base_param.dtype if base_param.dtype.is_floating_point else torch.float32
        dynamics_model.to(device=base_param.device, dtype=dtype)
        self._local_nextlat_dynamics = dynamics_model
        self._add_local_nextlat_optimizer_group(dynamics_model)
        self._policy_rank_log(
            "nextlat_local_init",
            params=sum(p.numel() for p in dynamics_model.parameters() if p.requires_grad),
            dtype=str(dtype),
            device=str(base_param.device),
        )

    def _add_local_nextlat_optimizer_group(self, dynamics_model: NextLatDynamicsModel) -> None:
        if self.optimizer is None:
            return
        base_group = self.optimizer.param_groups[0]
        base_lr = base_group.get("initial_lr", base_group.get("lr", 0.0))
        self.optimizer.add_param_group(
            {
                "params": list(dynamics_model.parameters()),
                "lr": base_group.get("lr", base_lr),
                "initial_lr": base_lr,
            }
        )
        scheduler = getattr(self, "scheduler", None)
        if scheduler is None:
            return
        if hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs.append(base_lr)
        if hasattr(scheduler, "lr_lambdas") and scheduler.lr_lambdas:
            scheduler.lr_lambdas.append(scheduler.lr_lambdas[0])
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr.append(self.optimizer.param_groups[-1].get("lr", base_lr))

    def initialize_aux_policy_modules(self, wrapped_model, model_config) -> None:
        nextlat_cfg = self._build_nextlat_config()
        if nextlat_cfg is None:
            return
        if getattr(self, "_defer_nextlat_init", False):
            return
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
        self._policy_rank_log(
            "aux_loss_start",
            global_step=global_step,
            total_training_steps=total_training_steps,
            policy_loss=self._scalar_for_log(policy_loss),
            world_model_coeff=world_model_coeff,
            nextlat_coeff=float(getattr(loss_config, "nextlat_coeff", 0.0) or 0.0),
            microbatch_weight=microbatch_weight,
            grad_sum_correction_factor=grad_sum_correction_factor,
            world_mask_tokens=self._tensor_sum_for_log(extras.get("world_loss_mask")),
            warning_mask_tokens=self._tensor_sum_for_log(extras.get("world_warning_mask")),
            env_mask_tokens=self._tensor_sum_for_log(extras.get("world_env_mask")),
            nextlat_mask_tokens=self._tensor_sum_for_log(extras.get("nextlat_loss_mask")),
        )
        world_loss_scaled, world_metrics = compute_world_model_loss(
            action_log_probs,
            extras.get("world_loss_mask"),
            config=loss_config,
            warning_mask=extras.get("world_warning_mask"),
            env_mask=extras.get("world_env_mask"),
            full_observation_count=extras.get("world_full_observation_count"),
            token_normalization_denominator=context.get("world_token_normalization_denominator"),
        )
        self._policy_rank_log(
            "world_loss_done",
            world_loss_scaled=self._scalar_for_log(world_loss_scaled),
            world_loss_unscaled=self._scalar_for_log(world_metrics.get("world_loss_unscaled")),
            world_ce=self._scalar_for_log(world_metrics.get("world_ce_selected_per_token")),
            world_tokens=self._scalar_for_log(world_metrics.get("world_tokens_selected")),
            zero_world_tokens=self._scalar_for_log(world_metrics.get("world_zero_token")),
        )
        if self._scalar_for_log(world_metrics.get("world_zero_token")) == 1.0:
            self._policy_rank_log(
                "zero_world_tokens",
                policy_loss=self._scalar_for_log(policy_loss),
                loss_mask_tokens=self._tensor_sum_for_log(experience.loss_mask),
                world_model_coeff=world_model_coeff,
                nextlat_coeff=float(getattr(loss_config, "nextlat_coeff", 0.0) or 0.0),
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
            self._policy_rank_log(
                "nextlat_loss_done",
                nextlat_loss_scaled=self._scalar_for_log(nextlat_scaled),
                nextlat_loss_unscaled=self._scalar_for_log(nextlat_metrics.get("nextlat_loss_unscaled")),
                nextlat_tokens=self._scalar_for_log(nextlat_metrics.get("nextlat_tokens_selected")),
            )

        if world_loss_scaled is None and nextlat_scaled is None:
            coeff_tensor = torch.tensor(float(world_model_coeff), device=action_log_probs.device)
            self._policy_rank_log("aux_loss_skip", reason="no_aux_terms", world_model_coeff=world_model_coeff)
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
        if self._policy_metric_logging_enabled():
            log_values = {
                "policy_loss": self._scalar_for_log(policy_loss),
                "aux_loss": self._scalar_for_log(aux_loss),
                "world_loss_scaled": self._scalar_for_log(world_metrics.get("world_loss_scaled")),
                "world_loss_unscaled": self._scalar_for_log(world_metrics.get("world_loss_unscaled")),
                "world_ce": self._scalar_for_log(world_metrics.get("world_ce_selected_per_token")),
                "world_tokens": self._scalar_for_log(world_metrics.get("world_tokens_selected")),
                "world_warning_tokens": self._scalar_for_log(world_metrics.get("world_tokens_warning")),
                "world_env_tokens": self._scalar_for_log(world_metrics.get("world_tokens_env")),
                "zero_world_tokens": self._scalar_for_log(world_metrics.get("world_zero_token")),
                "world_policy_ratio": self._scalar_for_log(world_metrics.get("world_policy_loss_ratio")),
                "nextlat_loss_scaled": self._scalar_for_log(world_metrics.get("nextlat_loss_scaled")),
                "nextlat_ratio": self._scalar_for_log(world_metrics.get("nextlat_policy_loss_ratio")),
            }
            logger.info(
                "[policy-train] aux_loss "
                f"rank={self._rank_for_log()} microbatch_weight={microbatch_weight:.6g} "
                f"grad_sum_correction={grad_sum_correction_factor:.6g} "
                + " ".join(f"{key}={value}" for key, value in log_values.items())
            )
            self._policy_rank_log(
                "aux_loss_done",
                microbatch_weight=microbatch_weight,
                grad_sum_correction_factor=grad_sum_correction_factor,
                **log_values,
            )
        return aux_loss, world_metrics


PolicyWorker = ray.remote(num_gpus=1)(EchoFSDPPolicyWorkerBase)
