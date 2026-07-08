from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except ImportError:  # pragma: no cover - FSDP is optional in local unit tests.
    FSDP = None


@dataclass
class NextLatRLConfig:
    coeff: float = 0.0
    base_coeff: Optional[float] = None
    lambda_mse: float = 1.0
    lambda_kl: float = 0.0
    lambda_ce: float = 0.0
    mtp_horizon: int = 1
    proj_factor: float = 1.0
    bias: bool = False
    norm_eps: float = 1e-6
    logit_temperature: float = 1.0
    token_loss_chunk_size: int = 32


class BiasOptionalLayerNorm(nn.Module):
    def __init__(self, hidden_size: int, *, bias: bool, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return F.layer_norm(input_tensor, self.weight.shape, self.weight, self.bias, self.eps)
        return F.rms_norm(input_tensor, self.weight.shape, self.weight, self.eps)


class QwenStyleSwiGLUMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, bias: bool):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class NextLatDynamicsModel(nn.Module):
    """Predict h_{t+1} from h_t and embed(x_{t+1})."""

    def __init__(self, hidden_size: int, config: NextLatRLConfig):
        super().__init__()
        input_dim = hidden_size * 2
        hidden_dim = int(config.proj_factor * input_dim)
        hidden_dim = max(128, 128 * round(hidden_dim / 128))

        self.norm_x = BiasOptionalLayerNorm(input_dim, bias=config.bias, eps=config.norm_eps)
        self.mlp = QwenStyleSwiGLUMLP(input_dim, hidden_dim, hidden_size, bias=config.bias)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, current_states: torch.Tensor, next_token_embeds: torch.Tensor) -> torch.Tensor:
        param = next(self.parameters())
        current_states = current_states.to(device=param.device, dtype=param.dtype)
        next_token_embeds = next_token_embeds.to(device=param.device, dtype=param.dtype)
        x = torch.cat([next_token_embeds, current_states], dim=-1)
        delta = self.mlp(self.norm_x(x))
        return delta + current_states


def build_nextlat_token_mask(
    nextlat_loss_mask: torch.Tensor,
    *,
    sequence_length: int,
    num_actions: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(nextlat_loss_mask.shape[0], sequence_length, device=device, dtype=torch.bool)
    action_mask = nextlat_loss_mask.to(device=device).bool()
    if action_mask.shape[1] != num_actions:
        action_mask = action_mask[:, -num_actions:]
    mask[:, -num_actions:] = action_mask
    return mask


def align_nextlat_token_mask(nextlat_token_mask: torch.Tensor, *, sequence_length: int) -> torch.Tensor:
    """Right-align a padded sequence mask to the model output sequence length."""
    if nextlat_token_mask.shape[1] == sequence_length:
        return nextlat_token_mask
    if nextlat_token_mask.shape[1] > sequence_length:
        return nextlat_token_mask[:, -sequence_length:]
    pad_width = sequence_length - nextlat_token_mask.shape[1]
    return F.pad(nextlat_token_mask, (pad_width, 0), value=False)


def align_token_ids(token_ids: torch.Tensor, *, sequence_length: int, pad_value: int = 0) -> torch.Tensor:
    """Right-align token ids to match padded model outputs."""
    if token_ids.shape[1] == sequence_length:
        return token_ids
    if token_ids.shape[1] > sequence_length:
        return token_ids[:, -sequence_length:]
    pad_width = sequence_length - token_ids.shape[1]
    return F.pad(token_ids, (pad_width, 0), value=pad_value)


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if not (targets != -100).any():
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )


def categorical_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    token_pred_mask: torch.Tensor,
) -> torch.Tensor:
    if not token_pred_mask.any():
        return student_logits.sum() * 0.0
    log_teacher = F.log_softmax(teacher_logits, dim=-1)
    log_student = F.log_softmax(student_logits, dim=-1)
    kl_pointwise = F.kl_div(log_student, log_teacher, log_target=True, reduction="none")
    kl_per_token = kl_pointwise.sum(dim=-1)
    mask = token_pred_mask.to(dtype=kl_per_token.dtype)
    return (kl_per_token * mask).sum() / mask.sum().clamp_min(1.0)


def _materialize_detached_tensor(tensor: torch.Tensor, *, clone: bool) -> torch.Tensor:
    tensor = tensor.detach()
    full_tensor = getattr(tensor, "full_tensor", None)
    if callable(full_tensor):
        tensor = full_tensor().detach()
    if clone:
        tensor = tensor.clone()
    return tensor


def _lm_head_snapshot_detached(
    lm_head: nn.Module,
    *,
    fsdp_model: nn.Module | None = None,
    expected_vocab_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    def snapshot_current_weight(clone: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight = _materialize_detached_tensor(lm_head.weight, clone=clone)
        bias = getattr(lm_head, "bias", None)
        if bias is not None:
            bias = _materialize_detached_tensor(bias, clone=clone)
        return weight, bias

    if fsdp_model is not None:
        if FSDP is None:
            raise RuntimeError("FSDP auxiliary LM-head projection requires torch.distributed.fsdp.")
        with FSDP.summon_full_params(fsdp_model, recurse=False, writeback=False):
            weight, bias = snapshot_current_weight(clone=True)
    else:
        weight, bias = snapshot_current_weight(clone=False)

    if weight.dim() != 2 or (expected_vocab_size is not None and weight.size(0) != expected_vocab_size):
        raise RuntimeError(
            "Auxiliary LM-head projection did not materialize the full output weight. "
            "Check the FSDP wrapping policy for the output embedding."
        )
    return weight, bias


def _lm_head_linear_detached(
    lm_head: nn.Module,
    hidden_states: torch.Tensor,
    *,
    fsdp_model: nn.Module | None = None,
    expected_vocab_size: int | None = None,
) -> torch.Tensor:
    weight, bias = _lm_head_snapshot_detached(
        lm_head,
        fsdp_model=fsdp_model,
        expected_vocab_size=expected_vocab_size,
    )
    return F.linear(hidden_states.to(device=weight.device, dtype=weight.dtype), weight, bias)


def _chunked_lm_head_token_losses(
    *,
    lm_head: nn.Module,
    hidden_states: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_tokens: torch.Tensor,
    token_pred_mask: torch.Tensor,
    fsdp_model: nn.Module | None,
    logit_temperature: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.shape[:2] != teacher_logits.shape[:2] or hidden_states.shape[:2] != target_tokens.shape[:2]:
        raise RuntimeError(
            "NextLat KL/CE tensor alignment mismatch: "
            f"hidden={tuple(hidden_states.shape[:2])}, "
            f"teacher={tuple(teacher_logits.shape[:2])}, "
            f"targets={tuple(target_tokens.shape[:2])}"
        )
    if hidden_states.shape[:2] != token_pred_mask.shape[:2]:
        raise RuntimeError(
            "NextLat KL/CE mask alignment mismatch: "
            f"hidden={tuple(hidden_states.shape[:2])}, mask={tuple(token_pred_mask.shape[:2])}"
        )
    if not token_pred_mask.any():
        zero = hidden_states.sum() * 0.0
        return zero, zero

    weight, bias = _lm_head_snapshot_detached(
        lm_head,
        fsdp_model=fsdp_model,
        expected_vocab_size=teacher_logits.size(-1),
    )
    selected_indices = token_pred_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1]).detach()
    flat_targets = target_tokens.reshape(-1)
    total_tokens = selected_indices.numel()
    chunk_size = max(1, int(chunk_size or 32))
    total_kl = hidden_states.sum() * 0.0
    total_ce = hidden_states.sum() * 0.0
    temperature = max(float(logit_temperature or 1.0), 1e-6)

    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
        chunk_indices = selected_indices[start:end]
        hidden_chunk = flat_hidden.index_select(0, chunk_indices).to(device=weight.device, dtype=weight.dtype)
        teacher_chunk = flat_teacher.index_select(0, chunk_indices).to(device=weight.device, dtype=weight.dtype)
        target_chunk = flat_targets.index_select(0, chunk_indices).to(device=weight.device)
        logits = F.linear(hidden_chunk, weight, bias)
        teacher_chunk = teacher_chunk / temperature
        student_chunk = logits / temperature
        log_teacher = F.log_softmax(teacher_chunk, dim=-1)
        log_student = F.log_softmax(student_chunk, dim=-1)
        total_kl = total_kl + F.kl_div(log_student, log_teacher, log_target=True, reduction="sum")
        total_ce = total_ce + F.cross_entropy(logits, target_chunk, reduction="sum")

    denom = torch.tensor(float(total_tokens), device=weight.device, dtype=total_kl.dtype)
    kl = (total_kl / denom) * (temperature * temperature)
    token_ce = total_ce / denom.to(dtype=total_ce.dtype)
    return kl, token_ce


def _get_output_logits(model_output: Any) -> torch.Tensor:
    logits = getattr(model_output, "logits", None)
    if logits is None and isinstance(model_output, dict):
        logits = model_output.get("logits")
    if logits is None:
        raise RuntimeError("NextLat KL/CE requires logits from the same policy forward.")
    return logits


def compute_nextlat_mse_loss(
    dynamics_model: NextLatDynamicsModel,
    *,
    model_output: Any,
    nextlat_token_mask: torch.Tensor,
    config: NextLatRLConfig,
    base_reward_gate: Optional[torch.Tensor] = None,
    lm_head: Optional[nn.Module] = None,
    sequences: Optional[torch.Tensor] = None,
    fsdp_model: Optional[nn.Module] = None,
) -> Tuple[Optional[torch.Tensor], dict[str, torch.Tensor]]:
    if config.coeff <= 0:
        return None, {}
    hidden_states_tuple = getattr(model_output, "hidden_states", None)
    if hidden_states_tuple is None and isinstance(model_output, dict):
        hidden_states_tuple = model_output.get("hidden_states")
    if not hidden_states_tuple:
        raise RuntimeError("NextLat MSE requires model hidden_states from the same policy forward.")

    input_embeds = hidden_states_tuple[0]
    hidden_states = hidden_states_tuple[-1]
    if input_embeds.shape[:2] != hidden_states.shape[:2]:
        raise RuntimeError(
            f"NextLat hidden/input shape mismatch: {tuple(hidden_states.shape)} vs {tuple(input_embeds.shape)}"
        )

    mse_mask = nextlat_token_mask.to(device=hidden_states.device, dtype=torch.bool)
    if mse_mask.shape[0] != hidden_states.shape[0]:
        raise RuntimeError(
            f"NextLat mask batch mismatch: {tuple(mse_mask.shape)} vs hidden {tuple(hidden_states.shape)}"
        )
    mse_mask = align_nextlat_token_mask(mse_mask, sequence_length=hidden_states.shape[1])
    selected = mse_mask.sum()
    if selected <= 0:
        zero = hidden_states.sum() * 0.0
        return zero, {
            "nextlat_loss_unscaled": zero.detach(),
            "nextlat_loss_scaled": zero.detach(),
            "nextlat_tokens_selected": selected.detach(),
            "nextlat_smooth_l1": zero.detach(),
        }

    base_coeff = config.coeff if config.base_coeff is None else float(config.base_coeff)
    if config.coeff <= 0:
        base_grad_scale = 0.0
    else:
        base_grad_scale = base_coeff / config.coeff
    base_grad_scale_tensor: torch.Tensor | float = base_grad_scale
    reward_gate_mean = reward_gate_min = reward_gate_max = None
    if base_reward_gate is not None:
        gate = base_reward_gate.to(device=hidden_states.device, dtype=hidden_states.dtype).detach().clamp(0.0, 1.0)
        if gate.ndim != 1 or gate.shape[0] != hidden_states.shape[0]:
            raise RuntimeError(
                f"NextLat reward gate must have shape ({hidden_states.shape[0]},), got {tuple(gate.shape)}"
            )
        reward_gate_mean = gate.float().mean()
        reward_gate_min = gate.float().min()
        reward_gate_max = gate.float().max()
        base_grad_scale_tensor = gate.view(-1, 1, 1) * base_grad_scale

    next_states = hidden_states
    if isinstance(base_grad_scale_tensor, float) and base_grad_scale_tensor == 1.0:
        hidden_states_for_aux = hidden_states
        input_embeds_for_aux = input_embeds
    else:
        hidden_states_for_aux = hidden_states.detach() + base_grad_scale_tensor * (hidden_states - hidden_states.detach())
        input_embeds_for_aux = input_embeds.detach() + base_grad_scale_tensor * (input_embeds - input_embeds.detach())
    pred_next_states = hidden_states_for_aux
    next_tokens = input_embeds_for_aux
    total_smooth_l1 = torch.zeros((), device=hidden_states.device)
    total_kl = torch.zeros((), device=hidden_states.device)
    token_ce_losses = []
    compute_token_losses = config.lambda_kl != 0.0 or config.lambda_ce != 0.0
    if compute_token_losses:
        if lm_head is None or sequences is None:
            raise RuntimeError("NextLat KL/CE requires lm_head and sequences.")
        full_teacher_logits = _get_output_logits(model_output)
        if full_teacher_logits.shape[:2] != hidden_states.shape[:2]:
            raise RuntimeError(
                "NextLat teacher logits must align with hidden states, "
                f"got {tuple(full_teacher_logits.shape[:2])} and {tuple(hidden_states.shape[:2])}"
            )
        teacher_logits = full_teacher_logits[:, :-1]
        aligned_sequences = align_token_ids(
            sequences.to(device=hidden_states.device),
            sequence_length=hidden_states.shape[1],
            pad_value=0,
        )
        target_tokens = aligned_sequences[:, 1:]
    else:
        teacher_logits = None
        target_tokens = None

    for _ in range(config.mtp_horizon):
        pred_next_states = pred_next_states[:, :-1]
        next_tokens = next_tokens[:, 1:]
        next_states = next_states[:, 1:]
        mse_mask = mse_mask[:, 1:]
        if compute_token_losses:
            assert target_tokens is not None and teacher_logits is not None
            target_tokens = target_tokens[:, 1:]
            teacher_logits = teacher_logits[:, 1:]

        pred_next_states = dynamics_model(pred_next_states, next_tokens)
        target_states = next_states.detach().to(device=pred_next_states.device, dtype=pred_next_states.dtype)
        smooth_l1_elem = F.smooth_l1_loss(pred_next_states, target_states, reduction="none")
        weight = mse_mask.unsqueeze(-1).to(dtype=smooth_l1_elem.dtype)
        denom = weight.expand_as(smooth_l1_elem).sum().clamp_min(1.0)
        total_smooth_l1 = total_smooth_l1 + (smooth_l1_elem * weight).sum() / denom
        if compute_token_losses:
            token_pred_mask = mse_mask[:, 1:].to(device=pred_next_states.device, dtype=torch.bool)
            kl, token_ce = _chunked_lm_head_token_losses(
                lm_head=lm_head,
                hidden_states=pred_next_states[:, :-1],
                teacher_logits=teacher_logits,
                target_tokens=target_tokens,
                token_pred_mask=token_pred_mask,
                fsdp_model=fsdp_model,
                logit_temperature=config.logit_temperature,
                chunk_size=config.token_loss_chunk_size,
            )
            total_kl = total_kl + kl
            token_ce_losses.append(token_ce)

    smooth_l1 = total_smooth_l1 / config.mtp_horizon
    kl_loss = total_kl / config.mtp_horizon
    if token_ce_losses:
        token_ce_loss = torch.stack(token_ce_losses).mean()
    else:
        token_ce_loss = smooth_l1.sum() * 0.0
    unscaled = (
        float(config.lambda_mse) * smooth_l1
        + float(config.lambda_kl) * kl_loss
        + float(config.lambda_ce) * token_ce_loss
    )
    scaled = config.coeff * unscaled
    return scaled, {
        "nextlat_loss_unscaled": unscaled,
        "nextlat_loss_scaled": scaled,
        "nextlat_base_coeff": torch.tensor(base_coeff, device=hidden_states.device),
        "nextlat_base_grad_scale": torch.as_tensor(base_grad_scale_tensor, device=hidden_states.device).float().mean(),
        "nextlat_base_grad_scale_min": torch.as_tensor(base_grad_scale_tensor, device=hidden_states.device).float().min(),
        "nextlat_base_grad_scale_max": torch.as_tensor(base_grad_scale_tensor, device=hidden_states.device).float().max(),
        "nextlat_base_reward_gate_mean": reward_gate_mean
        if reward_gate_mean is not None
        else torch.tensor(1.0, device=hidden_states.device),
        "nextlat_base_reward_gate_min": reward_gate_min
        if reward_gate_min is not None
        else torch.tensor(1.0, device=hidden_states.device),
        "nextlat_base_reward_gate_max": reward_gate_max
        if reward_gate_max is not None
        else torch.tensor(1.0, device=hidden_states.device),
        "nextlat_tokens_selected": selected,
        "nextlat_smooth_l1": smooth_l1,
        "nextlat_kl_loss": kl_loss,
        "nextlat_token_ce": token_ce_loss,
    }
