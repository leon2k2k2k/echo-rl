from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class NextLatRLConfig:
    coeff: float = 0.0
    mtp_horizon: int = 1
    proj_factor: float = 1.0
    bias: bool = False
    norm_eps: float = 1e-5


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


class NextLatDynamicsModel(nn.Module):
    """Predict h_{t+1} from h_t and embed(x_{t+1})."""

    def __init__(self, hidden_size: int, config: NextLatRLConfig):
        super().__init__()
        input_dim = hidden_size * 2
        hidden_dim = int(config.proj_factor * input_dim)
        hidden_dim = max(128, 128 * round(hidden_dim / 128))

        self.norm_x = BiasOptionalLayerNorm(input_dim, bias=config.bias, eps=config.norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_size, bias=config.bias),
        )
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


def compute_nextlat_mse_loss(
    dynamics_model: NextLatDynamicsModel,
    *,
    model_output: Any,
    nextlat_token_mask: torch.Tensor,
    config: NextLatRLConfig,
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

    next_states = hidden_states
    pred_next_states = hidden_states
    next_tokens = input_embeds
    total_smooth_l1 = torch.zeros((), device=hidden_states.device)

    for _ in range(config.mtp_horizon):
        pred_next_states = pred_next_states[:, :-1]
        next_tokens = next_tokens[:, 1:]
        next_states = next_states[:, 1:]
        mse_mask = mse_mask[:, 1:]

        pred_next_states = dynamics_model(pred_next_states, next_tokens)
        target_states = next_states.detach().to(device=pred_next_states.device, dtype=pred_next_states.dtype)
        smooth_l1_elem = F.smooth_l1_loss(pred_next_states, target_states, reduction="none")
        weight = mse_mask.unsqueeze(-1).to(dtype=smooth_l1_elem.dtype)
        denom = weight.expand_as(smooth_l1_elem).sum().clamp_min(1.0)
        total_smooth_l1 = total_smooth_l1 + (smooth_l1_elem * weight).sum() / denom

    smooth_l1 = total_smooth_l1 / config.mtp_horizon
    scaled = config.coeff * smooth_l1
    return scaled, {
        "nextlat_loss_unscaled": smooth_l1,
        "nextlat_loss_scaled": scaled,
        "nextlat_tokens_selected": selected,
        "nextlat_smooth_l1": smooth_l1,
    }
