from types import SimpleNamespace

import pytest
import torch

from echo_rl.world_modeling.nextlat import (
    NextLatDynamicsModel,
    NextLatRLConfig,
    _lm_head_linear_detached,
    align_nextlat_token_mask,
    compute_nextlat_mse_loss,
)


def test_align_nextlat_token_mask_right_trims_padding():
    mask = torch.tensor([[False, False, False, True, True, False]])

    aligned = align_nextlat_token_mask(mask, sequence_length=3)

    assert aligned.tolist() == [[True, True, False]]


def test_nextlat_loss_aligns_padded_mask_to_hidden_sequence_length():
    cfg = NextLatRLConfig(coeff=0.05, mtp_horizon=1, proj_factor=1.0)
    dynamics = NextLatDynamicsModel(hidden_size=4, config=cfg)
    input_embeds = torch.randn(1, 5, 4)
    hidden_states = torch.randn(1, 5, 4, requires_grad=True)
    padded_mask = torch.tensor([[False, False, False, True, True, True, True, True]])
    model_output = SimpleNamespace(hidden_states=(input_embeds, hidden_states))

    loss, metrics = compute_nextlat_mse_loss(
        dynamics,
        model_output=model_output,
        nextlat_token_mask=padded_mask,
        config=cfg,
    )

    assert loss is not None
    assert loss.requires_grad
    assert metrics["nextlat_tokens_selected"].item() == 5
    loss.backward()
    assert hidden_states.grad is not None


def test_nextlat_base_coeff_scales_only_base_gradient():
    torch.manual_seed(0)
    cfg_full = NextLatRLConfig(coeff=0.05, mtp_horizon=1, proj_factor=1.0)
    cfg_damped = NextLatRLConfig(coeff=0.05, base_coeff=0.0125, mtp_horizon=1, proj_factor=1.0)
    dynamics_full = NextLatDynamicsModel(hidden_size=4, config=cfg_full)
    dynamics_damped = NextLatDynamicsModel(hidden_size=4, config=cfg_damped)
    dynamics_damped.load_state_dict(dynamics_full.state_dict())

    input_full = torch.randn(1, 5, 4, requires_grad=True)
    hidden_full = torch.randn(1, 5, 4, requires_grad=True)
    input_damped = input_full.detach().clone().requires_grad_(True)
    hidden_damped = hidden_full.detach().clone().requires_grad_(True)
    mask = torch.ones(1, 5, dtype=torch.bool)

    loss_full, _ = compute_nextlat_mse_loss(
        dynamics_full,
        model_output=SimpleNamespace(hidden_states=(input_full, hidden_full)),
        nextlat_token_mask=mask,
        config=cfg_full,
    )
    loss_damped, metrics = compute_nextlat_mse_loss(
        dynamics_damped,
        model_output=SimpleNamespace(hidden_states=(input_damped, hidden_damped)),
        nextlat_token_mask=mask,
        config=cfg_damped,
    )

    loss_full.backward()
    loss_damped.backward()

    assert metrics["nextlat_base_grad_scale"].item() == pytest.approx(0.25)
    assert torch.allclose(hidden_damped.grad, hidden_full.grad * 0.25, atol=1e-6, rtol=1e-5)
    assert torch.allclose(input_damped.grad, input_full.grad * 0.25, atol=1e-6, rtol=1e-5)

    full_head_grad = next(dynamics_full.parameters()).grad
    damped_head_grad = next(dynamics_damped.parameters()).grad
    assert torch.allclose(damped_head_grad, full_head_grad, atol=1e-6, rtol=1e-5)


def test_nextlat_reward_gate_keeps_base_gradient_only_for_correct_rows():
    torch.manual_seed(0)
    cfg = NextLatRLConfig(coeff=0.05, mtp_horizon=1, proj_factor=1.0)
    dynamics_full = NextLatDynamicsModel(hidden_size=4, config=cfg)
    dynamics_gated = NextLatDynamicsModel(hidden_size=4, config=cfg)
    dynamics_gated.load_state_dict(dynamics_full.state_dict())

    input_full = torch.randn(2, 5, 4, requires_grad=True)
    hidden_full = torch.randn(2, 5, 4, requires_grad=True)
    input_gated = input_full.detach().clone().requires_grad_(True)
    hidden_gated = hidden_full.detach().clone().requires_grad_(True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    reward_gate = torch.tensor([1.0, 0.0])

    loss_full, _ = compute_nextlat_mse_loss(
        dynamics_full,
        model_output=SimpleNamespace(hidden_states=(input_full, hidden_full)),
        nextlat_token_mask=mask,
        config=cfg,
    )
    loss_gated, metrics = compute_nextlat_mse_loss(
        dynamics_gated,
        model_output=SimpleNamespace(hidden_states=(input_gated, hidden_gated)),
        nextlat_token_mask=mask,
        config=cfg,
        base_reward_gate=reward_gate,
    )

    loss_full.backward()
    loss_gated.backward()

    assert metrics["nextlat_base_grad_scale"].item() == pytest.approx(0.5)
    assert metrics["nextlat_base_grad_scale_min"].item() == pytest.approx(0.0)
    assert metrics["nextlat_base_grad_scale_max"].item() == pytest.approx(1.0)
    assert metrics["nextlat_base_reward_gate_mean"].item() == pytest.approx(0.5)
    assert torch.allclose(hidden_gated.grad[0], hidden_full.grad[0], atol=1e-6, rtol=1e-5)
    assert torch.allclose(input_gated.grad[0], input_full.grad[0], atol=1e-6, rtol=1e-5)
    assert torch.allclose(hidden_gated.grad[1], torch.zeros_like(hidden_full.grad[1]), atol=1e-6, rtol=1e-5)
    assert torch.allclose(input_gated.grad[1], torch.zeros_like(input_full.grad[1]), atol=1e-6, rtol=1e-5)

    full_head_grad = next(dynamics_full.parameters()).grad
    gated_head_grad = next(dynamics_gated.parameters()).grad
    assert torch.allclose(gated_head_grad, full_head_grad, atol=1e-6, rtol=1e-5)


def test_nextlat_reward_gate_does_not_scale_policy_gradient():
    torch.manual_seed(0)
    cfg = NextLatRLConfig(coeff=0.05, mtp_horizon=1, proj_factor=1.0)
    dynamics = NextLatDynamicsModel(hidden_size=4, config=cfg)
    lm_head = torch.nn.Linear(4, 7, bias=False)
    mask = torch.ones(2, 5, dtype=torch.bool)
    reward_gate = torch.zeros(2)

    input_policy = torch.randn(2, 5, 4, requires_grad=True)
    hidden_policy = torch.randn(2, 5, 4, requires_grad=True)
    policy_logits = lm_head(hidden_policy)
    policy_loss = policy_logits.sum()
    policy_loss.backward()
    expected_hidden_grad = hidden_policy.grad.detach().clone()
    expected_input_grad = input_policy.grad

    input_combined = input_policy.detach().clone().requires_grad_(True)
    hidden_combined = hidden_policy.detach().clone().requires_grad_(True)
    combined_logits = lm_head(hidden_combined)
    combined_policy_loss = combined_logits.sum()
    nextlat_loss, _ = compute_nextlat_mse_loss(
        dynamics,
        model_output=SimpleNamespace(hidden_states=(input_combined, hidden_combined)),
        nextlat_token_mask=mask,
        config=cfg,
        base_reward_gate=reward_gate,
    )
    (combined_policy_loss + nextlat_loss).backward()

    assert torch.allclose(hidden_combined.grad, expected_hidden_grad, atol=1e-6, rtol=1e-5)
    assert input_combined.grad is not None
    assert torch.allclose(input_combined.grad, torch.zeros_like(input_combined.grad), atol=1e-6, rtol=1e-5)
    assert expected_input_grad is None
    assert next(dynamics.parameters()).grad is not None


def test_nextlat_kl_uses_frozen_lm_head_and_keeps_single_graph():
    torch.manual_seed(0)
    cfg = NextLatRLConfig(
        coeff=0.05,
        lambda_mse=1.0,
        lambda_kl=1.0,
        lambda_ce=0.5,
        mtp_horizon=1,
        proj_factor=1.0,
    )
    dynamics = NextLatDynamicsModel(hidden_size=4, config=cfg)
    lm_head = torch.nn.Linear(4, 7, bias=False)

    input_embeds = torch.randn(2, 5, 4, requires_grad=True)
    hidden_states = torch.randn(2, 5, 4, requires_grad=True)
    teacher_logits = torch.randn(2, 5, 7, requires_grad=True)
    sequences = torch.randint(0, 7, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.bool)
    model_output = SimpleNamespace(
        hidden_states=(input_embeds, hidden_states),
        logits=teacher_logits,
    )

    loss, metrics = compute_nextlat_mse_loss(
        dynamics,
        model_output=model_output,
        nextlat_token_mask=mask,
        config=cfg,
        lm_head=lm_head,
        sequences=sequences,
    )

    assert loss is not None
    assert loss.requires_grad
    assert metrics["nextlat_smooth_l1"].item() > 0.0
    assert metrics["nextlat_kl_loss"].item() >= 0.0
    assert metrics["nextlat_token_ce"].item() > 0.0

    loss.backward()
    assert hidden_states.grad is not None
    assert input_embeds.grad is not None
    assert next(dynamics.parameters()).grad is not None
    assert lm_head.weight.grad is None
    assert teacher_logits.grad is None


def test_lm_head_projection_materializes_full_tensor_snapshot():
    class FakeDTensor:
        def __init__(self, local: torch.Tensor, full: torch.Tensor):
            self.local = local
            self.full = full

        def detach(self):
            return self

        def full_tensor(self):
            return self.full

    full_weight = torch.randn(7, 4)
    local_weight = torch.empty(0)
    lm_head = SimpleNamespace(weight=FakeDTensor(local_weight, full_weight), bias=None)
    hidden = torch.randn(2, 3, 4)

    logits = _lm_head_linear_detached(lm_head, hidden, expected_vocab_size=7)

    assert torch.allclose(logits, torch.nn.functional.linear(hidden, full_weight))
