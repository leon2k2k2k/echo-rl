from types import SimpleNamespace

import torch

from echo_rl.world_modeling.loss import compute_world_model_loss


def test_zero_world_tokens_returns_graph_connected_zero_loss():
    action_log_probs = torch.randn(2, 4, requires_grad=True)
    world_loss_mask = torch.zeros(2, 4)
    config = SimpleNamespace(world_model_coeff=0.05, loss_reduction="token_mean")

    loss, metrics = compute_world_model_loss(action_log_probs, world_loss_mask, config=config)

    assert loss is not None
    assert loss.requires_grad
    assert metrics["world_tokens_selected"].item() == 0
    assert metrics["world_zero_token"].item() == 1

    loss.backward()
    assert action_log_probs.grad is not None
    assert torch.equal(action_log_probs.grad, torch.zeros_like(action_log_probs))


def test_nonzero_world_tokens_reports_nonzero_token_path():
    action_log_probs = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=True)
    world_loss_mask = torch.tensor([[0.0, 1.0, 1.0]])
    config = SimpleNamespace(world_model_coeff=0.1, loss_reduction="token_mean")

    loss, metrics = compute_world_model_loss(action_log_probs, world_loss_mask, config=config)

    assert loss is not None
    assert metrics["world_tokens_selected"].item() == 2
    assert metrics["world_zero_token"].item() == 0
    assert loss.item() > 0
