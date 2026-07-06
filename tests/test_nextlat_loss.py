from types import SimpleNamespace

import torch

from echo_rl.world_modeling.nextlat import (
    NextLatDynamicsModel,
    NextLatRLConfig,
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
