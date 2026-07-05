# Codex Handoff: ECHO + NextLat RL

This repository is the ECHO RL codebase with an experimental NextLat auxiliary loss added on top of the existing SkyRL training path.

## Read First

- Main repo location in this environment: `/home/claude-user/ML/echo-rl`
- Upstream ECHO remote: `git@github.com:microsoft/echo-rl.git`
- This branch adds MSE-only NextLat support for FSDP RL training.
- The implementation is intentionally off by default:
  - `trainer.algorithm.nextlat_coeff: 0.0`
  - `generator.nextlat_loss_target: none`

## What Changed

The ECHO baseline already adds environment-token cross entropy on top of SkyRL RL. This change adds a second optional auxiliary objective:

```text
L_total = L_RL + L_KL - L_entropy + L_ECHO + L_NextLat
```

Current NextLat objective:

```text
pred = dynamics_head(h_t, embed(x_{t+1}))
target = h_{t+1}.detach()
L_NextLat = nextlat_coeff * SmoothL1(pred, target)
```

This is the single-backward, MSE-only version. There is no KL/CE NextLat token-head loss in this implementation.

## Important Files

- `echo_rl/terminal_agent/entrypoint.py`
  - Adds `generator.nextlat_loss_target`.
- `echo_rl/terminal_agent/interaction.py`
  - Adds `completion_nextlat_masks`.
- `echo_rl/terminal_agent/terminal_agent_generator.py`
  - Builds `nextlat_loss_masks` from terminal trajectories.
  - Supported targets:
    - `none`
    - `env_only`
    - `warning_only`
    - `warning_plus_env`
    - `full_observation`
    - `assistant_only`
    - `assistant_plus_env`
    - `all_completion`
- `echo_rl/world_modeling/trainer.py`
  - Carries `nextlat_loss_mask` into SkyRL training batches.
- `echo_rl/world_modeling/nextlat.py`
  - Defines the dynamics head and MSE/SmoothL1 loss.
- `echo_rl/world_modeling/fsdp_worker.py`
  - Installs `nextlat_dynamics` before FSDP wrapping.
  - Requests hidden states only when `nextlat_coeff > 0`.
  - Adds `nextlat_scaled * microbatch_weight` to the auxiliary loss.
- `patches/skyrl_minimal_hooks.patch`
  - Adds the small SkyRL hooks needed for hidden states and auxiliary losses.

## First Smoke Run To Do

Start with a very small RL smoke, not a long benchmark. The purpose is only to prove:

- SkyRL patch applies.
- FSDP initializes `nextlat_dynamics`.
- Hidden states are returned.
- `nextlat_tokens_selected > 0`.
- `nextlat_loss_scaled` is finite.
- Backward succeeds.
- Checkpointing does not crash.

Suggested first settings:

```yaml
trainer:
  algorithm:
    world_model_coeff: 0.05
    nextlat_coeff: 0.01
    nextlat_mtp_horizon: 1
generator:
  nextlat_loss_target: env_only
```

After `env_only` passes, test `assistant_plus_env`. Leave `all_completion` for later because it is the broadest and highest-risk mask.

## Metrics To Watch

In the training logs / WandB, verify these are present and sane:

- `status/nextlat_coeff`
- `nextlat_loss_scaled`
- `nextlat_smooth_l1`
- `nextlat_tokens_selected`
- `nextlat_policy_loss_ratio`
- `nextlat_trainable_params`
- `nextlat_param_norm`
- `nextlat_grad_norm_pre_backward`

Note: `nextlat_grad_norm_pre_backward` is logged before the current backward call. Treat it as a weak diagnostic only. The stronger checks are nonzero trainable params, finite param norm, nonzero selected tokens, finite loss, and successful backward.

## Known Risks

- HF export may not preserve `nextlat_dynamics` as a standalone reloadable head. FSDP checkpoints are more likely to include it, but this should be explicitly tested before long runs.
- `nextlat_trainable_params` may be local/sharded under FSDP, so use it as a nonzero smoke signal rather than an exact global count.
- Mask alignment depends on chat-template token deltas. Smoke should log `nextlat_tokens_selected` for the selected mask mode.
- Sequence parallel is intentionally blocked for NextLat MSE right now; the worker raises if `sequence_parallel_size != 1`.

## Validation Already Run

From `/home/claude-user/ML/echo-rl`:

```bash
python3 -m compileall -q echo_rl
git diff --check -- . ':(exclude)patches/skyrl_minimal_hooks.patch'
```

Both passed at handoff time.

