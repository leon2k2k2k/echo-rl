# ECHO Fast Timeout Probe Timing

This note tracks the expected runtime shape for the fast TMax timeout ECHO
probe:

- launcher: `scripts/submit_echo_tmax_timeout_probe.sh`
- config: `configs/qwen3_8b_rl_echo_fast_timeout_smoke.yaml`
- purpose: reproduce timeout/fallback zero-token samples quickly while keeping
  the 4-GPU ECHO/FSDP/vLLM training shape.

## Baseline From Job 374042

Job `374042` used the same real TMax smoke shape, but with
`agent_timeout=300.0`.

Observed stage timings after the trainer reached the main loop:

| Stage | Start | Finish | Duration |
| --- | --- | --- | --- |
| `sync_weights` | 04:54:19 | 04:54:23 | 4s |
| wait before first batch step | 04:54:23 | 04:55:38 | 75s |
| `generate` | 04:55:38 | 05:00:50 | 312s |
| `fwd_logprobs_values_reward` | 05:00:50 | 05:00:56 | 5s |
| `compute_advantages_and_returns` | 05:00:56 | 05:00:56 | <1s |
| `policy_train` before failure | 05:00:56 | 05:11:59 | 664s |

The long `generate` phase was caused by `trajectory=1_0` timing out during a
slow install command. That produced the zero-token fallback sample.

## Expected Fast Timeout Probe Timing

The fast timeout probe changes only the rollout timeout from 300s to 60s.
Expected timings after model/vLLM initialization:

| Stage | Estimate |
| --- | --- |
| model/vLLM init before `sync_weights` | 5-15 min |
| `sync_weights` | ~5s |
| wait before first batch step | ~1-2 min |
| `generate` | ~60-90s if a trajectory times out |
| fwd logprobs + advantages | ~5-10s |
| `policy_train` | success should be ~20-60s; old failure mode took ~11 min |

Useful signal should appear within 10-20 minutes after the Slurm job starts on
a node.

## Log Command

Use this after submitting a job:

```bash
cd /home/fit/alex/WORK/leon/echo-rl

J=PASTE_JOB_ID

grep -E "ENTRYPOINT_START|Started: 'sync_weights'|Finished: 'sync_weights'|Started: 'step'|Started: 'generate'|Finished: 'generate'|Started: 'fwd_logprobs_values_reward'|Finished: 'fwd_logprobs_values_reward'|Started: 'compute_advantages_and_returns'|Finished: 'compute_advantages_and_returns'|generator_sample|generator_sample_summary|Started: 'policy_train'|zero_world_tokens|forward_backward_done|Finished: 'policy_train'|ActorDiedError|SYSTEM_ERROR|Traceback|ENTRYPOINT_DONE" \
  logs/echo-rl-smoke-${J}.err logs/echo-rl-smoke-${J}.out | tail -240
```

For live monitoring:

```bash
cd /home/fit/alex/WORK/leon/echo-rl

J=PASTE_JOB_ID

tail -f logs/echo-rl-smoke-${J}.err | grep --line-buffered -E \
  "rollout_done|generator_sample|generator_sample_summary|Started: 'policy_train'|zero_world_tokens|forward_backward_done|Finished: 'policy_train'|ActorDiedError|SYSTEM_ERROR|Traceback"
```
