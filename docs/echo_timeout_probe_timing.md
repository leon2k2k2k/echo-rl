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

## No-Signal Timeout Samples

Terminal-agent rollouts can fail before producing any trainable completion,
for example when a command blocks until `agent_timeout`. The generator then
returns a one-token fallback with all training masks set to zero:

- `loss_tokens=0`
- `world_tokens=0`
- `warning_tokens=0`
- `env_tokens=0`
- `nextlat_tokens=0`

On a 4-GPU FSDP update with `micro_train_batch_size_per_gpu=1`, that can assign
one data-parallel rank a completely no-signal row while the other ranks receive
real rows. Jobs `374042` and `374051` showed this pattern: the no-signal rank
finished quickly, the nonzero ranks never logged `forward_backward_done`, and
Ray eventually reported worker `SYSTEM_ERROR`/EOF after about 664s.

We decided not to make distributed backward handle a no-signal rank. Instead,
ECHO replaces true no-signal rows before training with a donor copy of a
nonzero row from the same batch. This preserves SkyRL's fixed
`train_batch_size`/prompt-UID invariant while preventing any rank from receiving
an all-zero fallback sample.

Expected log signal after this fix:

```text
[policy-train] replaced_no_signal_samples count=1 ...
[policy-train] generator_sample_summary batch=4 zero_world_samples=0 no_signal_samples=0
```

This is a pragmatic training-path guard. With longer rollout timeouts, these
fallback rows should be rarer, but the guard remains useful for terminal tasks
where external commands can hang or spend the whole timeout on setup.

## Batch Step Semantics

For the smoke configs, each update step first gathers a complete generated
batch:

- `dataset_max_rows=4`
- `n_samples_per_prompt=1`
- `train_batch_size=4`
- `policy_num_gpus_per_node=4`
- `micro_train_batch_size_per_gpu=1`

So the trainer waits for all 4 terminal trajectories in the batch before
postprocessing, computing logprobs/advantages, and running the policy update.
The slowest trajectory determines the `generate` wall time for that update.
After generation completes, the four training rows are split one per
data-parallel rank for the FSDP step.

## Straggler Mitigation

The current SkyRL/ECHO path is synchronous at the update boundary: it does not
start postprocessing or training from the first completed terminal rollouts
while other rollouts are still running. A fully gradual rollout path would need
an async/replay-buffer style trainer that can keep collecting terminal
trajectories and form train batches from whatever valid samples are ready.

Practical mitigations for now:

- keep `agent_max_concurrency` high enough that many rollouts run in parallel;
- choose `agent_timeout` based on expected task setup time, not just model turn
  time;
- keep per-update `train_batch_size` modest so one straggler blocks fewer
  samples;
- retain the no-signal replacement guard so timed-out fallback rows cannot give
  a rank an all-zero training sample.

For longer runs, the main remaining cost is wall-clock straggling: every update
waits for its slowest rollout. The correctness guard avoids the distributed
training failure, but it does not make rollout collection asynchronous.

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

grep -E "ENTRYPOINT_START|Started: 'sync_weights'|Finished: 'sync_weights'|Started: 'step'|Started: 'generate'|Finished: 'generate'|Started: 'fwd_logprobs_values_reward'|Finished: 'fwd_logprobs_values_reward'|Started: 'compute_advantages_and_returns'|Finished: 'compute_advantages_and_returns'|generator_sample|generator_sample_summary|replaced_no_signal|Started: 'policy_train'|zero_world_tokens|forward_backward_done|Finished: 'policy_train'|ActorDiedError|SYSTEM_ERROR|Traceback|ENTRYPOINT_DONE" \
  logs/echo-rl-smoke-${J}.err logs/echo-rl-smoke-${J}.out | tail -240
```

For live monitoring:

```bash
cd /home/fit/alex/WORK/leon/echo-rl

J=PASTE_JOB_ID

tail -f logs/echo-rl-smoke-${J}.err | grep --line-buffered -E \
  "rollout_done|generator_sample|generator_sample_summary|replaced_no_signal|Started: 'policy_train'|zero_world_tokens|forward_backward_done|Finished: 'policy_train'|ActorDiedError|SYSTEM_ERROR|Traceback"
```
