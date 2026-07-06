# Terminal-Agent Async Rollout Handoff

This branch is for investigating async rollout collection for ECHO terminal-agent
training. The immediate goal is not to replace the working synchronous smokes.
The immediate goal is to map and prototype the lowest-risk path to reuse
SkyRL's existing `FullyAsyncRayPPOTrainer` with ECHO's terminal-agent generator,
world-model loss, and NextLat loss.

## Why This Branch Exists

Terminal-agent rollout durations are heavy-tailed. In the synchronous trainer,
one update waits for every trajectory in the batch before postprocessing and
training. In the fast TMax probe, three trajectories finished in about 16-21s,
while one timeout trajectory took about 71s. The whole update's `generate`
stage therefore took about 72s.

For full runs with longer timeouts and more turns, this synchronous barrier can
dominate wall time. Async rollout collection should let completed trajectories
fill a training buffer while slow terminal jobs continue or eventually time out.

## Current Known-Good Sync Work

The current `main` branch has been focused on making all synchronous modes work:

- ECHO-only
- NextLat-only
- ECHO + NextLat

Relevant recent fixes:

- `cb0e2fb` replaces no-signal timeout samples with donor rows before training.
- `34498e5` keeps the NextLat head local to the policy worker to avoid
  FSDP2/DTensor/vLLM sync problems.
- `f4612af` right-aligns NextLat masks to the actual model hidden-state length.
- `1394477` mirrors scheduler state when adding the local NextLat optimizer
  param group.

The no-signal replacement should remain in async mode. Async rollout reduces
straggler waiting, but terminal rollouts can still time out and produce
all-zero training masks.

## Existing Synchronous Shape

For the fast TMax probes:

- `dataset_max_rows=4`
- `n_samples_per_prompt=1`
- `train_batch_size=4`
- `policy_num_gpus_per_node=4`
- `micro_train_batch_size_per_gpu=1`
- `agent_timeout=60.0`

The synchronous update waits for all 4 terminal trajectories, postprocesses the
whole `GeneratorOutput`, computes logprobs/advantages, and then runs one FSDP
update.

See `docs/echo_timeout_probe_timing.md` for the current smoke timing notes.

## Upstream Async Path To Reuse

SkyRL already has a fully async trainer:

- `/home/claude-user/ML/SkyRL/skyrl/train/fully_async_trainer.py`
- `/home/claude-user/ML/SkyRL/examples/train/fully_async/main_fully_async.py`
- `/home/claude-user/ML/SkyRL/examples/train/fully_async/fully_async_run_gsm8k.sh`

Important constraints from `FullyAsyncRayPPOTrainer`:

- `trainer.train_batch_size == trainer.policy_mini_batch_size`
- `generator.batched == false`
- `generator.inference_engine.async_engine == true`
- `trainer.placement.colocate_all == false`
- generation workers call `generator.generate()` on one prompt group at a time
- completed groups enter a FIFO buffer and are consumed in mini-batches
- staleness is controlled by:
  - `trainer.fully_async.max_staleness_steps`
  - `trainer.fully_async.num_parallel_generation_workers`

This is a good match for terminal-agent straggler mitigation, but it requires a
new ECHO async entrypoint/config because the current ECHO entrypoint uses
`EchoPPOTrainer`.

## ECHO Integration Points

Current ECHO entrypoint:

- `echo_rl/terminal_agent/entrypoint.py`
- `EchoTerminalAgentExp.get_worker_classes()`
- `EchoTerminalAgentExp.get_generator()`
- `EchoTerminalAgentExp.get_trainer()`
- `EchoTerminalAgentExp.get_train_dataset()`

The likely async entrypoint should be a sibling, not a replacement:

- keep `EchoTerminalAgentSkyRLConfig`
- keep `TerminalAgentGenerator`
- keep `TerminalAgentTaskDataset`
- keep ECHO's `PolicyWorker`, `CriticWorker`, and `RefWorker`
- swap trainer class to a new ECHO async trainer derived from
  `FullyAsyncRayPPOTrainer`

The async trainer should preserve ECHO's overrides from
`echo_rl/world_modeling/trainer.py`, especially:

- `_log_generator_sample_summaries`
- `postprocess_generator_output`
- `_replace_no_signal_samples`
- `convert_to_training_input`
- `train_critic_and_policy` logging

The cleanest code shape may be a mixin for ECHO generator-output handling, then:

- `EchoPPOTrainer(EchoWorldModelingMixin, RayPPOTrainer)`
- `EchoFullyAsyncPPOTrainer(EchoWorldModelingMixin, FullyAsyncRayPPOTrainer)`

Avoid copying the whole trainer if possible.

## Config Sketch

Start with a small async smoke config derived from the combined fast TMax probe:

- base: `configs/qwen3_8b_rl_echo_nextlat_fast_timeout_smoke.yaml`
- set `trainer.placement.colocate_all=false`
- set `generator.batched=false`
- keep `generator.inference_engine.async_engine=true`
- keep `trainer.train_batch_size == trainer.policy_mini_batch_size`
- add:
  - `trainer.fully_async.max_staleness_steps`
  - `trainer.fully_async.num_parallel_generation_workers`

A first smoke should use conservative numbers, for example:

- `trainer.train_batch_size=4`
- `trainer.policy_mini_batch_size=4`
- `trainer.fully_async.max_staleness_steps=1`
- `trainer.fully_async.num_parallel_generation_workers=4` or `8`

After the first pass works, increase generation workers and dataset rows to see
whether straggler time is hidden.

## Known Risks

1. `generator.batched=false` may require config support in
   `TerminalAgentGeneratorConfig` if inherited defaults are not enough.

2. Fully async uses one-prompt groups. ECHO's `TerminalAgentGenerator.generate`
   should already handle one prompt, but transcript IDs and rollout metadata
   should be checked carefully.

3. The current no-signal replacement chooses donor rows within the training
   mini-batch. That still works conceptually, but async mini-batches are
   assembled by finish-time FIFO, not original prompt order.

4. Fully async marks UIDs consumed after training. If no-signal replacement
   duplicates a donor row, the consumed UID set still corresponds to the
   original generated groups, not the donor trajectory metadata. This is probably
   acceptable for a smoke, but should be documented in logs.

5. ECHO's current `get_train_dataset()` asserts dataset length is at least
   `train_batch_size`. Fully async also requires dataloader length divisible by
   mini-batch size. For real TMax, use enough rows before increasing worker
   count.

6. Cluster constraints remain: `g08` has working Docker access; other nodes may
   fail early on Docker permissions. Async testing should start on `g08`.

## Suggested First Tasks

1. Refactor ECHO-specific trainer logic into a small mixin while keeping current
   sync tests and smokes intact.

2. Add `echo_rl/terminal_agent/async_entrypoint.py` modelled on SkyRL's
   `examples/train/fully_async/main_fully_async.py`.

3. Add a minimal async combined config using the fast TMax probe shape.

4. Add a submit script that mirrors
   `scripts/submit_echo_nextlat_tmax_timeout_probe.sh`, but points at the async
   entrypoint/config.

5. Run one g08 smoke and check for:
   - generation buffer progress
   - `async/staleness_*` metrics
   - `replaced_no_signal_samples` still firing when timeout rows appear
   - nonzero `world_tokens` and `nextlat_tokens`
   - `Finished: 'policy_train'`
   - checkpoint save and `Training done!`

## What Not To Do Yet

- Do not remove or rewrite the synchronous smokes.
- Do not tune full-run async throughput before one tiny async smoke passes.
- Do not drop the no-signal replacement guard just because async is enabled.
- Do not assume async fixes Docker/node placement issues.

