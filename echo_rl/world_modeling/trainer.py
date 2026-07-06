from __future__ import annotations

from collections import defaultdict
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.generators.base import GeneratorOutput
from skyrl.train.generators.utils import get_metrics_from_generator_output, merge_stepwise_output
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import Timer
from skyrl.train.utils.trainer_utils import zero_variance_filter
from loguru import logger


class EchoPPOTrainer(RayPPOTrainer):
    """Ray PPO trainer extension that carries world-model masks as generic extras."""

    @staticmethod
    def _policy_metric_logging_enabled() -> bool:
        return os.environ.get("ECHO_LOG_POLICY_TRAIN_METRICS", "1") != "0"

    @staticmethod
    def _tensor_sum_for_log(value: Any) -> Optional[float]:
        if value is None or not hasattr(value, "detach"):
            return None
        return float(value.detach().float().sum().cpu().item())

    @staticmethod
    def _list_sum_for_log(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(sum(value))
        except TypeError:
            return None

    @staticmethod
    def _trajectory_id_for_log(value: Any) -> str:
        if hasattr(value, "to_string"):
            return str(value.to_string())
        return str(value)

    def _log_generator_sample_summaries(self, generator_output: GeneratorOutput) -> None:
        if not self._policy_metric_logging_enabled():
            return

        response_ids = generator_output["response_ids"]
        loss_masks = generator_output.get("loss_masks") or []
        world_masks = generator_output.get("world_loss_masks") or []
        warning_masks = generator_output.get("world_warning_masks") or []
        env_masks = generator_output.get("world_env_masks") or []
        nextlat_masks = generator_output.get("nextlat_loss_masks") or []
        trajectory_ids = generator_output.get("trajectory_ids") or []
        stop_reasons = generator_output.get("stop_reasons") or []
        metadata = generator_output.get("trajectory_metadata") or []

        no_signal = 0
        zero_world = 0
        for i, response in enumerate(response_ids):
            loss_tokens = self._list_sum_for_log(loss_masks[i] if i < len(loss_masks) else None)
            world_tokens = self._list_sum_for_log(world_masks[i] if i < len(world_masks) else None)
            warning_tokens = self._list_sum_for_log(warning_masks[i] if i < len(warning_masks) else None)
            env_tokens = self._list_sum_for_log(env_masks[i] if i < len(env_masks) else None)
            nextlat_tokens = self._list_sum_for_log(nextlat_masks[i] if i < len(nextlat_masks) else None)
            if world_tokens == 0.0:
                zero_world += 1
            if all((tokens or 0.0) == 0.0 for tokens in (loss_tokens, world_tokens, warning_tokens, env_tokens, nextlat_tokens)):
                no_signal += 1

            meta = metadata[i] if i < len(metadata) and isinstance(metadata[i], dict) else {}
            logger.info(
                "[policy-train] generator_sample "
                f"idx={i} trajectory={self._trajectory_id_for_log(trajectory_ids[i]) if i < len(trajectory_ids) else None} "
                f"path={meta.get('path')} stop_reason={stop_reasons[i] if i < len(stop_reasons) else meta.get('stop_reason')} "
                f"response_len={len(response)} loss_tokens={loss_tokens} world_tokens={world_tokens} "
                f"warning_tokens={warning_tokens} env_tokens={env_tokens} nextlat_tokens={nextlat_tokens}"
            )

        logger.info(
            "[policy-train] generator_sample_summary "
            f"batch={len(response_ids)} zero_world_samples={zero_world} no_signal_samples={no_signal}"
        )
        self.all_metrics.update(
            {
                "generate/zero_world_samples": float(zero_world),
                "generate/no_signal_samples": float(no_signal),
            }
        )

    def _training_batch_summary_for_log(self, data: TrainingInputBatch) -> Dict[str, Any]:
        sequences = data.get("sequences")
        seq_len = int(sequences.shape[1]) if sequences is not None and hasattr(sequences, "shape") else None
        boundaries = (data.metadata or {}).get("policy_mini_batch_boundaries")
        return {
            "batch": len(data),
            "seq_len": seq_len,
            "response_length": (data.metadata or {}).get("response_length"),
            "pad_size": (data.metadata or {}).get("pad_size", 0),
            "policy_minibatches": len(boundaries) if boundaries is not None else None,
            "loss_tokens": self._tensor_sum_for_log(data.get("loss_mask")),
            "world_tokens": self._tensor_sum_for_log(data.get("world_loss_mask")),
            "world_warning_tokens": self._tensor_sum_for_log(data.get("world_warning_mask")),
            "world_env_tokens": self._tensor_sum_for_log(data.get("world_env_mask")),
            "nextlat_tokens": self._tensor_sum_for_log(data.get("nextlat_loss_mask")),
            "adv_mean": float(data["advantages"].detach().float().mean().cpu().item())
            if data.get("advantages") is not None
            else None,
            "reward_mean": float(data["rewards"].detach().float().mean().cpu().item())
            if data.get("rewards") is not None
            else None,
        }

    def convert_to_training_input(self, generator_output: GeneratorOutput, uids: List[str]) -> TrainingInputBatch:
        self._log_generator_sample_summaries(generator_output)
        training_input = super().convert_to_training_input(generator_output, uids)
        response_length = int(training_input.metadata["response_length"])
        response_ids = generator_output["response_ids"]
        pad_size = int(training_input.metadata.get("pad_size", 0) or 0)

        def right_align(mask_lists: Optional[List[List[int]]], name: str) -> Optional[torch.Tensor]:
            if mask_lists is None:
                return None
            if len(mask_lists) != len(response_ids):
                raise AssertionError(f"{name} must have one row per response")
            tensor = torch.zeros(len(response_ids), response_length, dtype=torch.float)
            for i, mask in enumerate(mask_lists):
                if len(mask) != len(response_ids[i]):
                    raise AssertionError(
                        f"{name} must match response length for sample {i}, "
                        f"got {len(mask)} and {len(response_ids[i])}"
                    )
                tensor[i, response_length - len(mask) :] = torch.tensor(mask, dtype=torch.float)
            if pad_size:
                tensor = torch.cat([tensor, torch.zeros(pad_size, response_length, dtype=tensor.dtype)], dim=0)
            return tensor

        world_loss_mask = right_align(generator_output.get("world_loss_masks"), "world_loss_masks")
        world_warning_mask = right_align(generator_output.get("world_warning_masks"), "world_warning_masks")
        world_env_mask = right_align(generator_output.get("world_env_masks"), "world_env_masks")
        nextlat_loss_mask = right_align(generator_output.get("nextlat_loss_masks"), "nextlat_loss_masks")

        counts = generator_output.get("world_full_observation_counts")
        world_full_observation_count = None
        if counts is not None:
            if len(counts) != len(response_ids):
                raise AssertionError("world_full_observation_counts must have one value per response")
            world_full_observation_count = torch.zeros(len(response_ids), response_length, dtype=torch.float)
            for i, count in enumerate(counts):
                world_full_observation_count[i, response_length - len(response_ids[i]) :] = float(count)
            if pad_size:
                world_full_observation_count = torch.cat(
                    [
                        world_full_observation_count,
                        torch.zeros(pad_size, response_length, dtype=world_full_observation_count.dtype),
                    ],
                    dim=0,
                )

        training_input["world_loss_mask"] = world_loss_mask
        training_input["world_warning_mask"] = world_warning_mask
        training_input["world_env_mask"] = world_env_mask
        training_input["nextlat_loss_mask"] = nextlat_loss_mask
        training_input["world_full_observation_count"] = world_full_observation_count

        zero_pad_keys = set(training_input.metadata.get("zero_pad_keys", []))
        zero_pad_keys.update(
            {
                "world_loss_mask",
                "world_warning_mask",
                "world_env_mask",
                "nextlat_loss_mask",
                "world_full_observation_count",
            }
        )
        training_input.metadata["zero_pad_keys"] = sorted(zero_pad_keys)
        return training_input

    def postprocess_generator_output(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> Tuple[GeneratorOutput, List[str]]:
        rewards = generator_output["rewards"]
        if rewards and not isinstance(rewards[0], list):
            self._apply_world_model_only(generator_output)
            self._apply_world_model_filter(generator_output, uids)
        return super().postprocess_generator_output(generator_output, uids)

    def _apply_world_model_only(self, generator_output: GeneratorOutput) -> None:
        world_model_only = generator_output.get("world_model_only")
        if not world_model_only:
            return
        for i, wm_only in enumerate(world_model_only):
            if wm_only:
                generator_output["loss_masks"][i] = [0] * len(generator_output["loss_masks"][i])

    def _apply_world_model_filter(self, generator_output: GeneratorOutput, uids: List[str]) -> None:
        cfg = self.cfg.trainer.algorithm
        filters_active = (
            cfg.wm_filter_min_valid_tool_call_pct is not None
            or cfg.wm_filter_min_parse_clean_pct is not None
            or cfg.wm_filter_min_correct_pct is not None
        )
        if not filters_active:
            return
        if (
            generator_output.get("trajectory_metadata") is None
            or generator_output.get("world_loss_masks") is None
            or generator_output.get("correct") is None
        ):
            return

        total = passed_quality = kept = correct_kept = 0
        dropped_low_turns = dropped_by_tool_call = dropped_by_parse_clean = dropped_by_correctness = 0
        prompts_with_zero_kept = 0
        kept_reward_sum = kept_parse_err_per_turn_sum = kept_num_turns_sum = 0.0
        dropped_reward_sum = dropped_parse_err_per_turn_sum = dropped_num_turns_sum = 0.0
        kept_stats_n = dropped_stats_n = 0

        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, uid in enumerate(uids):
            uid_to_indices[uid].append(i)

        rewards = generator_output["rewards"]
        metadata = generator_output["trajectory_metadata"]
        correct = generator_output["correct"]

        for indices in uid_to_indices.values():
            total += len(indices)
            filter_result: dict[int, str] = {}
            rollout_stats: dict[int, tuple[float, float, int, bool]] = {}
            for idx in indices:
                meta = metadata[idx] or {}
                turns = (meta.get("trace") or {}).get("turns") or []
                n_turns = int(meta.get("num_agent_turns", 0) or 0)
                if n_turns <= 0:
                    filter_result[idx] = "low_turns"
                    continue
                parse_errors = sum(1 for turn in turns if turn.get("parse_error"))
                clean_turns = sum(
                    1 for turn in turns if not turn.get("parse_error") and not turn.get("format_violations")
                )
                valid_tc_pct = (n_turns - parse_errors) / n_turns
                parse_clean_pct = clean_turns / n_turns
                is_correct = bool(correct[idx])
                rollout_stats[idx] = (float(rewards[idx]), parse_errors / n_turns, n_turns, is_correct)
                if cfg.wm_filter_min_valid_tool_call_pct is not None and valid_tc_pct < cfg.wm_filter_min_valid_tool_call_pct:
                    filter_result[idx] = "tool_call_drop"
                elif cfg.wm_filter_min_parse_clean_pct is not None and parse_clean_pct < cfg.wm_filter_min_parse_clean_pct:
                    filter_result[idx] = "parse_clean_drop"
                else:
                    filter_result[idx] = "quality_passed"

            survivors = [(idx, rollout_stats[idx][3]) for idx in indices if filter_result[idx] == "quality_passed"]
            passed_quality += len(survivors)
            if cfg.wm_filter_min_correct_pct is not None:
                theta = cfg.wm_filter_min_correct_pct
                correct_indices = [idx for idx, is_correct in survivors if is_correct]
                incorrect_indices = [idx for idx, is_correct in survivors if not is_correct]
                if theta >= 1.0 or not correct_indices:
                    kept_indices = set(correct_indices)
                else:
                    max_incorrect = int(len(correct_indices) * (1.0 - theta) / theta)
                    kept_indices = set(correct_indices) | set(incorrect_indices[:max_incorrect])
            else:
                kept_indices = {idx for idx, _ in survivors}

            kept += len(kept_indices)
            correct_kept += sum(1 for idx, is_correct in survivors if is_correct and idx in kept_indices)
            if not kept_indices:
                prompts_with_zero_kept += 1

            for idx in indices:
                result = filter_result[idx]
                if result == "low_turns":
                    dropped_low_turns += 1
                    continue
                score, parse_err_pt, n_turns, _ = rollout_stats[idx]
                if result == "tool_call_drop":
                    dropped_by_tool_call += 1
                    in_kept = False
                elif result == "parse_clean_drop":
                    dropped_by_parse_clean += 1
                    in_kept = False
                else:
                    in_kept = idx in kept_indices
                    if not in_kept:
                        dropped_by_correctness += 1
                if in_kept:
                    kept_reward_sum += score
                    kept_parse_err_per_turn_sum += parse_err_pt
                    kept_num_turns_sum += n_turns
                    kept_stats_n += 1
                else:
                    dropped_reward_sum += score
                    dropped_parse_err_per_turn_sum += parse_err_pt
                    dropped_num_turns_sum += n_turns
                    dropped_stats_n += 1

            for idx in indices:
                if idx in kept_indices:
                    continue
                for key in ("world_loss_masks", "world_warning_masks", "world_env_masks", "nextlat_loss_masks"):
                    if generator_output.get(key) is not None:
                        generator_output[key][idx] = [0] * len(generator_output[key][idx])

        def avg(total_value: float, count: int) -> float:
            return total_value / count if count else 0.0

        self.all_metrics.update(
            {
                "train/wm_filter_total": float(total),
                "train/wm_filter_passed_quality": float(passed_quality),
                "train/wm_filter_kept": float(kept),
                "train/wm_filter_correct_kept": float(correct_kept),
                "train/wm_filter_kept_frac": (kept / total) if total else 0.0,
                "train/wm_filter_dropped_low_turns": float(dropped_low_turns),
                "train/wm_filter_dropped_by_tool_call": float(dropped_by_tool_call),
                "train/wm_filter_dropped_by_parse_clean": float(dropped_by_parse_clean),
                "train/wm_filter_dropped_by_correctness": float(dropped_by_correctness),
                "train/wm_filter_kept_avg_reward": avg(kept_reward_sum, kept_stats_n),
                "train/wm_filter_kept_avg_parse_errors_per_turn": avg(kept_parse_err_per_turn_sum, kept_stats_n),
                "train/wm_filter_kept_avg_num_turns": avg(kept_num_turns_sum, kept_stats_n),
                "train/wm_filter_dropped_avg_reward": avg(dropped_reward_sum, dropped_stats_n),
                "train/wm_filter_dropped_avg_parse_errors_per_turn": avg(dropped_parse_err_per_turn_sum, dropped_stats_n),
                "train/wm_filter_dropped_avg_num_turns": avg(dropped_num_turns_sum, dropped_stats_n),
                "train/wm_filter_prompts_with_zero_kept": float(prompts_with_zero_kept),
            }
        )

    def train_critic_and_policy(self, data: TrainingInputBatch):
        data.metadata["world_model_schedule_step"] = max(self.global_step - 1, 0)
        data.metadata["total_training_steps"] = self.total_training_steps or 0
        if not self._policy_metric_logging_enabled():
            return super().train_critic_and_policy(data)

        summary = self._training_batch_summary_for_log(data)
        logger.info(
            "[policy-train] trainer_input "
            f"global_step={self.global_step} total_steps={self.total_training_steps} "
            f"batch={summary['batch']} seq_len={summary['seq_len']} response_length={summary['response_length']} "
            f"pad_size={summary['pad_size']} policy_minibatches={summary['policy_minibatches']} "
            f"loss_tokens={summary['loss_tokens']} world_tokens={summary['world_tokens']} "
            f"warning_tokens={summary['world_warning_tokens']} env_tokens={summary['world_env_tokens']} "
            f"nextlat_tokens={summary['nextlat_tokens']} adv_mean={summary['adv_mean']} reward_mean={summary['reward_mean']}"
        )
        started = time.monotonic()
        try:
            status = super().train_critic_and_policy(data)
        except Exception:
            logger.exception(
                f"[policy-train] trainer_failed global_step={self.global_step} sec={time.monotonic() - started:.2f}"
            )
            raise
        logger.info(
            "[policy-train] trainer_done "
            f"global_step={self.global_step} sec={time.monotonic() - started:.2f} "
            f"status={{{', '.join(f'{key}: {value}' for key, value in sorted(status.items()))}}}"
        )
        return status
