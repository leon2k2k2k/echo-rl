#!/usr/bin/env python3
"""Validate terminal-agent config and overrides without starting Ray."""

import argparse
import shlex

from skyrl.train.utils import validate_cfg

from echo_rl.terminal_agent.entrypoint import _load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overrides", default="")
    args = parser.parse_args()

    argv = ["--config", args.config, *shlex.split(args.overrides)]
    cfg = _load_config(argv)
    validate_cfg(cfg)

    print("CONFIG_PREFLIGHT_OK")
    print(f"config={args.config}")
    print(f"train_batch_size={cfg.trainer.train_batch_size}")
    print(f"policy_mini_batch_size={cfg.trainer.policy_mini_batch_size}")
    print(f"n_samples_per_prompt={cfg.generator.n_samples_per_prompt}")
    print(f"dataset_max_rows={cfg.generator.dataset_max_rows}")
    print(f"eval_before_train={cfg.trainer.eval_before_train}")
    print(f"eval_interval={cfg.trainer.eval_interval}")
    print(f"ckpt_interval={cfg.trainer.ckpt_interval}")
    print(f"policy_lr={cfg.trainer.policy.optimizer_config.lr}")
    print(f"nextlat_lr={cfg.trainer.algorithm.nextlat_lr}")
    print(f"nextlat_coeff={cfg.trainer.algorithm.nextlat_coeff}")
    print(f"nextlat_lambda_mse={cfg.trainer.algorithm.nextlat_lambda_mse}")
    print(f"nextlat_lambda_kl={cfg.trainer.algorithm.nextlat_lambda_kl}")
    print(f"nextlat_lambda_ce={cfg.trainer.algorithm.nextlat_lambda_ce}")
    print(f"nextlat_base_reward_gate={cfg.trainer.algorithm.nextlat_base_reward_gate}")
    print(f"policy_num_gpus_per_node={cfg.trainer.placement.policy_num_gpus_per_node}")
    print(f"ref_num_gpus_per_node={cfg.trainer.placement.ref_num_gpus_per_node}")
    print(f"num_engines={cfg.generator.inference_engine.num_engines}")


if __name__ == "__main__":
    main()
