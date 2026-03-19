"""Train B2-W wheel locomotion policy using RSL-RL PPO."""

import argparse
import os
import sys

from omni.isaac.lab.app import AppLauncher

# -- CLI args --
parser = argparse.ArgumentParser(description="Train B2-W wheel locomotion policy.")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=1500, help="Number of training iterations.")
parser.add_argument("--run_name", type=str, default="b2w_wheel", help="Name for this run.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# -- Launch Isaac Sim --
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- Imports (after sim launch) --
import torch
from datetime import datetime

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils.dict import print_dict

from rsl_rl.runners import OnPolicyRunner
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, "/home/ubuntu/b2w_project")
from b2w_wheel_env import B2WWheelEnvCfg


##
# RSL-RL PPO agent config
##

agent_cfg = {
    "seed": args_cli.seed,
    "device": "cuda:0",
    "num_steps_per_env": 24,
    "max_iterations": args_cli.max_iterations,
    "save_interval": 100,
    "empirical_normalization": False,
    "policy": {
        "class_name": "ActorCritic",
        "init_noise_std": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
    },
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
}


##
# Main
##

def main():
    # -- Environment --
    env_cfg = B2WWheelEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # -- Log directory --
    log_root = os.path.join("/home/ubuntu/runs", args_cli.run_name)
    os.makedirs(log_root, exist_ok=True)

    # -- Runner --
    wrapped_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped_env, agent_cfg, log_dir=log_root, device=agent_cfg["device"])

    print("[INFO] Starting training...")
    print(f"[INFO] Logging to: {log_root}")
    print(f"[INFO] Num envs: {args_cli.num_envs}")
    print(f"[INFO] Max iterations: {args_cli.max_iterations}")

    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)

    # -- Save final policy --
    final_ckpt = os.path.join(log_root, "final_policy.pt")
    runner.save(final_ckpt)
    print(f"[INFO] Final policy saved to: {final_ckpt}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()