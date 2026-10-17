import os
import sys
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pandaman_env_imitation import PandamanImitationEnv
from networks import ImprovedPolicy
from config import cfg

ROOT = os.path.dirname(os.path.abspath(__file__))
REF_TRAJ = os.path.join(ROOT, "data", "reference", "reference_traj.npy")
REF_STANCE = os.path.join(ROOT, "data", "reference", "ref_stance.npy")


def linear_schedule(initial_value, final_value):
    def func(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)
    return func


class AdaptiveEntropyCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_reward = -float("inf")
        self.stagnation_counter = 0

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            if mean_reward > self.best_reward * 1.01:
                self.best_reward = mean_reward
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            progress = self.model.num_timesteps / self.model._total_timesteps
            base_entropy = max(0.0001, 0.002 * (0.99 ** (progress * 1000)))
            if self.stagnation_counter > 100:
                base_entropy *= 1.5
            self.model.ent_coef = base_entropy
        return True


def make_env(ref_mode="sinusoid", vel_w=0.0, phase_w=0.0, rank=0,
             ref_traj=None, stance=None):
    def _init():
        env = PandamanImitationEnv(
            ref_mode=ref_mode,
            ref_traj_path=ref_traj or (REF_TRAJ if ref_mode == "human" else None),
            stance_path=stance or (REF_STANCE if ref_mode == "human" else None),
            vel_tracking_weight=vel_w,
            phase_align_weight=phase_w,
        )
        env = Monitor(env)
        return env
    return _init


def train(experiment_name, total_timesteps=5_000_000, n_envs=4,
          ref_mode="sinusoid", vel_w=0.0, phase_w=0.0,
          pretrained_zip=None, log_dir=None, seed=None,
          ref_traj=None, stance=None):
    if seed is not None:
        experiment_name = f"{experiment_name}_s{seed}"
    if log_dir is None:
        log_dir = os.path.join(cfg.LOG_DIR, experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([make_env(ref_mode, vel_w, phase_w, i,
                                ref_traj, stance) for i in range(n_envs)])

    model = PPO(
        ImprovedPolicy,
        env,
        learning_rate=linear_schedule(5e-4, 5e-5),
        n_steps=2048,
        batch_size=64,
        n_epochs=2,
        gamma=0.994,
        gae_lambda=0.9,
        clip_range=linear_schedule(0.3, 0.1),
        ent_coef=0.002,
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda",
        seed=seed,
    )

    if pretrained_zip and os.path.exists(pretrained_zip):
        print(f"[{experiment_name}] loading pretrained: {pretrained_zip}")
        model = PPO.load(pretrained_zip, env=env, device="cuda",
                         learning_rate=linear_schedule(5e-4, 5e-5),
                         clip_range=linear_schedule(0.3, 0.1),
                         ent_coef=0.002)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000, save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix=f"{experiment_name}",
    )
    eval_env = DummyVecEnv([make_env(ref_mode, vel_w, phase_w, -1,
                                     ref_traj, stance)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best"),
        log_path=log_dir,
        eval_freq=50000,
        deterministic=True,
        render=False,
    )
    adaptive_callback = AdaptiveEntropyCallback(verbose=1)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, adaptive_callback],
        tb_log_name=experiment_name,
    )
    model.save(os.path.join(log_dir, "final_model"))
    env.close()
    eval_env.close()
    print(f"[{experiment_name}] done -> {log_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=["baseline", "human", "ablation", "smoke"],
                   required=True)
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pretrained", type=str, default=None,
                   help="path to a zip to continue from")
    p.add_argument("--ref_path", type=str, default=None,
                   help="custom reference trajectory npy (for ablation D)")
    p.add_argument("--stance_path", type=str, default=None,
                   help="custom stance mask npy")
    args = p.parse_args()

    if args.run == "baseline":
        train("robio_baseline", args.steps, args.envs, "sinusoid",
              seed=args.seed)
    elif args.run == "human":
        train("robio_human", args.steps, args.envs, "human",
              vel_w=0.5, phase_w=0.5, seed=args.seed,
              pretrained_zip=args.pretrained,
              ref_traj=args.ref_path, stance=args.stance_path)
    elif args.run == "ablation":
        train("robio_ablation", args.steps, args.envs, "human",
              vel_w=0.0, phase_w=0.0, seed=args.seed,
              pretrained_zip=args.pretrained,
              ref_traj=args.ref_path, stance=args.stance_path)
    elif args.run == "smoke":
        train("robio_smoke", 20_000, 4, "human",
              vel_w=0.5, phase_w=0.5, seed=42)
