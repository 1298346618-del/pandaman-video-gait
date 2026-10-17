import os
import sys
import json
import argparse
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pandaman_env_imitation import PandamanImitationEnv
from config import cfg

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(cfg.LOG_DIR, "improved_ab", "best", "best_model.zip")
OUT = os.path.join(ROOT, "results", "a_rmse_verify.json")
N_EVAL_EPISODES = 20
MAX_EP_STEPS = 500


def main():
    p = argparse.ArgumentParser(
        description="Verify condition A (ICIA sinusoid baseline) RMSE vs its "
                    "own sinusoidal reference, using the eval_robio protocol.")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--episodes", type=int, default=N_EVAL_EPISODES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    env = PandamanImitationEnv(ref_mode="sinusoid")
    model = PPO.load(args.model, device="cuda")

    rewards, ep_lens, survived = [], [], []
    rmse_acc_all, vel_rmse_acc_all = [], []

    for ep in range(args.episodes):
        np.random.seed(args.seed + ep)
        obs, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        rmse_steps, vel_rmse_steps = [], []

        for step in range(MAX_EP_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            ref_dof = env._compute_ref_dof(0)
            rmse_steps.append(np.sqrt(np.mean((env.dof_pos - ref_dof) ** 2)))
            ref_vel = env._compute_ref_vel(0)
            vel_rmse_steps.append(np.sqrt(np.mean((env.dof_vel - ref_vel) ** 2)))

            if terminated or truncated:
                break

        rewards.append(ep_reward)
        ep_lens.append(step + 1)
        survived.append(1.0 if (step + 1) >= MAX_EP_STEPS else 0.0)
        rmse_acc_all.append(np.mean(rmse_steps))
        vel_rmse_acc_all.append(np.mean(vel_rmse_steps))

    env.close()

    summary = {
        "model": args.model,
        "n_episodes": args.episodes,
        "max_ep_steps": MAX_EP_STEPS,
        "ref": "sinusoidal (analytic, TARGET_JOINT_POS_SCALE=0.26)",
        "total_reward": {"mean": float(np.mean(rewards)),
                         "std": float(np.std(rewards))},
        "ep_length": {"mean": float(np.mean(ep_lens)),
                      "std": float(np.std(ep_lens))},
        "survived_full": float(np.mean(survived)),
        "imitation_rmse": {"mean": float(np.mean(rmse_acc_all)),
                           "std": float(np.std(rmse_acc_all)),
                           "min": float(np.min(rmse_acc_all)),
                           "max": float(np.max(rmse_acc_all))},
        "imitation_vel_rmse": {"mean": float(np.mean(vel_rmse_acc_all)),
                               "std": float(np.std(vel_rmse_acc_all))},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Model:        {args.model}")
    print(f"total_reward: {summary['total_reward']['mean']:.2f} "
          f"+/- {summary['total_reward']['std']:.2f}")
    print(f"ep_length:    {summary['ep_length']['mean']:.2f}")
    print(f"survived_full:{summary['survived_full']:.2%}")
    print(f"imitation_rmse:     {summary['imitation_rmse']['mean']:.4f} rad "
          f"(min={summary['imitation_rmse']['min']:.4f}, "
          f"max={summary['imitation_rmse']['max']:.4f})")
    print(f"imitation_vel_rmse: {summary['imitation_vel_rmse']['mean']:.4f} rad/s")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
