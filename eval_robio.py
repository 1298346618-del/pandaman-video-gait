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
REF_TRAJ = os.path.join(ROOT, "data", "reference", "reference_traj.npy")
REF_STANCE = os.path.join(ROOT, "data", "reference", "ref_stance.npy")
REF_TRAJ_RAW = os.path.join(ROOT, "data", "reference",
                            "reference_traj_raw.npy")

N_EVAL_EPISODES = 20
MAX_EP_STEPS = 500


def evaluate_model(model_path, ref_mode="human", vel_w=0.0, phase_w=0.0,
                   n_episodes=N_EVAL_EPISODES, push_enabled=False,
                   seed=0, ref_traj=None, stance=None):
    """Evaluate a trained model. Returns summary dict with ROBIO metrics.

    Metrics:
      total_reward, ep_length, joint_pos, tracking_lin_vel, tracking_ang_vel,
      orientation, torques, foot_slip, action_smoothness   (ICIA-compatible)
      imitation_rmse   (rad, joint-angle RMSE vs reference)
      imitation_vel_rmse (rad/s)
      push_recovery_rate (fraction of episodes surviving 500 steps w/ pushes)
    """
    traj_path = ref_traj or (REF_TRAJ if ref_mode == "human" else None)
    stance_path = stance or (REF_STANCE if ref_mode == "human" else None)
    env = PandamanImitationEnv(
        ref_mode=ref_mode,
        ref_traj_path=traj_path,
        stance_path=stance_path,
        vel_tracking_weight=vel_w,
        phase_align_weight=phase_w,
    )
    model = PPO.load(model_path, device="cuda")

    # reference buffer for RMSE (aligned by phase)
    ref = np.load(traj_path) if ref_mode == "human" else None

    metrics = {
        "total_reward": [], "ep_length": [],
        "joint_pos": [], "tracking_lin_vel": [], "tracking_ang_vel": [],
        "orientation": [], "torques": [], "foot_slip": [],
        "action_smoothness": [],
        "imitation_rmse": [], "imitation_vel_rmse": [],
        "survived_full": [],
    }

    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_metrics = {k: [] for k in metrics if k not in
                      ("total_reward", "ep_length", "survived_full",
                       "imitation_rmse", "imitation_vel_rmse")}
        rmse_acc = []       # per-step joint RMSE
        vel_rmse_acc = []   # per-step velocity RMSE

        for step in range(MAX_EP_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if ref is not None:
                ref_dof = env._compute_ref_dof(0)
                rmse_acc.append(np.sqrt(np.mean((env.dof_pos - ref_dof) ** 2)))
                ref_vel = env._compute_ref_vel(0)
                vel_rmse_acc.append(np.sqrt(np.mean((env.dof_vel - ref_vel) ** 2)))

            if "rewards" in info:
                for k, v in info["rewards"].items():
                    if k in ep_metrics:
                        ep_metrics[k].append(v)

            if terminated or truncated:
                break

        metrics["total_reward"].append(ep_reward)
        metrics["ep_length"].append(step + 1)
        metrics["survived_full"].append(1.0 if (step + 1) >= MAX_EP_STEPS else 0.0)
        for k, vals in ep_metrics.items():
            metrics[k].append(np.mean(vals) if vals else 0.0)
        if rmse_acc:
            metrics["imitation_rmse"].append(np.mean(rmse_acc))
            metrics["imitation_vel_rmse"].append(np.mean(vel_rmse_acc))
        else:
            metrics["imitation_rmse"].append(0.0)
            metrics["imitation_vel_rmse"].append(0.0)

    env.close()

    summary = {}
    for k, vals in metrics.items():
        vals = np.array(vals)
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    summary["push_recovery_rate"] = summary["survived_full"]["mean"]
    return summary


def run_all(experiments, out_path, n_episodes=20, push=False):
    """experiments: list of (key, label, model_path, ref_mode, vel_w, phase_w)"""
    results = {}
    for key, label, model_path, ref_mode, vel_w, phase_w in experiments:
        if not os.path.exists(model_path):
            print(f"[SKIP] {label}: model not found {model_path}")
            continue
        print(f"\n{'='*60}\nEvaluating [{label}]\n{'='*60}")
        s = evaluate_model(model_path, ref_mode, vel_w, phase_w,
                           n_episodes=n_episodes, push_enabled=push)
        results[key] = s
        print(f"  total_reward: {s['total_reward']['mean']:.2f} ± {s['total_reward']['std']:.2f}")
        print(f"  ep_length:    {s['ep_length']['mean']:.1f}")
        print(f"  imitation_rmse: {s['imitation_rmse']['mean']:.4f} rad")
        print(f"  push_recovery:  {s['push_recovery_rate']:.2%}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    return results


def run_multiseed(groups, out_path, n_episodes=20):
    """Evaluate across seeds: groups = [(key, label, ref_mode, vel_w, phase_w,
    [model_paths...]), ...]. Aggregates per-seed means across seeds.
    Returns {key: {metric: {mean, std, seeds: [...]}}}."""
    out = {}
    for item in groups:
        if len(item) == 7:
            key, label, ref_mode, vel_w, phase_w, paths, ref_traj = item
        else:
            key, label, ref_mode, vel_w, phase_w, paths = item
            ref_traj = None
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            print(f"[SKIP] {label}: no models found")
            continue
        print(f"\n{'='*60}\n[{label}] {len(paths)} seed(s)\n{'='*60}")
        seed_results = []
        for p in paths:
            s = evaluate_model(p, ref_mode, vel_w, phase_w,
                               n_episodes=n_episodes, ref_traj=ref_traj)
            seed_results.append(s)
            print(f"  {os.path.basename(os.path.dirname(p))}: "
                  f"reward={s['total_reward']['mean']:.0f} "
                  f"rmse={s['imitation_rmse']['mean']:.4f} "
                  f"push={s['push_recovery_rate']:.0%}")
        agg = {}
        for m in seed_results[0]:
            vals = np.array([s[m]["mean"] if isinstance(s[m], dict)
                             else s[m] for s in seed_results])
            agg[m] = {"mean": float(vals.mean()),
                      "std": float(vals.std()),
                      "seeds": [float(v) for v in vals]}
        out[key] = agg
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(ROOT, "results",
                                                 "comparison_robio.json"))
    p.add_argument("--episodes", type=int, default=N_EVAL_EPISODES)
    p.add_argument("--push", action="store_true")
    p.add_argument("--multiseed", action="store_true",
                   help="aggregate across seed dirs (robio_human_s42 etc.)")
    args = p.parse_args()

    res_dir = os.path.join(cfg.LOG_DIR)
    if args.multiseed:
        groups = [
            ("baseline", "Sinusoidal Ref (ICIA)", "sinusoid", 0.0, 0.0,
             [os.path.join(res_dir, "improved_ab", "best", "best_model.zip")]),
            ("human", "Human Ref + Imitation (explicit)",
             "human", 0.5, 0.5,
             [os.path.join(res_dir, d, "final_model.zip")
              for d in ("robio_human_s42", "robio_human_s100",
                        "robio_human_s200")
              ]),
            ("ablation", "Human Ref, lightweight (ours)",
             "human", 0.0, 0.0,
             [os.path.join(res_dir, d, "final_model.zip")
              for d in ("robio_ablation_s42", "robio_ablation_s100",
                        "robio_ablation_s200")
              ]),
            ("raw", "Human Ref, no post-processing (D)",
             "human", 0.0, 0.0,
             [os.path.join(res_dir, "robio_ablation_raw_s42",
                           "final_model.zip")],
             REF_TRAJ_RAW),
        ]
        run_multiseed(groups, args.out, n_episodes=args.episodes)
    else:
        experiments = [
            # Run A: ICIA baseline (sinusoid reference, RA-PPO)
            ("baseline", "Sinusoidal Ref (ICIA RA-PPO)",
             os.path.join(res_dir, "improved_ab", "best", "best_model.zip"),
             "sinusoid", 0.0, 0.0),
            # Run B: human reference + imitation rewards (seed 42 checkpoint;
            # use --multiseed for the paper's 3-seed aggregate)
            ("human", "Human Ref + Imitation (Ours)",
             os.path.join(res_dir, "robio_human_s42", "final_model.zip"),
             "human", 0.5, 0.5),
            # Run C: ablation - human ref, no imitation rewards
            ("ablation", "Human Ref, no Imitation Rewards",
             os.path.join(res_dir, "robio_ablation_s42", "final_model.zip"),
             "human", 0.0, 0.0),
        ]
        run_all(experiments, args.out, n_episodes=args.episodes, push=args.push)
