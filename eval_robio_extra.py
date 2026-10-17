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

MAX_EP_STEPS = 500


def evaluate_speed_profile(model_path, ref_mode="human", vel_w=0.0,
                           phase_w=0.0, n_episodes=10):
    """Evaluate at fixed velocity commands. Returns {cmd: dict}."""
    env = PandamanImitationEnv(
        ref_mode=ref_mode,
        ref_traj_path=REF_TRAJ if ref_mode == "human" else None,
        stance_path=REF_STANCE if ref_mode == "human" else None,
        vel_tracking_weight=vel_w, phase_align_weight=phase_w,
    )
    model = PPO.load(model_path, device="cuda")
    results = {}
    for spd in [0.0, 0.2, 0.4, 0.6]:
        rewards, lengths, survived = [], [], []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=1000 + ep)
            env.commands[:] = [spd, 0.0, 0.0]
            ep_r, ep_len = 0.0, 0
            for step in range(MAX_EP_STEPS):
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                ep_r += r
                ep_len = step + 1
                if term or trunc:
                    break
            rewards.append(ep_r)
            lengths.append(ep_len)
            survived.append(1.0 if ep_len >= MAX_EP_STEPS else 0.0)
        results[str(spd)] = {
            "total_reward_mean": float(np.mean(rewards)),
            "total_reward_std": float(np.std(rewards)),
            "push_recovery_rate": float(np.mean(survived)),
        }
    env.close()
    return results


def evaluate_gait_naturalness(model_path, ref_mode="human", vel_w=0.0,
                              phase_w=0.0, n_episodes=5):
    """Gait naturalness: cadence (Hz), foot slip, mean torque (Nm)."""
    env = PandamanImitationEnv(
        ref_mode=ref_mode,
        ref_traj_path=REF_TRAJ if ref_mode == "human" else None,
        stance_path=REF_STANCE if ref_mode == "human" else None,
        vel_tracking_weight=vel_w, phase_align_weight=phase_w,
    )
    model = PPO.load(model_path, device="cuda")
    from scipy.signal import argrelextrema
    cadences, slips, energies = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        env.commands[:] = [0.4, 0.0, 0.0]
        knee_l, slip_acc, torq_acc = [], [], []
        for step in range(MAX_EP_STEPS):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            knee_l.append(env.dof_pos[3])
            slip_acc.append(info["rewards"].get("foot_slip", 0.0))
            torq_acc.append(np.mean(np.abs(env.torques)))
            if term or trunc:
                break
        knee = np.array(knee_l)
        peaks = argrelextrema(knee, np.greater, order=10)[0]
        if len(peaks) >= 2:
            dt = (peaks[-1] - peaks[0]) / (len(peaks) - 1) * cfg.DT * cfg.DECIMATION
            cadences.append(1.0 / dt if dt > 0 else 0.0)
        else:
            cadences.append(0.0)
        slips.append(-np.mean(slip_acc))
        energies.append(np.mean(torq_acc))
    out = {
        "cadence_hz": float(np.mean(cadences)),
        "foot_slip": float(np.mean(slips)),
        "mean_torque_nm": float(np.mean(energies)),
    }
    env.close()
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()

    res = cfg.LOG_DIR
    # paper-final models: 3-seed dirs (use s42 as the reported model)
    models = [
        ("baseline_A", os.path.join(res, "improved_ab", "best", "best_model.zip"),
         "sinusoid", 0.0, 0.0),
        ("human_B_s42", os.path.join(res, "robio_human_s42", "final_model.zip"),
         "human", 0.5, 0.5),
        ("human_C_ours_s42", os.path.join(res, "robio_ablation_s42", "final_model.zip"),
         "human", 0.0, 0.0),
    ]
    out_all = {}
    for key, path, mode, vw, pw in models:
        if not os.path.exists(path):
            print(f"[SKIP] {key}: model missing")
            continue
        print(f"\n=== {key} — speed profile ===")
        sp = evaluate_speed_profile(path, mode, vw, pw, args.episodes)
        for s, v in sp.items():
            print(f"  {s}m/s: reward={v['total_reward_mean']:.0f} "
                  f"push={v['push_recovery_rate']:.0%}")
        print(f"=== {key} — gait naturalness ===")
        gn = evaluate_gait_naturalness(path, mode, vw, pw, max(5, args.episodes // 2))
        print(f"  cadence={gn['cadence_hz']:.2f}Hz slip={gn['foot_slip']:.4f} "
              f"torque={gn['mean_torque_nm']:.3f}Nm")
        out_all[key] = {"speed_profile": sp, "gait_naturalness": gn}

    out_path = os.path.join(ROOT, "results", "robio_extra.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_all, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
