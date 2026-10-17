import os
import sys
import json
import argparse
import numpy as np
from scipy import stats
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pandaman_env_imitation import PandamanImitationEnv
from config import cfg

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = cfg.LOG_DIR
REF_TRAJ = os.path.join(ROOT, "data", "reference", "reference_traj.npy")
REF_STANCE = os.path.join(ROOT, "data", "reference", "ref_stance.npy")
REF_TRAJ_RAW = os.path.join(ROOT, "data", "reference",
                            "reference_traj_raw.npy")
OUT = os.path.join(ROOT, "results", "stats_per_episode.json")

N_EPISODES = 20
MAX_EP_STEPS = 500

MODELS = {
    "A": [
        ("A_single", os.path.join(RES, "improved_ab", "best", "best_model.zip"),
         "sinusoid", 0.0, 0.0, None),
    ],
    "B": [
        (d, os.path.join(RES, d, "final_model.zip"), "human", 0.5, 0.5, REF_TRAJ)
        for d in ("robio_human_s42", "robio_human_s100", "robio_human_s200")
    ],
    "C": [
        (d, os.path.join(RES, d, "final_model.zip"), "human", 0.0, 0.0, REF_TRAJ)
        for d in ("robio_ablation_s42", "robio_ablation_s100",
                  "robio_ablation_s200")
    ],
    "D": [
        ("robio_ablation_raw_s42",
         os.path.join(RES, "robio_ablation_raw_s42", "final_model.zip"),
         "human", 0.0, 0.0, REF_TRAJ_RAW),
    ],
}


def evaluate_model(model_path, ref_mode, vel_w, phase_w, ref_traj,
                   n_episodes=N_EPISODES, seed=0):
    traj_path = ref_traj or (REF_TRAJ if ref_mode == "human" else None)
    stance_path = REF_STANCE if ref_mode == "human" else None
    env = PandamanImitationEnv(
        ref_mode=ref_mode, ref_traj_path=traj_path, stance_path=stance_path,
        vel_tracking_weight=vel_w, phase_align_weight=phase_w)
    model = PPO.load(model_path, device="cuda")
    ref = np.load(traj_path) if ref_mode == "human" else None

    episodes = []
    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        rmse_steps = []
        for step in range(MAX_EP_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if ref is not None:
                ref_dof = env._compute_ref_dof(0)
                rmse_steps.append(
                    np.sqrt(np.mean((env.dof_pos - ref_dof) ** 2)))
            elif ref_mode == "sinusoid":
                ref_dof = env._compute_ref_dof(0)
                rmse_steps.append(
                    np.sqrt(np.mean((env.dof_pos - ref_dof) ** 2)))
            if terminated or truncated:
                break
        episodes.append({
            "reward": float(ep_reward),
            "len": step + 1,
            "survived": 1.0 if (step + 1) >= MAX_EP_STEPS else 0.0,
            "rmse": float(np.mean(rmse_steps)) if rmse_steps else None,
        })
    env.close()
    return episodes


def hedge_g(x, y):
    n1, n2 = len(x), len(y)
    s1 = np.var(x, ddof=1)
    s2 = np.var(y, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    d = (np.mean(x) - np.mean(y)) / sp
    return d * (1 - 3 / (4 * (n1 + n2) - 9))


def bootstrap_diff(x, y, n_boot=10000, rng=None):
    rng = rng or np.random.default_rng(42)
    diffs = []
    nx, ny = len(x), len(y)
    for _ in range(n_boot):
        bx = rng.choice(x, size=nx, replace=True)
        by = rng.choice(y, size=ny, replace=True)
        diffs.append(np.mean(bx) - np.mean(by))
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    results = {}
    for cond, models in MODELS.items():
        for key, path, mode, vw, pw, ref_traj in models:
            if not os.path.exists(path):
                print(f"[SKIP] {key}: {path} not found")
                continue
            print(f"[eval] {key} ({mode}, vel_w={vw}, phase_w={pw})", flush=True)
            eps = evaluate_model(path, mode, vw, pw, ref_traj,
                                 n_episodes=args.episodes)
            results[key] = eps
            r = [e["reward"] for e in eps]
            print(f"       reward {np.mean(r):.1f} +/- {np.std(r):.1f}  "
                  f"len {np.mean([e['len'] for e in eps]):.0f}  "
                  f"survived {np.mean([e['survived'] for e in eps]):.0%}",
                  flush=True)

    def seed_means(keys):
        return np.array([np.mean([e["reward"] for e in results[k]])
                         for k in keys])

    def pool(keys):
        return np.array([e["reward"] for k in keys for e in results[k]])

    out = {"n_episodes": args.episodes, "max_ep_steps": MAX_EP_STEPS,
           "per_model": {}, "tests": {}}
    for k, eps in results.items():
        r = np.array([e["reward"] for e in eps])
        out["per_model"][k] = {
            "reward_mean": float(r.mean()), "reward_std": float(r.std()),
            "ep_len_mean": float(np.mean([e["len"] for e in eps])),
            "survived_full": float(np.mean([e["survived"] for e in eps])),
            "rmse_mean": float(np.mean([e["rmse"] for e in eps
                                        if e["rmse"] is not None])),
        }

    rng = np.random.default_rng(42)
    comparisons = [
        ("C_vs_B", ["robio_ablation_s42", "robio_ablation_s100",
                    "robio_ablation_s200"],
         ["robio_human_s42", "robio_human_s100", "robio_human_s200"]),
        ("C_vs_A", ["robio_ablation_s42", "robio_ablation_s100",
                    "robio_ablation_s200"], ["A_single"]),
        ("B_vs_A", ["robio_human_s42", "robio_human_s100",
                    "robio_human_s200"], ["A_single"]),
        ("C_vs_D", ["robio_ablation_s42", "robio_ablation_s100",
                    "robio_ablation_s200"], ["robio_ablation_raw_s42"]),
    ]
    for name, c_keys, o_keys in comparisons:
        if not all(k in results for k in c_keys) or \
           not all(k in results for k in o_keys):
            continue
        cm, om = seed_means(c_keys), seed_means(o_keys)
        cp, op = pool(c_keys), pool(o_keys)
        out["tests"][name] = {
            "per_seed_means": {"C": cm.tolist(), "other": om.tolist()},
            "mean_diff_seed_level": float(cm.mean() - om.mean()),
            "bootstrap95_seed_level": bootstrap_diff(cm, om, rng=rng),
            "mannwhitney_seed_level": {
                "U": float(stats.mannwhitneyu(cm, om,
                                              alternative="two-sided").statistic),
                "p": float(stats.mannwhitneyu(cm, om,
                                              alternative="two-sided").pvalue),
            },
            "welch_pooled_episodes": {
                "t": float(stats.ttest_ind(cp, op,
                                           equal_var=False).statistic),
                "p": float(stats.ttest_ind(cp, op,
                                           equal_var=False).pvalue),
            },
            "mannwhitney_pooled_episodes": {
                "p": float(stats.mannwhitneyu(cp, op,
                                              alternative="two-sided").pvalue),
            },
            "hedges_g_seed_level": float(hedge_g(cm, om)),
            "hedges_g_pooled_episodes": float(hedge_g(cp, op)),
            "bootstrap95_pooled_episodes": bootstrap_diff(cp, op, rng=rng),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
