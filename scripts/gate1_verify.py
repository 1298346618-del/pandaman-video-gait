import os
import sys
import json
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retarget import load_model_compat, JOINT_NAMES, DEFAULT_ANGLES, JOINT_LIMITS


def play_reference(xml_path, ref_path, cycles=2, verbose=True):
    """Play a reference trajectory open-loop through MuJoCo FK and report
    gait statistics: foot swing, periodicity, joint limit margin."""
    ref = np.load(ref_path)
    n = ref.shape[0]
    model = load_model_compat(xml_path)
    data = mujoco.MjData(model)
    dof_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
               for n in JOINT_NAMES]
    qadr = [model.jnt_qposadr[i] for i in dof_ids]
    foot_bodies = ["left_ankle_link", "right_ankle_link"]
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                for b in foot_bodies]

    foot_z = {0: [], 1: []}
    for c in range(cycles):
        for i in range(n):
            mujoco.mj_resetData(model, data)
            data.qpos[2] = 0.60
            for k, a in enumerate(ref[i]):
                data.qpos[qadr[k]] = a
            mujoco.mj_forward(model, data)
            for leg in (0, 1):
                foot_z[leg].append(data.xpos[foot_ids[leg]][2])

    foot_z = {k: np.array(v) for k, v in foot_z.items()}
    half = n // 2
    # phase lag: L foot height vs R foot height shifted by half cycle
    a = foot_z[0][:n] - foot_z[0][:n].mean()
    b = foot_z[1][:n] - foot_z[1][:n].mean()
    corr_opp = np.corrcoef(a, np.roll(b, half))[0, 1]
    corr_same = np.corrcoef(a, b)[0, 1]

    # joint limit margin: how close ref gets to limits
    lo_margin = (ref - JOINT_LIMITS[:, 0]).min(axis=0)
    hi_margin = (JOINT_LIMITS[:, 1] - ref).min(axis=0)

    stats = {
        "n_samples": n,
        "L_foot_swing_amp": float(foot_z[0].max() - foot_z[0].min()),
        "R_foot_swing_amp": float(foot_z[1].max() - foot_z[1].min()),
        "foot_antiphase_corr": float(corr_opp),
        "foot_samephase_corr": float(corr_same),
        "min_limit_margin_lo": float(lo_margin.min()),
        "min_limit_margin_hi": float(hi_margin.min()),
        "in_limits": bool(np.all(ref >= JOINT_LIMITS[:, 0] - 1e-6)
                          and np.all(ref <= JOINT_LIMITS[:, 1] + 1e-6)),
    }
    if verbose:
        print("=== Gate-1 FK verification (open-loop kinematics) ===")
        print(f"L foot swing amp: {stats['L_foot_swing_amp']*100:.1f} cm")
        print(f"R foot swing amp: {stats['R_foot_swing_amp']*100:.1f} cm")
        print(f"L vs R(shift half) corr: {corr_opp:+.3f} "
              f"(want high +, i.e. anti-phase)")
        print(f"L vs R(same) corr: {corr_same:+.3f} (want low/negative)")
        print(f"All joints in limits: {stats['in_limits']} "
              f"(min margins lo={stats['min_limit_margin_lo']:+.3f} "
              f"hi={stats['min_limit_margin_hi']:+.3f})")
    return stats


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    xml = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "assets", "robots", "pandaman", "mjcf", "pandaman.xml")
    ref = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(ROOT, "data", "reference", "reference_traj.npy")
    stats = play_reference(xml, ref)
    n = stats["n_samples"]
    checks = [
        ("feet swing > 3cm",
         min(stats["L_foot_swing_amp"], stats["R_foot_swing_amp"]) > 0.03,
         f"L={stats['L_foot_swing_amp']*100:.1f} R={stats['R_foot_swing_amp']*100:.1f} cm"),
        ("feet anti-phase (corr(shift half) > 0.5)",
         stats["foot_antiphase_corr"] > 0.5,
         f"corr={stats['foot_antiphase_corr']:+.3f}"),
        ("left/right swing balanced (<20% diff)",
         abs(stats["L_foot_swing_amp"] - stats["R_foot_swing_amp"])
         / max(stats["L_foot_swing_amp"], stats["R_foot_swing_amp"], 1e-6) < 0.2,
         f"L={stats['L_foot_swing_amp']*100:.1f} R={stats['R_foot_swing_amp']*100:.1f} cm"),
        ("all joints within limits", stats["in_limits"],
         f"min margins lo={stats['min_limit_margin_lo']:+.3f} "
         f"hi={stats['min_limit_margin_hi']:+.3f}"),
    ]
    print()
    passed = True
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        passed &= ok
    sys.exit(0 if passed else 1)
