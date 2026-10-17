import os
import sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retarget import load_model_compat, JOINT_NAMES


def export_phase_info(xml_path, ref_path, out_path, foot_threshold=0.06):
    """Run FK over the reference cycle and export per-phase foot-height
    stance masks (which leg is on the ground at each reference phase).

    stance[leg, i] = 1.0 if foot z < foot_threshold (on ground).
    """
    ref = np.load(ref_path)
    n = ref.shape[0]
    model = load_model_compat(xml_path)
    data = mujoco.MjData(model)
    dof_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
               for n in JOINT_NAMES]
    qadr = [model.jnt_qposadr[i] for i in dof_ids]
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                for b in ("left_ankle_link", "right_ankle_link")]
    foot_z = np.zeros((2, n))
    for i in range(n):
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.60
        for k, a in enumerate(ref[i]):
            data.qpos[qadr[k]] = a
        mujoco.mj_forward(model, data)
        for leg in (0, 1):
            foot_z[leg, i] = data.xpos[foot_ids[leg]][2]
    stance = (foot_z < foot_threshold).astype(np.float64)
    np.save(out_path, stance)
    np.save(out_path.replace("stance", "foot_z"), foot_z)
    # report duty cycle
    print(f"Saved: {out_path}  shape={stance.shape}")
    print(f"L foot duty (stance fraction): {stance[0].mean():.2%}")
    print(f"R foot duty (stance fraction): {stance[1].mean():.2%}")
    print(f"L foot z range: {foot_z[0].min():.3f}~{foot_z[0].max():.3f}")
    print(f"R foot z range: {foot_z[1].min():.3f}~{foot_z[1].max():.3f}")
    # phase offset between legs (should be ~half cycle)
    for lag in range(n):
        if np.mean(np.abs(stance[0] - np.roll(stance[1], lag))) < 1e-6:
            print(f"Stance anti-phase offset: {lag}/{n} samples")
            break
    return stance


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    xml = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "assets", "robots", "pandaman", "mjcf", "pandaman.xml")
    ref = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(ROOT, "data", "reference", "reference_traj.npy")
    out = sys.argv[3] if len(sys.argv) > 3 else \
        os.path.join(ROOT, "data", "reference", "ref_stance.npy")
    export_phase_info(xml, ref, out)
