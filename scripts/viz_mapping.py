"""Visualize the video->robot mapping.

Plan A: side-by-side human skeleton (MediaPipe) vs Pandaman skeleton
        (MuJoCo FK, matplotlib line skeleton, no OpenGL needed) at
        matched gait phases.
Plan C: human joint angles (raw) vs mapped robot joint angles over one
        gait cycle, showing scale+offset mapping.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retarget import (load_model_compat, JOINT_NAMES, DEFAULT_ANGLES,
                      extract_human_joints, symmetrize_cycle, map_to_robot,
                      calibrate_joint_signs)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
LM_PATH = os.path.join(ROOT, "data", "landmarks", "landmarks.npy")
XML_PATH = os.path.join(ROOT, "assets", "robots", "pandaman",
                        "mjcf", "pandaman.xml")
REF_PATH = os.path.join(ROOT, "data", "reference", "reference_traj.npy")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 9, 'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.unicode_minus': False,
})

# MediaPipe indices
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_SH, R_SH = 11, 12


# ----------------------------------------------------------------------
# Plan A helpers
# ----------------------------------------------------------------------
def human_2d_skeleton(lm_frame):
    """Return segment endpoints (x, y_up) for near-side leg + torso.
    MediaPipe: x right, y DOWN. Convert to y-up. Pick leg closer to
    camera (larger z? MediaPipe z smaller = closer). We pick the leg with
    larger z (farther from camera is mirrored); use left leg if its heel
    z > right heel z."""
    def pt(i):
        return np.array([lm_frame[i, 0], -lm_frame[i, 1]])
    hip_l, hip_r = pt(L_HIP), pt(R_HIP)
    knee_l, knee_r = pt(L_KNEE), pt(R_KNEE)
    ank_l, ank_r = pt(L_ANKLE), pt(R_ANKLE)
    heel_l, heel_r = pt(L_HEEL), pt(R_HEEL)
    torso = [(pt(L_SH) + pt(R_SH)) / 2, (hip_l + hip_r) / 2]
    segs = [
        torso,
        [hip_l, knee_l], [knee_l, ank_l], [ank_l, heel_l],
        [hip_r, knee_r], [knee_r, ank_r], [ank_r, heel_r],
    ]
    return segs


def robot_2d_skeleton(model, data, qpos_joints):
    """Set joints, forward kinematics, return sagittal (x, z-up) segments."""
    dof_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
               for n in JOINT_NAMES]
    qadr = [model.jnt_qposadr[i] for i in dof_ids]
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.60
    for k, a in enumerate(qpos_joints):
        data.qpos[qadr[k]] = a
    mujoco.mj_forward(model, data)
    bodies = ["base_link", "left_leg_pitch_link", "left_knee_link",
              "left_ankle_link", "right_leg_pitch_link", "right_knee_link",
              "right_ankle_link"]
    pos = {}
    for b in bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        x = data.xpos[bid]
        pos[b] = np.array([x[0], x[2]])   # sagittal: forward x, up z
    segs = [
        [pos["base_link"], pos["left_leg_pitch_link"]],
        [pos["left_leg_pitch_link"], pos["left_knee_link"]],
        [pos["left_knee_link"], pos["left_ankle_link"]],
        [pos["base_link"], pos["right_leg_pitch_link"]],
        [pos["right_leg_pitch_link"], pos["right_knee_link"]],
        [pos["right_knee_link"], pos["right_ankle_link"]],
    ]
    return segs


def fig_planA():
    lm = np.load(LM_PATH)
    ref = np.load(REF_PATH)
    n_ref = ref.shape[0]
    n_vid = lm.shape[0]
    model = load_model_compat(XML_PATH)
    data = mujoco.MjData(model)

    phases = [0.0, 0.25, 0.5, 0.75]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
    for col, ph in enumerate(phases):
        vid_idx = int(round(ph * (n_vid - 1)))
        ref_idx = int(round(ph * (n_ref - 1)))
        # human
        ax = axes[0, col]
        segs = human_2d_skeleton(lm[vid_idx])
        for s in segs:
            xs = [s[0][0], s[1][0]]
            ys = [s[0][1], s[1][1]]
            ax.plot(xs, ys, 'k-', linewidth=2)
        ax.scatter([lm[vid_idx, L_HIP, 0], lm[vid_idx, L_HIP, 0]],
                   [-lm[vid_idx, L_HIP, 1], -lm[vid_idx, L_HIP, 1]],
                   s=20, c='r', zorder=3)
        ax.set_title(f'Human  (phase {ph:.2f})', fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(1.05, -0.05)
        ax.axis('off')
        # robot
        ax = axes[1, col]
        segs = robot_2d_skeleton(model, data, ref[ref_idx])
        for s in segs:
            xs = [s[0][0], s[1][0]]
            ys = [s[0][1], s[1][1]]
            ax.plot(xs, ys, 'b-', linewidth=2)
        ax.set_title(f'Pandaman  (phase {ph:.2f})', fontsize=8)
        ax.set_xlim(-0.15, 0.45)
        ax.set_ylim(0.35, 1.05)
        ax.axis('off')
        ax.set_aspect('equal')

    axes[0, 0].set_ylabel('Human\nMediaPipe', fontsize=9)
    axes[1, 0].set_ylabel('Pandaman\nMuJoCo FK', fontsize=9)
    fig.suptitle('Video-to-Robot Mapping at Matched Gait Phases',
                 fontweight='bold')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_mapping_sidebyside.png")
    fig.savefig(out, dpi=200)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"Saved: {out} (+.pdf)")


# ----------------------------------------------------------------------
# Plan C: human raw joint angles vs mapped robot angles
# ----------------------------------------------------------------------
def fig_planC():
    lm = np.load(LM_PATH)
    ref = np.load(REF_PATH)
    fps = 30.0
    hj = extract_human_joints(lm)
    # build one-cycle human angles aligned to robot ref (same pipeline)
    from retarget import extract_cycle
    hj_cycle, _ = extract_cycle(lm, hj, fps, n_cycle_out=ref.shape[0])
    hj_cycle = symmetrize_cycle(hj_cycle)
    # scales used in mapping
    SC = {"hip_scale": 0.55, "knee_scale": 0.85, "ankle_scale": 0.30}
    t = np.arange(ref.shape[0]) / ref.shape[0] * 0.64

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.5), sharex=True)
    rows = [
        ("Hip", 0, [2, 7], SC["hip_scale"]),
        ("Knee", 1, [3, 8], SC["knee_scale"]),
        ("Ankle", 2, [4, 9], SC["ankle_scale"]),
    ]
    for ax, (title, hj_idx, ref_idx, scale) in zip(axes, rows):
        # human left-leg angle (mirrored sign for display)
        hL = hj_cycle[:, hj_idx]
        hR = hj_cycle[:, 3 + hj_idx]
        ax.plot(t, hL, '--', color='#999999', linewidth=1.2,
                label='Human (raw, left/right)')
        ax.plot(t, hR, '--', color='#999999', linewidth=1.2)
        ax.plot(t, ref[:, ref_idx[0]], color='#1f77b4', linewidth=1.8,
                label='Robot ref (left)')
        ax.plot(t, ref[:, ref_idx[1]], color='#d62728', linewidth=1.8,
                label='Robot ref (right)')
        ax.set_ylabel(f'{title} (rad)')
        ax.grid(True, alpha=0.3)
        if title == "Hip":
            ax.legend(loc='upper right', fontsize=7, frameon=False)
    axes[-1].set_xlabel('Time within cycle (s)')
    fig.suptitle('Human Joint Angles vs Mapped Robot Reference '
                 '(scale: hip 0.55, knee 0.85, ankle 0.30)',
                 fontweight='bold', fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_mapping_joints.png")
    fig.savefig(out, dpi=200)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"Saved: {out} (+.pdf)")


if __name__ == "__main__":
    fig_planA()
    fig_planC()
    print("Done.")
