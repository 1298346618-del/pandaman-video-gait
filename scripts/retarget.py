import os
import sys
import json
import numpy as np
import mujoco
from scipy import signal as sg
from scipy.fft import rfft, rfftfreq


# ---------------------------------------------------------------------------
# Pandaman joint configuration (from sb3_train/config.py + pandaman.xml)
# ---------------------------------------------------------------------------
JOINT_NAMES = [
    "left_leg_roll_joint", "left_leg_yaw_joint", "left_leg_pitch_joint",
    "left_knee_joint", "left_ankle_joint",
    "right_leg_roll_joint", "right_leg_yaw_joint", "right_leg_pitch_joint",
    "right_knee_joint", "right_ankle_joint",
]
DEFAULT_ANGLES = np.array([
    0.0, 0.0, 0.17, -0.34, -0.09,
    0.0, 0.0, -0.17, 0.34, 0.09,
], dtype=np.float64)
JOINT_LIMITS = np.array([
    [-0.23, 0.23], [-1.0, 1.0], [-1.5, 1.5], [-1.8, 0.0], [-0.75, 0.75],
    [-0.23, 0.23], [-1.0, 1.0], [-1.5, 1.5], [0.0, 1.8], [-0.75, 0.75],
], dtype=np.float64)

# MediaPipe landmark indices
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32
L_SHOULDER, R_SHOULDER = 11, 12

LEGS = ["left", "right"]


# ---------------------------------------------------------------------------
# 1. Human joint angle extraction (sagittal-plane signed angles)
# ---------------------------------------------------------------------------
def signed_angle(v1, v2, normal):
    """Signed angle from v1 to v2 around `normal` in [-pi, pi]."""
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    c = np.clip(c, -1.0, 1.0)
    ang = np.arccos(c)
    cross = np.cross(v1, v2)
    if np.dot(cross, normal) < 0:
        ang = -ang
    return ang


def extract_human_joints(lm):
    """From (N,33,3) MediaPipe normalized landmarks, compute per-frame human
    joint angles in the sagittal plane.

    MediaPipe: x = image right, y = image DOWN, z = depth.
    We convert to a body frame: x_f = image x (walking dir), y_up = -y.
    Returns (N, 6): [hip_pitch_L, knee_L, ankle_L, hip_pitch_R, knee_R, ankle_R]
    """
    N = lm.shape[0]
    out = np.zeros((N, 6), dtype=np.float64)
    for i in range(N):
        p = lm[i]
        # world-ish frame: x = fwd (image x), y = up (-image y), z = depth
        def pt(idx):
            return np.array([p[idx, 0], -p[idx, 1], p[idx, 2]], dtype=np.float64)

        mid_hip = (pt(L_HIP) + pt(R_HIP)) / 2
        mid_shoulder = (pt(L_SHOULDER) + pt(R_SHOULDER)) / 2
        torso = mid_hip - mid_shoulder          # points downward
        torso[2] = 0.0                          # sagittal plane only
        if np.linalg.norm(torso) < 1e-6:
            torso = np.array([0.0, -1.0, 0.0])
        torso = torso / np.linalg.norm(torso)   # down direction
        normal = np.array([0.0, 0.0, 1.0])      # lateral axis of sagittal plane

        for j, side in enumerate(LEGS):
            if side == "left":
                hip, knee, ankle, heel = L_HIP, L_KNEE, L_ANKLE, L_HEEL
            else:
                hip, knee, ankle, heel = R_HIP, R_KNEE, R_ANKLE, R_HEEL
            hip_p, knee_p, ankle_p, heel_p = pt(hip), pt(knee), pt(ankle), pt(heel)
            thigh = knee_p - hip_p
            shank = ankle_p - knee_p
            foot = heel_p - ankle_p

            # hip pitch: signed angle from torso-down to thigh (fwd swing = +)
            hip_pitch = signed_angle(torso, thigh, normal)
            # knee: signed angle from thigh to shank (flexion sign raw)
            knee_ang = signed_angle(thigh, shank, normal)
            # ankle: signed angle from shank to foot
            ankle_ang = signed_angle(shank, foot, normal)
            out[i, j * 3 + 0] = hip_pitch
            out[i, j * 3 + 1] = knee_ang
            out[i, j * 3 + 2] = ankle_ang
    return out


# ---------------------------------------------------------------------------
# 2. MuJoCo sign calibration: for each joint, +0.1 rad moves the foot which way?
# ---------------------------------------------------------------------------
def load_model_compat(xml_path):
    """Load MJCF, stripping legacy attributes unsupported by MuJoCo >= 3.x."""
    with open(xml_path, "r", encoding="utf-8") as f:
        xml = f.read()
    xml = xml.replace(' sensornoise="enable"', "")
    xml = xml.replace(' frictionloss="enable"', "")
    tmp = os.path.join(os.path.dirname(os.path.abspath(xml_path)),
                       "_tmp_compat.xml")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(xml)
    try:
        model = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)
    return model


def calibrate_joint_signs(xml_path):
    """Determine, for each robot joint, whether a positive angle moves the
    foot forward (+x) or backward. Returns 10-vector of +/-1."""
    model = load_model_compat(xml_path)
    data = mujoco.MjData(model)
    signs = np.zeros(10)
    dof_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
               for n in JOINT_NAMES]
    qadr = [model.jnt_qposadr[i] for i in dof_ids]
    foot_bodies = ["left_ankle_link", "right_ankle_link"]
    for j in range(10):
        mujoco.mj_resetData(model, data)
        for k, a in enumerate(DEFAULT_ANGLES):
            data.qpos[qadr[k]] = a
        mujoco.mj_forward(model, data)
        leg = 0 if j < 5 else 1
        fb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot_bodies[leg])
        x0 = data.xpos[fb][0]
        data.qpos[qadr[j]] = DEFAULT_ANGLES[j] + 0.1
        mujoco.mj_forward(model, data)
        x1 = data.xpos[fb][0]
        signs[j] = 1.0 if x1 > x0 else -1.0
    return signs


# ---------------------------------------------------------------------------
# 3. Cycle extraction + resampling
# ---------------------------------------------------------------------------
def estimate_period(lm, fps):
    """Full gait-cycle period via FFT of left heel height (0.93 Hz human
    walking => ~1.08 s). Uses heel z (lateral) not directly; use heel
    image-y which oscillates once per full stride."""
    heel_y = -lm[:, L_HEEL, 1]          # image y down -> up
    heel_y = heel_y - np.nanmean(heel_y)
    n = len(heel_y)
    spec = np.abs(rfft(heel_y))
    freqs = rfftfreq(n, d=1.0 / fps)
    mask = freqs > 0.3
    if not mask.any():
        return 1.0
    f0 = freqs[mask][np.argmax(spec[mask])]
    return 1.0 / f0


def extract_cycle(lm, hj, fps, n_cycle_out=120):
    """Average full gait cycles into one representative cycle aligned on the
    left-knee maximum-flexion frame. Returns (n_cycle_out, 6)."""
    T = estimate_period(lm, fps)
    n_cycle = int(round(T * fps))
    # alignment anchor: left knee minimum (max flexion)
    knee = hj[:, 1]
    # find local minima positions
    from scipy.signal import argrelextrema
    mn = argrelextrema(knee, np.less, order=5)[0]
    # keep minima spaced roughly n_cycle apart, use as cycle starts
    starts = []
    for m in mn:
        if not starts or m - starts[-1] >= n_cycle * 0.7:
            starts.append(m)
    if len(starts) < 2:
        # fallback: grid from frame 0
        starts = list(range(0, len(hj) - n_cycle, n_cycle))
    cycles = []
    for s in starts:
        if s + n_cycle <= len(hj):
            cycles.append(hj[s:s + n_cycle])
    if not cycles:
        raise RuntimeError("Not enough full cycles in data")
    cyc = np.mean(cycles, axis=0)
    x_old = np.linspace(0, 1, n_cycle)
    x_new = np.linspace(0, 1, n_cycle_out, endpoint=False)
    out = np.stack([np.interp(x_new, x_old, cyc[:, k]) for k in range(6)], axis=1)
    return out, T


# ---------------------------------------------------------------------------
# 4. Map human angles -> robot joint reference
# ---------------------------------------------------------------------------
def map_to_robot(hj_cycle, signs, cfg=None):
    """hj_cycle: (M,6) [hipL,kneeL,ankleL,hipR,kneeR,ankleR]
    signs: (10,) robot joint sign from MuJoCo calibration
    Returns (M,10) reference joint trajectory (absolute angles).

    Sign conventions (verified via MuJoCo FK):
      - hip pitch: positive human angle = thigh forward; robot moves foot
        forward for +pitch on left leg, -pitch on right leg (mirrored).
      - knee: human flexion (kn<0, shank back) must bend left knee more
        negative (range [-1.8,0]) and right knee more positive (range [0,1.8]).
      - ankle: scaled small to stay within +/-0.75 limits.
    """
    if cfg is None:
        cfg = {
            "hip_scale": 0.55, "knee_scale": 0.85, "ankle_scale": 0.30,
        }
    M = hj_cycle.shape[0]
    ref = np.zeros((M, 10))
    for j, side in enumerate(LEGS):
        hp = hj_cycle[:, j * 3 + 0] * cfg["hip_scale"]
        kn = hj_cycle[:, j * 3 + 1] * cfg["knee_scale"]
        an = hj_cycle[:, j * 3 + 2] * cfg["ankle_scale"]
        o = j * 5
        # hip pitch: mirror on right leg (signs[o+2] == +1 left, -1 right)
        ref[:, o + 2] = DEFAULT_ANGLES[o + 2] + signs[o + 2] * hp
        # knee flexion: left bends negative, right bends positive
        if side == "left":
            ref[:, o + 3] = DEFAULT_ANGLES[o + 3] + kn
        else:
            ref[:, o + 3] = DEFAULT_ANGLES[o + 3] - kn
        # ankle: mirror similarly (signs[o+4] == -1 for both)
        ref[:, o + 4] = DEFAULT_ANGLES[o + 4] + signs[o + 4] * an
        # roll / yaw stay at default (flat-ground walking)
        ref[:, o + 0] = DEFAULT_ANGLES[o + 0]
        ref[:, o + 1] = DEFAULT_ANGLES[o + 1]
    return ref


def clip_to_limits(ref):
    ref = ref.copy()
    for j in range(10):
        ref[:, j] = np.clip(ref[:, j], JOINT_LIMITS[j, 0], JOINT_LIMITS[j, 1])
    return ref


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def symmetrize_cycle(hj):
    """Average left/right legs with half-cycle mirroring so the robot gait is
    perfectly symmetric (removes monocular noise asymmetry). Input/Output:
    (M,6) [hipL,kneeL,ankleL,hipR,kneeR,ankleR]."""
    M = hj.shape[0]
    half = M // 2
    out = hj.copy()
    for k in range(3):  # hip, knee, ankle
        L = hj[:, k]
        R = hj[:, 3 + k]
        L_sym = 0.5 * (L + np.roll(R, half))
        R_sym = 0.5 * (R + np.roll(L, half))
        out[:, k] = L_sym
        out[:, 3 + k] = R_sym
    return out


def build_reference(landmarks_path, xml_path, out_path, meta_path,
                    n_cycle_out=120, smooth_window=11, smooth_order=3):
    lm = np.load(landmarks_path)
    fps = 30.0
    hj = extract_human_joints(lm)
    # smooth per column
    for k in range(6):
        col = hj[:, k]
        if not np.isnan(col).any():
            hj[:, k] = sg.savgol_filter(col, smooth_window, smooth_order)
    hj_cycle, T_human = extract_cycle(lm, hj, fps, n_cycle_out)
    hj_cycle = symmetrize_cycle(hj_cycle)
    signs = calibrate_joint_signs(xml_path)
    ref = map_to_robot(hj_cycle, signs)
    ref = clip_to_limits(ref)
    np.save(out_path, ref.astype(np.float32))
    meta = {
        "n_cycle_out": n_cycle_out,
        "cycle_time_robot": 0.64,
        "human_stride_period_s": float(T_human),
        "fps": fps,
        "joint_signs": signs.tolist(),
        "joint_names": JOINT_NAMES,
        "default_angles": DEFAULT_ANGLES.tolist(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return ref, meta


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    lm_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "data", "landmarks", "landmarks.npy")
    xml = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(ROOT, "assets", "robots", "pandaman", "mjcf", "pandaman.xml")
    out = sys.argv[3] if len(sys.argv) > 3 else \
        os.path.join(ROOT, "data", "reference", "reference_traj.npy")
    meta = sys.argv[4] if len(sys.argv) > 4 else \
        os.path.join(ROOT, "data", "reference", "meta.json")
    ref, m = build_reference(lm_path, xml, out, meta)
    print(f"Saved: {out}  shape={ref.shape}")
    print(f"Human stride period: {m['human_stride_period_s']:.3f}s")
    print(f"Joint signs: {np.array(m['joint_signs']).astype(int)}")
    for j, name in enumerate(JOINT_NAMES):
        lo, hi = ref[:, j].min(), ref[:, j].max()
        print(f"  {name:22s} min={lo:+.3f} max={hi:+.3f} "
              f"limit=[{JOINT_LIMITS[j,0]:+.2f},{JOINT_LIMITS[j,1]:+.2f}]")
