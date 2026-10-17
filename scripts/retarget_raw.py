"""Generate the minimally processed reference trajectory for ablation D.

This variant retains stride-period estimation, multi-cycle averaging,
phase resampling, robot-joint mapping/scaling, and (by default) joint-limit
clipping.  Relative to the processed reference, it omits Savitzky--Golay
smoothing and left/right half-cycle symmetrization.

The public function name, parameter names, and output paths are retained for
backward compatibility.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retarget import (extract_human_joints, extract_cycle, map_to_robot,
                      calibrate_joint_signs, clip_to_limits, JOINT_NAMES,
                      DEFAULT_ANGLES, JOINT_LIMITS)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LM_PATH = os.path.join(ROOT, "data", "landmarks", "landmarks.npy")
XML_PATH = os.path.join(ROOT, "assets", "robots", "pandaman",
                        "mjcf", "pandaman.xml")
OUT_PATH = os.path.join(ROOT, "data", "reference", "reference_traj_raw.npy")
META_PATH = os.path.join(ROOT, "data", "reference", "meta_raw.json")


def build_raw_reference(landmarks_path, xml_path, out_path, meta_path,
                        n_cycle_out=120, skip_clip=False):
    """Build the ablation-D reference.

    ``skip_clip`` is a backward-compatible flag: ``False`` (the default)
    applies joint-limit clipping, while ``True`` explicitly disables it.
    """
    lm = np.load(landmarks_path)
    fps = 30.0
    hj = extract_human_joints(lm)          # intentionally unsmoothed
    hj_cycle, T_human = extract_cycle(lm, hj, fps, n_cycle_out)
    # NOTE: no symmetrize_cycle call
    signs = calibrate_joint_signs(xml_path)
    ref = map_to_robot(hj_cycle, signs)
    clip_enabled = not skip_clip
    if clip_enabled:
        ref = clip_to_limits(ref)
    np.save(out_path, ref.astype(np.float32))

    retained_processing = [
        "fft_stride_period_estimation",
        "multi_cycle_averaging",
        f"phase_resampling_to_{n_cycle_out}_samples",
        "robot_joint_mapping_and_scaling",
    ]
    omitted_processing = [
        "savitzky_golay_smoothing",
        "left_right_half_cycle_symmetrization",
    ]
    if clip_enabled:
        retained_processing.append("joint_limit_clipping")
    else:
        omitted_processing.append("joint_limit_clipping")

    meta = {
        "reference_variant": "no_smoothing_or_lr_symmetrization",
        "retained_processing": retained_processing,
        "omitted_processing": omitted_processing,
        "clip_to_limits": clip_enabled,
        "n_cycle_out": n_cycle_out,
        "cycle_time_robot": 0.64,
        "human_stride_period_s": float(T_human),
        "fps": fps,
        "note": (
            "Ablation D omits S-G smoothing and L/R symmetrization; "
            "the processing lists record the exact executed variant."
        ),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved ablation-D reference: {out_path}  shape={ref.shape}")
    print(f"Human stride period: {T_human:.3f}s")
    for j, name in enumerate(JOINT_NAMES):
        print(f"  {name:22s} min={ref[:, j].min():+.3f} "
              f"max={ref[:, j].max():+.3f} "
              f"limit=[{JOINT_LIMITS[j,0]:+.2f},{JOINT_LIMITS[j,1]:+.2f}]")
    return ref, meta


if __name__ == "__main__":
    build_raw_reference(LM_PATH, XML_PATH, OUT_PATH, META_PATH)
