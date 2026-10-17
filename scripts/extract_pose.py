import os
import sys
import numpy as np
import cv2
import mediapipe as mp


POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Lower-body landmarks needed for retargeting (MediaPipe indices)
LOWER_BODY_INDICES = {
    "mid_hip": None,        # computed as midpoint of 23/24
    "mid_shoulder": None,   # computed as midpoint of 11/12
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

DEFAULT_MIN_DETECTION_CONFIDENCE = 0.5
DEFAULT_MIN_TRACKING_CONFIDENCE = 0.5


def extract_landmarks(video_path, output_dir, frame_step=1,
                      min_det_conf=DEFAULT_MIN_DETECTION_CONFIDENCE,
                      min_track_conf=DEFAULT_MIN_TRACKING_CONFIDENCE,
                      verbose=True):
    """Extract 33x3 world landmarks from a video using MediaPipe Pose.

    Returns dict with keys: landmarks (N,33,3), visibility (N,33), presence (N,),
    fps, timestamps (N,), frame_indices (N,).
    Saves landmarks.npy / visibility.npy / meta.json into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=min_det_conf,
        min_tracking_confidence=min_track_conf,
    )

    landmarks = []
    visibility = []
    presence = []
    frame_indices = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_step != 0:
            idx += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            lm = np.array([[p.x, p.y, p.z] for p in result.pose_landmarks.landmark])
            vis = np.array([p.visibility for p in result.pose_landmarks.landmark])
            landmarks.append(lm)
            visibility.append(vis)
            presence.append(1)
        else:
            landmarks.append(np.full((33, 3), np.nan))
            visibility.append(np.zeros(33))
            presence.append(0)
        frame_indices.append(idx)
        idx += 1

    cap.release()
    pose.close()

    landmarks = np.array(landmarks, dtype=np.float32)   # (N, 33, 3) normalized coords
    visibility = np.array(visibility, dtype=np.float32) # (N, 33)
    presence = np.array(presence, dtype=np.int8)        # (N,)

    np.save(os.path.join(output_dir, "landmarks.npy"), landmarks)
    np.save(os.path.join(output_dir, "visibility.npy"), visibility)
    np.save(os.path.join(output_dir, "presence.npy"), presence)
    np.save(os.path.join(output_dir, "frame_indices.npy"), np.array(frame_indices))

    meta = {
        "video": os.path.basename(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "extracted_frames": len(frame_indices),
        "detected_frames": int(presence.sum()),
        "detection_rate": float(presence.mean()),
        "normalized_coords": True,
    }
    import json
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"[extract_landmarks] video={os.path.basename(video_path)} fps={fps:.1f}")
        print(f"[extract_landmarks] extracted={meta['extracted_frames']} "
              f"detected={meta['detected_frames']} rate={meta['detection_rate']:.2%}")

    return {
        "landmarks": landmarks,
        "visibility": visibility,
        "presence": presence,
        "fps": fps,
        "frame_indices": np.array(frame_indices),
        "meta": meta,
    }


def interpolate_missing(data, axis=0):
    """Linearly interpolate NaN values along a given axis."""
    arr = data.copy()
    n = arr.shape[axis]
    for i in range(n):
        sl = [slice(None)] * arr.ndim
        sl[axis] = i
        arr[tuple(sl)] = interpolate_1d(arr[tuple(sl)])
    return arr


def interpolate_1d(v):
    if v.ndim == 1:
        idx = np.arange(len(v))
        nan = np.isnan(v)
        if nan.all():
            return v
        if nan.any():
            v = v.copy()
            v[nan] = np.interp(idx[nan], idx[~nan], v[~nan])
        return v
    out = v.copy()
    for j in range(v.shape[0]):
        out[j] = interpolate_1d(v[j])
    return out


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    video = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "data", "video", "walk_sideview.mp4")
    outdir = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(ROOT, "data", "landmarks")
    extract_landmarks(video, outdir)
