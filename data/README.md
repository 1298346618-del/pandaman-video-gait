# Data

## `landmarks/` — MediaPipe output for the source video

Produced by `scripts/extract_pose.py` from a single monocular side-view walking clip.

| File | Shape / content |
|---|---|
| `landmarks.npy` | `(356, 33, 3)` MediaPipe Pose landmarks, normalised image coordinates |
| `visibility.npy` | `(356, 33)` per-landmark visibility |
| `presence.npy` | `(356, 33)` per-landmark presence |
| `frame_indices.npy` | frame ids corresponding to the rows |
| `meta.json` | source video name, fps, frame counts, detection rate |

The full 356-frame clip was detected and tracked (detection rate 1.00, 29.97 fps, 11.9 s).

## Source video

The original clip is **Mixkit stock footage, video ID 4855** (side-view walking person,
30 fps, 11.9 s). It is **not** redistributed here: the Mixkit free license permits use in
projects but does not grant redistribution of the clip as a standalone asset.

Because the landmarks above are shipped, **every stage after pose estimation is
reproducible without the video**. To re-run the first stage, download the clip yourself
(Mixkit #4855 or any side-view walking video) and run:

```bash
python scripts/extract_pose.py path/to/walk_sideview.mp4 data/landmarks
```

## `reference/` — phase-indexed robot references

| File | Content |
|---|---|
| `reference_traj.npy` | `(120, 10)` reference joint angles for condition **C** (smoothed + L/R symmetrised), one gait cycle, 120 phase samples |
| `reference_traj_raw.npy` | `(120, 10)` condition **D** variant — Savitzky–Golay smoothing and L/R symmetrisation omitted |
| `ref_stance.npy` | stance-phase mask used to initialise the environment |
| `ref_foot_z.npy` | reference foot height profile |
| `meta.json` | stride period (1.0788 s human → 0.64 s robot), joint signs, default angles |
| `meta_raw.json` | exact processing list retained/omitted for condition D |

Robot gait cycle is fixed at **0.64 s** so that the control interface is identical to the
RA-PPO baseline; only the cadence is time-warped, never the angular morphology.

Regenerate with:

```bash
python scripts/retarget.py          # → reference_traj.npy + meta.json
python scripts/retarget_raw.py      # → reference_traj_raw.npy + meta_raw.json
python scripts/export_phase_info.py # → ref_stance.npy
```
