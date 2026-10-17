import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(data_dir):
    lm = np.load(os.path.join(data_dir, "landmarks.npy"))
    vis = np.load(os.path.join(data_dir, "visibility.npy"))
    return lm, vis


def plot_trajectories(lm, vis, out_path):
    """Plot x/y/z trajectories of the 9 lower-body landmarks over time."""
    pts = {
        "L_hip": 23, "R_hip": 24,
        "L_knee": 25, "R_knee": 26,
        "L_ankle": 27, "R_ankle": 28,
        "L_heel": 29, "R_heel": 30,
        "L_foot": 31, "R_foot": 32,
    }
    n = lm.shape[0]
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    axes[0].set_title("X (lateral)", fontweight='bold')
    axes[1].set_title("Y (vertical, up)", fontweight='bold')
    axes[2].set_title("Z (depth)", fontweight='bold')
    colors = plt.cm.tab10(np.linspace(0, 1, len(pts)))
    for (name, idx), c in zip(pts.items(), colors):
        for ax, dim in zip(axes, range(3)):
            ax.plot(lm[:, idx, dim], color=c, alpha=0.8, label=name)
    axes[0].legend(ncol=5, fontsize=8, loc='upper right')
    axes[0].set_ylabel("norm x"); axes[1].set_ylabel("norm y"); axes[2].set_ylabel("norm z")
    axes[2].set_xlabel("frame")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_path.replace('.png', '.pdf'))
    print(f"Saved: {out_path}")


def plot_skeleton_sequence(lm, out_path, step=20, max_frames=None):
    """Draw 2D skeleton (x-y) every `step` frames to check gait phase."""
    n = lm.shape[0]
    if max_frames:
        n = min(n, max_frames)
    idxs = list(range(0, n, step))
    cols = 8
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    axes = np.atleast_2d(axes)
    bones = [
        (11, 23), (12, 24), (23, 25), (24, 26),
        (25, 27), (26, 28), (27, 29), (28, 30),
        (29, 31), (30, 32), (23, 24),
    ]
    for ax, fi in zip(axes.flatten(), idxs):
        x = lm[fi, :, 0]
        y = lm[fi, :, 1]
        for (i, j) in bones:
            ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', linewidth=2)
        ax.scatter(x[23:33], y[23:33], s=30, c='red', zorder=3)
        ax.scatter(x[11:13], y[11:13], s=30, c='blue', zorder=3)
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(1.1, -0.1)
        ax.set_title(f"f{fi}", fontsize=8)
        ax.axis('off')
    for ax in axes.flatten()[len(idxs):]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_path.replace('.png', '.pdf'))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)
    data_dir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "data", "landmarks")
    fig_dir = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    lm, vis = load(data_dir)
    print(f"landmarks shape: {lm.shape}, detection: {np.any(~np.isnan(lm), axis=(1,2)).mean():.1%}")
    plot_trajectories(lm, vis, os.path.join(fig_dir, "d1_landmark_traj.png"))
    plot_skeleton_sequence(lm, os.path.join(fig_dir, "d1_skeleton_seq.png"))
