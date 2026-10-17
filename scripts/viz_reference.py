import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retarget import JOINT_NAMES

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.unicode_minus': False, 'axes.linewidth': 1.0,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ref = np.load(sys.argv[1] if len(sys.argv) > 1
              else os.path.join(ROOT, "data", "reference", "reference_traj.npy"))
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures")
os.makedirs(out_dir, exist_ok=True)

n = ref.shape[0]
t = np.arange(n) / n * 0.64  # robot cycle time

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
groups = [
    ("Hip pitch (rad)", [2, 7]),
    ("Knee (rad)", [3, 8]),
    ("Ankle (rad)", [4, 9]),
]
colors = {'L': '#3498db', 'R': '#e74c3c'}
for ax, (title, idxs) in zip(axes, groups):
    ax.plot(t, ref[:, idxs[0]], color=colors['L'], label='Left', linewidth=2)
    ax.plot(t, ref[:, idxs[1]], color=colors['R'], label='Right', linewidth=2)
    ax.set_ylabel(title)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('Time within cycle (s)')
axes[0].set_title('Pandaman reference trajectory from human video '
                  '(one gait cycle)')
plt.tight_layout()
out = os.path.join(out_dir, 'd2_reference_traj.png')
plt.savefig(out, dpi=200)
plt.savefig(out.replace('.png', '.pdf'))
print(f"Saved: {out}")

# quick stats
print("\nPer-joint amplitude (rad):")
for j, name in enumerate(JOINT_NAMES):
    if j in (0, 1, 5, 6):
        continue
    print(f"  {name:22s} range [{ref[:, j].min():+.3f}, {ref[:, j].max():+.3f}]")
