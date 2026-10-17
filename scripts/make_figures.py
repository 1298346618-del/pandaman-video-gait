import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.unicode_minus': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1.3,
    'figure.titlesize': 10,
    'figure.titleweight': 'bold',
    'axes.titleweight': 'bold',
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(ROOT, "figures")
DATA_DIR = os.path.join(ROOT, "data")
RES_DIR = os.path.join(ROOT, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


def save(fig, name):
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {out} (+.pdf)")


def fig1_pipeline():
    """System pipeline: white boxes, black thin borders, tech-detail subtitles.
    Full-width layout for figure* (two-column)."""
    fig, ax = plt.subplots(figsize=(17, 3.4))
    ax.axis('off')
    boxes = [
        ("1. Human Walking\nVideo", "monocular,\n30 fps"),
        ("2. Pose Estimation\n(MediaPipe)", "33 landmarks,\nlower-body 10 pts"),
        ("3. Kinematic Retarget\n+ Post-Processing", "S-G smoothing,\nlimit proj., foot\nalign., symmetrize"),
        ("4. Phase-Coupled\nReference Buffer", "FFT period 1.08 s\n-> 0.64 s cycle"),
        ("5. PPO Training\n(RA-PPO)", "PD control,\ndomain random."),
        ("6. MuJoCo\nEvaluation", "RMSE, push\nrecovery"),
    ]
    n = len(boxes)
    x = 0.005
    w, h = 0.155, 0.5
    positions = []
    for (label, tech) in boxes:
        box = FancyBboxPatch((x, 0.30), w, h, boxstyle="round,pad=0.006",
                             linewidth=1.0, edgecolor="black",
                             facecolor="white")
        ax.add_patch(box)
        ax.text(x + w / 2, 0.62, label, ha='center', va='center',
                fontsize=9, fontweight='bold')
        ax.text(x + w / 2, 0.36, tech, ha='center', va='center',
                fontsize=7, color='#444444')
        positions.append((x + w, 0.55))
        x += w + 0.012
    for i in range(len(positions) - 1):
        ar = FancyArrowPatch((positions[i][0] - 0.004, 0.55),
                             (positions[i + 1][0] - w - 0.008, 0.55),
                             arrowstyle='-|>', mutation_scale=13,
                             color="black", linewidth=1.0)
        ax.add_patch(ar)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1)
    fig.tight_layout(pad=0.3)
    save(fig, "fig1_pipeline.png")


def fig2_reference():
    """Reference trajectory: (a) hip (b) knee (c) ankle.
    Left solid, right dash-dot, sinusoidal dashed. Shared bottom legend."""
    ref = np.load(os.path.join(DATA_DIR, "reference", "reference_traj.npy"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from retarget import DEFAULT_ANGLES
    n = ref.shape[0]
    t = np.arange(n) / n * 0.64
    s1 = 0.26
    s2 = 2 * s1
    sin = np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False))
    sin_l = np.where(sin < 0, sin, 0)
    sin_r = np.where(sin > 0, sin, 0)
    sin_ref = np.zeros((n, 10))
    sin_ref[:, 2] = -sin_l * s1 + DEFAULT_ANGLES[2]
    sin_ref[:, 3] = sin_l * s2 + DEFAULT_ANGLES[3]
    sin_ref[:, 4] = sin_l * s1 + DEFAULT_ANGLES[4]
    sin_ref[:, 7] = -sin_r * s1 + DEFAULT_ANGLES[7]
    sin_ref[:, 8] = sin_r * s2 + DEFAULT_ANGLES[8]
    sin_ref[:, 9] = sin_r * s1 + DEFAULT_ANGLES[9]

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.0), sharex=True)
    groups = [("Hip pitch", 2, 7), ("Knee", 3, 8), ("Ankle", 4, 9)]
    letters = ['(a)', '(b)', '(c)']
    for ax, (title, iL, iR), letter in zip(axes, groups, letters):
        ax.plot(t, sin_ref[:, iL], '--', color='black', linewidth=1.0,
                label='A: Sinusoidal baseline')
        ax.plot(t, ref[:, iL], '-', color='black', linewidth=1.3,
                label='C: Human-derived (left)')
        ax.plot(t, ref[:, iR], '-.', color='black', linewidth=1.3,
                label='C: Human-derived (right)')
        ax.text(0.02, 0.95, f'{letter} {title}', transform=ax.transAxes,
                fontsize=8, va='top')
        ax.set_ylabel('Angle (rad)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.64)
    axes[-1].set_xlabel('Time within cycle (s)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "fig2_reference.png")


def _load_training_curves(exp_dirs):
    import glob
    from tensorboard.backend.event_processing.event_accumulator import \
        EventAccumulator
    curves = {}
    for exp, dirs in exp_dirs.items():
        all_curves = []
        for d in dirs:
            best = None
            for p in glob.glob(os.path.join(d, "**", "events.out*"),
                               recursive=True):
                if not os.path.exists(p):
                    continue
                try:
                    ea = EventAccumulator(p)
                    ea.Reload()
                    tags = ea.Tags()['scalars']
                    if 'rollout/ep_rew_mean' in tags:
                        ev = ea.Scalars('rollout/ep_rew_mean')
                        if best is None or len(ev) > len(best):
                            best = ev
                except Exception:
                    pass
            if best:
                all_curves.append(([e.step / 1e6 for e in best],
                                   [e.value for e in best]))
        all_curves = [c for c in all_curves if c[0][-1] > 2.0]
        if all_curves:
            curves[exp] = all_curves
    return curves


def fig3_training():
    """Training curves: mean + std shading per condition (figure* width)."""
    res = os.path.join(ROOT, "results")
    exp_dirs = {
        "A: Sinusoidal baseline": [
            os.path.join(res, "improved_ab", "improved_ab_2"),
        ],
        "C: Lightweight video ref. (Ours)": [
            os.path.join(res, "robio_ablation", "robio_ablation_1"),
            os.path.join(res, "robio_ablation_s100"),
            os.path.join(res, "robio_ablation_s200"),
        ],
        "B: Explicit imitation": [
            os.path.join(res, "robio_human", "robio_human_2"),
            os.path.join(res, "robio_human_s100"),
            os.path.join(res, "robio_human_s200"),
        ],
        "D: Raw video (no post-proc.)": [
            os.path.join(res, "robio_ablation_raw_s42"),
        ],
    }
    curves = _load_training_curves(exp_dirs)
    colors = {'A: Sinusoidal baseline': '#999999',
              'C: Lightweight video ref. (Ours)': '#1f77b4',
              'B: Explicit imitation': '#d62728',
              'D: Raw video (no post-proc.)': '#555555'}
    linestyles = {'A: Sinusoidal baseline': '--',
                  'C: Lightweight video ref. (Ours)': '-',
                  'B: Explicit imitation': '-.',
                  'D: Raw video (no post-proc.)': ':'}
    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    for label, all_curves in curves.items():
        xs = None
        vals = []
        for steps, vs in all_curves:
            if xs is None:
                xs = np.array(steps)
            # interpolate onto common grid
            v = np.interp(xs, np.array(steps), np.array(vs))
            vals.append(v)
        if len(vals) == 1:
            ax.plot(xs, vals[0], color=colors[label],
                    linestyle=linestyles[label], linewidth=1.5, label=label)
        else:
            arr = np.array(vals)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax.plot(xs, mean, color=colors[label],
                    linestyle=linestyles[label], linewidth=1.5, label=label)
            ax.fill_between(xs, mean - std, mean + std,
                            color=colors[label], alpha=0.2)
    ax.set_xlabel('Training steps ($\\times10^{6}$)')
    ax.set_ylabel('Mean episode reward')
    ax.set_ylim(0, 10500)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.4)
    save(fig, "fig3_training.png")


def fig4_metrics():
    """Metric bars: (a) reward (b) RMSE (c) push recovery.
    Horizontal 1x3 layout for full-width placement (figure*)."""
    jpath = os.path.join(RES_DIR, "comparison_robio.json")
    if not os.path.exists(jpath):
        print(f"[fig4] SKIP: {jpath} not found")
        return
    data = json.load(open(jpath, encoding="utf-8"))
    # order: A (baseline) -> B (explicit) -> C (ours) -> D (raw)
    order = ["baseline", "human", "ablation", "raw"]
    labels = ["A: Sinusoidal\nbaseline", "B: Explicit\nimitation",
              "C: Lightweight\nvideo ref. (Ours)", "D: Raw video\n(no post-proc.)"]
    # grayscale-safe: A light-gray hatched, B mid-gray grid, C solid black
    # (our method), D light-gray dotted
    colors = ['#bdbdbd', '#7f7f7f', '#000000', '#d9d9d9']
    hatches = ['///', 'xxx', '', '...']
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.9))

    def _get(o, k, field):
        v = data[o].get(k)
        if isinstance(v, dict):
            return v.get(field, 0)
        return v if field == "mean" and v is not None else 0

    # (a) total reward
    ax = axes[0]
    means = [_get(o, "total_reward", "mean") for o in order]
    stds = [_get(o, "total_reward", "std") for o in order]
    bars = ax.bar(range(4), means, color=colors, yerr=stds, capsize=4,
                  edgecolor='black', linewidth=0.8, width=0.6)
    for b, h_ in zip(bars, hatches):
        b.set_hatch(h_)
    ax.text(0.02, 0.95, '(a) Total reward', transform=ax.transAxes,
            fontsize=8, va='top', fontweight='bold')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, max(means) * 1.35)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=7)
    ax.text(bars[2].get_x() + bars[2].get_width() / 2,
            means[2] + stds[2] + max(means) * 0.06, '★',
            ha='center', va='bottom', fontsize=9)

    # (b) imitation RMSE
    ax = axes[1]
    means = [_get(o, "imitation_rmse", "mean") for o in order]
    stds = [_get(o, "imitation_rmse", "std") for o in order]
    bars = ax.bar(range(4), means, color=colors, yerr=stds, capsize=4,
                  edgecolor='black', linewidth=0.8, width=0.6)
    for b, h_ in zip(bars, hatches):
        b.set_hatch(h_)
    ax.text(0.02, 0.95, '(b) Imitation RMSE (rad)',
            transform=ax.transAxes, fontsize=8, va='top', fontweight='bold')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, max(means) * 1.35)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    ax.text(bars[2].get_x() + bars[2].get_width() / 2,
            means[2] + stds[2] + max(means) * 0.06, '★',
            ha='center', va='bottom', fontsize=9)

    # (c) push recovery
    ax = axes[2]
    means = [_get(o, "push_recovery_rate", "mean") for o in order]
    bars = ax.bar(range(4), means, color=colors,
                  edgecolor='black', linewidth=0.8, width=0.6)
    for b, h_ in zip(bars, hatches):
        b.set_hatch(h_)
    ax.text(0.02, 0.95, '(c) Push recovery rate',
            transform=ax.transAxes, fontsize=8, va='top', fontweight='bold')
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0.90, 1.05)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val*100:.0f}%', ha='center', va='bottom', fontsize=7,
                color='black')
    ax.text(bars[2].get_x() + bars[2].get_width() / 2,
            means[2] + 0.015, '★', ha='center', va='bottom', fontsize=9)

    # global legend (one for all three subplots)
    from matplotlib.patches import Patch
    legend_labels = [
        "A: Sinusoidal baseline", "B: Explicit imitation",
        "C: Lightweight video ref. (Ours)", "D: Raw video (no post-proc.)",
    ]
    legend_handles = [
        Patch(facecolor=colors[0], hatch=hatches[0], edgecolor='black',
              label=legend_labels[0]),
        Patch(facecolor=colors[1], hatch=hatches[1], edgecolor='black',
              label=legend_labels[1]),
        Patch(facecolor=colors[2], hatch=hatches[2], edgecolor='black',
              label=legend_labels[2]),
        Patch(facecolor=colors[3], hatch=hatches[3], edgecolor='black',
              label=legend_labels[3]),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=4,
               fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(pad=0.4)
    save(fig, "fig4_metrics.png")


if __name__ == "__main__":
    fig1_pipeline()
    fig2_reference()
    fig3_training()
    fig4_metrics()
    print("All figure scripts ran.")
