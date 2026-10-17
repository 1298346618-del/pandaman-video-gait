# -*- coding: utf-8 -*-
"""
ROBIO 2026 figures v5
- fig1_pipeline: enriched modular architecture, IEEE-polished:
  balanced module widths, annotated RA-PPO net (skip connection), light-noise
  smoothing curves, unified font sizes, phase-buffer highlight (key novelty),
  data-name labels on arrows.
- fig4_metrics: tighter vertical spacing, x-labels only on bottom panel,
  star inside C bar, value labels clear of arrows.
Output: paper/fig1_pipeline.{pdf,png}, paper/fig4_metrics.{pdf,png}
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Patch

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.unicode_minus': False,
    'axes.linewidth': 0.8,
    'mathtext.fontset': 'stix',
})

PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
RES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(PAPER_DIR, exist_ok=True)


def save(fig, name):
    out = os.path.join(PAPER_DIR, name)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {out} (+.pdf)")


# ---------------------------------------------------------------------------
# Fig.1 visual helpers
# ---------------------------------------------------------------------------
def _walker_with_landmarks(ax, cx, cy, s=1.0):
    ax.add_patch(Circle((cx, cy + 0.46 * s), 0.13 * s, fill=False, lw=1.0))
    ax.plot([cx, cx], [cy + 0.32 * s, cy - 0.04 * s], 'k-', lw=1.1)
    ax.plot([cx, cx - 0.24 * s], [cy + 0.24 * s, cy + 0.06 * s], 'k-', lw=1.0)
    ax.plot([cx - 0.24 * s, cx - 0.33 * s], [cy + 0.06 * s, cy - 0.14 * s], 'k-', lw=1.0)
    ax.plot([cx, cx + 0.24 * s], [cy + 0.24 * s, cy + 0.03 * s], 'k-', lw=1.0)
    ax.plot([cx + 0.24 * s, cx + 0.29 * s], [cy + 0.03 * s, cy - 0.20 * s], 'k-', lw=1.0)
    ax.plot([cx, cx - 0.20 * s], [cy - 0.04 * s, cy - 0.34 * s], 'k-', lw=1.1)
    ax.plot([cx - 0.20 * s, cx - 0.12 * s], [cy - 0.34 * s, cy - 0.60 * s], 'k-', lw=1.1)
    ax.plot([cx, cx + 0.12 * s], [cy - 0.04 * s, cy - 0.28 * s], 'k-', lw=1.1)
    ax.plot([cx + 0.12 * s, cx + 0.24 * s], [cy - 0.28 * s, cy - 0.60 * s], 'k-', lw=1.0)
    ax.plot([cx - 0.12 * s, cx - 0.02 * s], [cy - 0.60 * s, cy - 0.60 * s], 'k-', lw=1.1)
    ax.plot([cx + 0.24 * s, cx + 0.34 * s], [cy - 0.60 * s, cy - 0.60 * s], 'k-', lw=1.1)
    joints = [(cx, cy - 0.04 * s), (cx - 0.20 * s, cy - 0.34 * s),
              (cx + 0.12 * s, cy - 0.28 * s), (cx - 0.12 * s, cy - 0.60 * s),
              (cx + 0.24 * s, cy - 0.60 * s)]
    for jx, jy in joints:
        ax.add_patch(Circle((jx, jy), 0.045 * s, fill=True, facecolor='white',
                            edgecolor='black', lw=0.9))


def _smoothing_curves(ax, cx, cy, w=5.5, h=2.2):
    """Light-noise (gray dashed) vs smoothed (black solid) trajectory."""
    x = np.linspace(0, 2 * np.pi, 180)
    rng = np.random.RandomState(7)
    noisy = np.sin(x) + 0.12 * rng.randn(len(x))
    ax.plot(cx - w / 2 + x / (2 * np.pi) * w, cy + 0.42 * h * noisy / 1.2,
            '--', color='#A0A0A0', lw=0.6)
    ax.plot(cx - w / 2 + x / (2 * np.pi) * w, cy + 0.42 * h * np.sin(x) / 1.2,
            '-', color='black', lw=1.3)
    ax.text(cx, cy - 0.66 * h, 'raw $\\rightarrow$ smoothed $\\theta(t)$',
            ha='center', fontsize=7.5, fontstyle='italic', color='#444444')


def _phase_wheel(ax, cx, cy, r=0.9):
    ax.add_patch(Circle((cx, cy), r, fill=False, lw=1.0))
    a = np.deg2rad(40)
    ax.annotate('', xy=(cx + r * np.cos(a), cy + r * np.sin(a)), xytext=(cx, cy),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.0, mutation_scale=9))
    ax.add_patch(Circle((cx, cy), 0.06 * r, fill=True, color='black'))
    ax.text(cx, cy - r - 0.35, '$\\varphi \\in [0,1)$', ha='center', fontsize=7.5,
            fontstyle='italic', color='#444444')


def _rapo_net(ax, cx, cy, w=8.0, h=3.0):
    """RA-PPO architecture with annotated residual skip connection."""
    boxes = [('Obs.\n615-d', cx - w / 2, 1.5),
             ('Res. blk\n+ LN', cx - w / 6, 1.9),
             ('Res. blk\n+ LN', cx + w / 6, 1.9),
             ('Action\n10-d', cx + w / 2, 1.5)]
    for i, (lab, x, bw) in enumerate(boxes):
        ax.add_patch(Rectangle((x - bw / 2, cy - h / 4), bw, h / 2,
                               fill=False, edgecolor='black', lw=0.9))
        ax.text(x, cy, lab, ha='center', va='center', fontsize=7.0)
        if i < len(boxes) - 1:
            ax.annotate('', xy=(boxes[i + 1][1] - boxes[i + 1][2] / 2 - 0.05, cy),
                        xytext=(x + bw / 2 + 0.05, cy),
                        arrowprops=dict(arrowstyle='-|>', color='#333333', lw=0.8,
                                        mutation_scale=7))
    # residual skip connection: arc over the two residual blocks
    x1, x2 = cx - w / 6, cx + w / 6
    ax.annotate('', xy=(x2 + 0.95, cy + h / 4 + 0.25),
                xytext=(x1 - 0.95, cy + h / 4 + 0.25),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=0.9,
                                mutation_scale=8,
                                connectionstyle='arc3,rad=-0.18'))
    ax.text(cx, cy + h / 4 + 0.85, 'residual skip', ha='center', fontsize=7.0,
            fontstyle='italic', color='#333333')
    ax.text(cx, cy - h / 2 - 0.15, 'RA-PPO (residual-adaptive)',
            ha='center', fontsize=7.5, fontstyle='italic', color='#444444')


def _robot_render(ax, cx, cy, s=1.0):
    ax.add_patch(Rectangle((cx - 0.14 * s, cy + 0.26 * s), 0.28 * s, 0.16 * s,
                           fill=False, edgecolor='black', lw=1.0))
    ax.add_patch(Rectangle((cx - 0.24 * s, cy - 0.02 * s), 0.48 * s, 0.26 * s,
                           fill=False, edgecolor='black', lw=1.0))
    ax.plot([cx - 0.16 * s, cx - 0.24 * s], [cy - 0.02 * s, cy - 0.40 * s], 'k-', lw=1.1)
    ax.plot([cx - 0.24 * s, cx - 0.16 * s], [cy - 0.40 * s, cy - 0.62 * s], 'k-', lw=1.1)
    ax.plot([cx + 0.16 * s, cx + 0.10 * s], [cy - 0.02 * s, cy - 0.38 * s], 'k-', lw=1.1)
    ax.plot([cx + 0.10 * s, cx + 0.22 * s], [cy - 0.38 * s, cy - 0.62 * s], 'k-', lw=1.1)
    ax.plot([cx - 0.16 * s, cx - 0.02 * s], [cy - 0.62 * s, cy - 0.62 * s], 'k-', lw=1.1)
    ax.plot([cx + 0.22 * s, cx + 0.36 * s], [cy - 0.62 * s, cy - 0.62 * s], 'k-', lw=1.1)
    ax.plot([cx - 0.40 * s, cx + 0.48 * s], [cy - 0.66 * s, cy - 0.66 * s], 'k--', lw=0.8)
    ax.text(cx, cy - 0.80 * s, 'MuJoCo', ha='center', fontsize=7.5,
            fontstyle='italic', color='#444444')


# ---------------------------------------------------------------------------
# Fig.1 enriched modular pipeline (balanced widths, phase-buffer highlight)
# ---------------------------------------------------------------------------
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(17, 4.7))
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)

    mods = [
        (2.0, 0.9, 29.0, 8.2, '#ECECEC', '-',  'Stage 1'),
        (35.0, 0.9, 28.0, 8.2, '#DDDDDD', '--', 'Stage 2'),
        (67.0, 0.9, 31.0, 8.2, '#CECECE', '-.', 'Stages 3--5'),
    ]
    for (x, y, w, h, fc, ls, stage) in mods:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.20",
                                    linewidth=1.2, edgecolor='#333333',
                                    facecolor=fc, linestyle=ls))
        ax.text(x + 0.8, y + h - 0.75, stage, fontsize=8, color='#555555',
                fontweight='bold', ha='left', va='center')

    # inter-module arrows with data-name labels
    ar = FancyArrowPatch((31.4, 5.2), (34.6, 5.2), arrowstyle='-|>',
                         mutation_scale=15, color='black', linewidth=1.1)
    ax.add_patch(ar)
    ax.text(33.0, 5.9, '3D lower-body\nlandmarks', ha='center', fontsize=7.5,
            fontstyle='italic', color='#222222')
    ar = FancyArrowPatch((63.4, 5.2), (66.6, 5.2), arrowstyle='-|>',
                         mutation_scale=15, color='black', linewidth=1.1)
    ax.add_patch(ar)
    ax.text(65.0, 5.9, 'Reference\ntrajectory $\\theta(\\varphi)$', ha='center',
            fontsize=7.5, fontstyle='italic', color='#222222')

    # ---- Module 1 ----
    ax.text(16.5, 8.15, '1) Perception -- pose estimation', ha='center',
            fontsize=10, fontweight='bold')
    _walker_with_landmarks(ax, 16.5, 5.7, s=1.45)
    _substeps(ax, 3.0, 30.0, 2.1, 4.0,
              ['Side-view video\n(30 fps)', 'MediaPipe Pose\n(3D landmarks)',
               'Lower-body subset\n(10 joints)'], fs=7.5)
    _output(ax, 30.0, 1.2, 'Output: 3D lower-body landmarks')

    # ---- Module 2 ----
    ax.text(49.0, 8.15, '2) Kinematic retargeting + post-processing', ha='center',
            fontsize=10, fontweight='bold')
    _smoothing_curves(ax, 49.0, 5.8, w=6.0, h=2.4)
    _substeps(ax, 36.0, 62.0, 2.1, 4.0,
              ['Smoothing', 'Joint-limit\nprojection', 'Foot-ground\nalignment',
               'L/R\nsymmetrization', 'Period est.\n+ time-warp'], fs=7.2)
    _output(ax, 62.0, 1.2, 'Output: physically admissible $\\theta(\\varphi)$')

    # ---- Module 3 (widened) ----
    ax.text(82.5, 8.15, '3) RL control -- imitation learning', ha='center',
            fontsize=10, fontweight='bold')
    _phase_wheel(ax, 73.0, 5.9, r=0.72)
    _rapo_net(ax, 82.0, 5.9, w=8.5, h=2.8)
    _robot_render(ax, 93.5, 5.9, s=1.1)
    _substeps(ax, 68.0, 97.0, 2.1, 4.0,
              ['Phase buffer\n$q_{ref}(\\varphi)$', 'RA-PPO\ntraining',
               'MuJoCo\nevaluation'], fs=7.5, highlight_idx=0)
    _output(ax, 97.0, 1.2, 'Output: gait policy, RMSE, push recovery')

    fig.tight_layout(pad=0.2)
    save(fig, "fig1_pipeline.png")


def _output(ax, x_right, y, text):
    ax.text(x_right - 0.5, y, text, ha='right', va='center', fontsize=8,
            fontstyle='italic', color='#444444')


def _substeps(ax, x0, x1, y0, y1, labels, fs=7.5, highlight_idx=None):
    n = len(labels)
    gap = 1.4
    w = (x1 - x0 - gap * (n - 1)) / n
    x = x0
    for i, lab in enumerate(labels):
        is_key = (highlight_idx is not None and i == highlight_idx)
        ax.add_patch(FancyBboxPatch((x, y0), w, y1 - y0,
                                    boxstyle="round,pad=0.10",
                                    linewidth=1.4 if is_key else 0.8,
                                    edgecolor='black' if is_key else '#555555',
                                    facecolor='#BFBFBF' if is_key else 'white'))
        ax.text(x + w / 2, (y0 + y1) / 2, lab, ha='center', va='center', fontsize=fs,
                fontweight='bold' if is_key else 'normal')
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.12, (y0 + y1) / 2),
                                         (x + w + gap - 0.12, (y0 + y1) / 2),
                                         arrowstyle='-|>', mutation_scale=9,
                                         color='#333333', linewidth=0.8))
        x += w + gap


# ---------------------------------------------------------------------------
# Fig.4 metrics (tighter spacing, star inside bar, x-labels once)
# ---------------------------------------------------------------------------
def fig4_metrics():
    jpath = os.path.join(RES_DIR, "comparison_robio.json")
    data = json.load(open(jpath, encoding="utf-8"))

    def _get(o, k, field="mean"):
        v = data[o].get(k)
        if isinstance(v, dict):
            return v.get(field, 0)
        return v if field == "mean" and v is not None else 0

    xlabels = ['A: Sin. baseline', 'B: Exp. imitation', 'C: Ours', 'D: Raw video']
    facecolors = ['white', '#C8C8C8', '#404040', 'white']
    hatches = ['///', 'xx', '', '....']

    reward = [_get('baseline', 'total_reward'), _get('human', 'total_reward'),
              _get('ablation', 'total_reward'), _get('raw', 'total_reward')]
    reward_std = [_get('baseline', 'total_reward', 'std'), _get('human', 'total_reward', 'std'),
                  _get('ablation', 'total_reward', 'std'), _get('raw', 'total_reward', 'std')]
    rmse = [0.1728, _get('human', 'imitation_rmse'), _get('ablation', 'imitation_rmse'),
            _get('raw', 'imitation_rmse')]
    rmse_std = [0.0, _get('human', 'imitation_rmse', 'std'),
                _get('ablation', 'imitation_rmse', 'std'), 0.0]
    push = [100 * _get('baseline', 'push_recovery_rate'),
            100 * _get('human', 'push_recovery_rate'),
            100 * _get('ablation', 'push_recovery_rate'),
            100 * _get('raw', 'push_recovery_rate')]
    push_std = [100 * _get('baseline', 'push_recovery_rate', 'std'),
                100 * _get('human', 'push_recovery_rate', 'std'),
                100 * _get('ablation', 'push_recovery_rate', 'std'), 0.0]

    fig = plt.figure(figsize=(8.8, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                          wspace=0.28, hspace=0.20)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    _draw_bars(ax_a, xlabels, reward, reward_std, facecolors, hatches,
               tag='(a)', fmt='{:.0f}', ymax=2550, show_x=False)
    _draw_bars(ax_b, xlabels, rmse, rmse_std, facecolors, hatches,
               tag='(b)', fmt='{:.4f}', ymax=0.35, show_x=False)
    _draw_bars(ax_c, xlabels, push, push_std, facecolors, hatches,
               tag='(c)', fmt='{:.0f}', ymax=118, pct=True, show_x=True)

    ax_a.annotate('', xy=(1.9, reward[2] + 200), xytext=(0.1, reward[0] + 200),
                  arrowprops=dict(arrowstyle='<->', color='black', linewidth=0.9,
                                  shrinkA=0, shrinkB=0))
    ax_a.text(1.0, reward[2] + 210, '+23.9%', ha='center', fontsize=8,
              fontweight='bold')
    ax_a.annotate('', xy=(2.0, reward[2] + 90), xytext=(3.0, reward[3] + 90),
                  arrowprops=dict(arrowstyle='->', color='black', linewidth=0.9,
                                  shrinkA=0, shrinkB=0))
    ax_a.text(2.5, reward[3] + 110, '$-20\\%$', ha='center', fontsize=8,
              fontweight='bold')

    fig.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.10,
                        wspace=0.28, hspace=0.20)
    save(fig, "fig4_metrics.png")


def _draw_bars(ax, xlabels, means, stds, facecolors, hatches, tag,
               fmt, ymax, pct=False, show_x=True):
    x = np.arange(4)
    bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.55,
                  edgecolor='black', linewidth=0.8)
    for b, fc, h in zip(bars, facecolors, hatches):
        b.set_facecolor(fc)
        b.set_hatch(h)
    bars[2].set_linewidth(2.0)
    ax.text(x[2], means[2] - 0.05 * ymax, '$\\bigstar$', ha='center', va='top',
            fontsize=9, color='white')
    for b, val in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015 * ymax,
                fmt.format(val), ha='center', va='bottom', fontsize=7.5)
    ax.text(0.02, 0.97, tag, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top')
    ax.set_xticks(x)
    if show_x:
        ax.set_xticklabels(xlabels, fontsize=7)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)
    if pct:
        ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylim(0, ymax)
    ax.tick_params(axis='both', labelsize=7.5)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    fig1_pipeline()
    fig4_metrics()
    print("Fig.1 & Fig.4 regenerated (v5).")
