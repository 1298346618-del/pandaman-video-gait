import os
import math
import numpy as np

from pandaman_env import PandamanEnv
from config import cfg


class PandamanImitationEnv(PandamanEnv):
    """Pandaman env with pluggable reference trajectory source.

    ref_mode:
      "sinusoid" - original sinusoidal reference (baseline, ICIA behavior)
      "human"    - reference trajectory extracted from human video
    imitation_kwargs:
      vel_tracking_weight : weight of joint-velocity imitation reward
      phase_align_weight  : weight of phase/stance alignment reward
      (0.0 disables the component -> ablation)
    """

    def __init__(self, render_mode=None, ref_mode="sinusoid",
                 ref_traj_path=None, stance_path=None,
                 vel_tracking_weight=0.0, phase_align_weight=0.0,
                 vel_scale=0.3, ref_start_joint=0):
        super().__init__(render_mode=render_mode)
        self.ref_mode = ref_mode
        self.ref_traj_path = ref_traj_path
        self.stance_path = stance_path
        self.vel_tracking_weight = vel_tracking_weight
        self.phase_align_weight = phase_align_weight
        self.vel_scale = vel_scale
        self.ref_start_joint = ref_start_joint

        self.ref_traj = None
        self.ref_vel = None
        self.ref_stance = None
        self.ref_n = 0
        if ref_mode == "human":
            assert ref_traj_path and os.path.exists(ref_traj_path), \
                f"ref_traj_path required for human mode: {ref_traj_path}"
            self.ref_traj = np.load(ref_traj_path).astype(np.float64)
            self.ref_n = self.ref_traj.shape[0]
            self.ref_vel = np.gradient(self.ref_traj, axis=0) \
                * (self.ref_n / cfg.CYCLE_TIME)
            if stance_path and os.path.exists(stance_path):
                self.ref_stance = np.load(stance_path).astype(np.float64)
            # precompute target dof velocity at each phase sample (derivative)
            self.ref_vel = np.gradient(self.ref_traj, axis=0) \
                * (self.ref_n / cfg.CYCLE_TIME)

    # ------------------------------------------------------------------
    def _phase_index(self, sin_pos):
        """Map current phase to continuous index in reference buffer."""
        phase = self._get_phase() % 1.0
        idx = phase * self.ref_n
        i0 = int(math.floor(idx)) % self.ref_n
        i1 = (i0 + 1) % self.ref_n
        frac = idx - math.floor(idx)
        return i0, i1, frac

    def _compute_ref_dof(self, sin_pos):
        if self.ref_mode == "sinusoid":
            return super()._compute_ref_dof(sin_pos)
        i0, i1, frac = self._phase_index(sin_pos)
        ref = (1 - frac) * self.ref_traj[i0] + frac * self.ref_traj[i1]
        return ref

    def _compute_ref_vel(self, sin_pos):
        if self.ref_mode == "sinusoid":
            # analytic derivative of sinusoidal reference
            phase = self._get_phase()
            s1 = cfg.TARGET_JOINT_POS_SCALE
            s2 = 2 * s1
            w = 2 * math.pi / cfg.CYCLE_TIME
            sin_pos = math.sin(2 * math.pi * phase)
            cos_pos = math.cos(2 * math.pi * phase)
            ref = np.zeros(cfg.NUM_ACTIONS, dtype=np.float64)
            # left leg (indices 2,3,4): -sin*s1, +sin*s2, +sin*s1
            ref[2] = -cos_pos * s1 * w
            ref[3] = cos_pos * s2 * w
            ref[4] = cos_pos * s1 * w
            # right leg (indices 7,8,9): +sin, -sin, -sin (anti-phase)
            ref[7] = -(-cos_pos) * s1 * w
            ref[8] = -cos_pos * s2 * w
            ref[9] = -cos_pos * s1 * w
            return ref
        i0, i1, frac = self._phase_index(sin_pos)
        return (1 - frac) * self.ref_vel[i0] + frac * self.ref_vel[i1]

    # ------------------------------------------------------------------
    def _compute_rewards(self, sin_pos, ref_dof, quat, omega,
                         base_lin_vel, eu_ang, contact):
        total, rewards = super()._compute_rewards(
            sin_pos, ref_dof, quat, omega, base_lin_vel, eu_ang, contact)

        # imitation: joint velocity tracking (DeepMimic-style,
        # per-joint normalized to avoid vanishing gradients)
        if self.vel_tracking_weight > 0:
            ref_vel = self._compute_ref_vel(sin_pos)
            diff = (self.dof_vel - ref_vel) * self.vel_scale
            r = math.exp(-0.5 * np.mean(diff * diff))
            rewards["imitation_vel"] = r * self.vel_tracking_weight
            total += rewards["imitation_vel"]

        # imitation: phase/stance alignment with reference gait
        if self.phase_align_weight > 0 and self.ref_stance is not None:
            i0, i1, frac = self._phase_index(sin_pos)
            ref_stance = (1 - frac) * self.ref_stance[:, i0] \
                + frac * self.ref_stance[:, i1]
            correct = sum(1 for c, s in zip(contact, ref_stance)
                          if (c > 0.5 and s > 0.5) or (c < 0.5 and s < 0.5))
            r = correct / 2.0
            rewards["imitation_phase"] = r * self.phase_align_weight
            total += rewards["imitation_phase"]
        return total, rewards

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.ref_mode == "human":
            # start from reference pose at phase 0
            qpos = self.data.qpos
            qpos[2] = cfg.BASE_HEIGHT_TARGET
            start = self.ref_start_joint
            for i, idx in enumerate(self.qpos_start):
                qpos[idx] = self.ref_traj[start % self.ref_n, i] \
                    + np.random.uniform(-0.02, 0.02)
            mujoco_import = __import__("mujoco")
            mujoco_import.mj_forward(self.model, self.data)
            # rebuild obs from new state
            self._get_q_dq()
            quat, omega, base_lin_vel = self._get_imu()
            eu_ang = self._get_euler(quat)
            _, sin_pos = self._get_gait_phase()
            cos_pos = math.cos(2 * math.pi * self._get_phase())
            self.obs_history.clear()
            for _ in range(cfg.FRAME_STACK):
                self.obs_history.append(np.zeros(cfg.NUM_SINGLE_OBS,
                                                 dtype=np.float32))
            new_obs = self._build_obs(sin_pos, cos_pos, quat, omega,
                                      base_lin_vel, eu_ang)
            self.obs_history.append(new_obs)
            full_obs = np.zeros(cfg.NUM_OBS, dtype=np.float32)
            for i in range(cfg.FRAME_STACK):
                full_obs[i * cfg.NUM_SINGLE_OBS:(i + 1) * cfg.NUM_SINGLE_OBS] \
                    = self.obs_history[i]
            return full_obs, {}
        return obs, info
