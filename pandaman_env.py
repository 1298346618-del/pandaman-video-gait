import math
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from scipy.spatial.transform import Rotation as R
from config import cfg


class PandamanEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-cfg.CLIP_OBS, high=cfg.CLIP_OBS,
            shape=(cfg.NUM_OBS,), dtype=np.float32)

        self.action_space = spaces.Box(
            low=-cfg.CLIP_ACTIONS, high=cfg.CLIP_ACTIONS,
            shape=(cfg.NUM_ACTIONS,), dtype=np.float32)

        self.model = mujoco.MjModel.from_xml_path(cfg.XML_PATH)
        self.model.opt.timestep = cfg.DT
        self.data = mujoco.MjData(self.model)

        if render_mode == "human":
            import mujoco_viewer
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        else:
            self.viewer = None

        self.dof_indices = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                            for n in cfg.JOINT_NAMES]
        self.qpos_start = [self.model.jnt_qposadr[i] for i in self.dof_indices]
        self.dof_vel_start = [self.model.jnt_dofadr[i] for i in self.dof_indices]

        self.dof_pos = np.zeros(cfg.NUM_ACTIONS, dtype=np.float64)
        self.dof_vel = np.zeros(cfg.NUM_ACTIONS, dtype=np.float64)
        self.actions = np.zeros(cfg.NUM_ACTIONS, dtype=np.float64)
        self.last_actions = np.zeros(cfg.NUM_ACTIONS, dtype=np.float64)
        self.last_dof_vel = np.zeros(cfg.NUM_ACTIONS, dtype=np.float64)
        self.torques = np.zeros(cfg.NUM_ACTIONS, dtype=np.float64)

        self.commands = np.zeros(3, dtype=np.float64)
        self.obs_history = deque(maxlen=cfg.FRAME_STACK)
        for _ in range(cfg.FRAME_STACK):
            self.obs_history.append(np.zeros(cfg.NUM_SINGLE_OBS, dtype=np.float32))

        self.episode_step = 0
        self.push_timer = 0.0
        self.last_feet_z = 0.05
        self.feet_height = np.zeros(2, dtype=np.float64)
        self.feet_air_time = np.zeros(2, dtype=np.float64)
        self.last_contacts = np.zeros(2, dtype=bool)
        self.contact_filt = np.zeros(2, dtype=bool)

    def _get_q_dq(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        for i, idx in enumerate(self.qpos_start):
            self.dof_pos[i] = qpos[idx]
        for i, idx in enumerate(self.dof_vel_start):
            self.dof_vel[i] = qvel[idx]

    def _get_imu(self):
        quat_raw = self.data.sensor("orientation").data.astype(np.float64)
        quat = np.array([quat_raw[1], quat_raw[2], quat_raw[3], quat_raw[0]])
        omega = self.data.sensor("angular-velocity").data.astype(np.float64)
        r = R.from_quat(quat)
        base_lin_vel = r.apply(self.data.qvel[:3], inverse=True).astype(np.float64)
        return quat, omega, base_lin_vel

    def _get_euler(self, quat):
        x, y, z, w = quat
        eu = np.zeros(3, dtype=np.float64)
        eu[0] = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        t2 = np.clip(2 * (w * y - z * x), -1.0, 1.0)
        eu[1] = np.arcsin(t2)
        eu[2] = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        eu[eu > math.pi] -= 2 * math.pi
        return eu

    def _get_phase(self):
        return self.episode_step * cfg.DT * cfg.DECIMATION / cfg.CYCLE_TIME

    def _get_gait_phase(self):
        phase = self._get_phase()
        sin_pos = math.sin(2 * math.pi * phase)
        stance = np.zeros(2, dtype=np.float64)
        stance[0] = 1.0 if sin_pos >= 0 else 0.0
        stance[1] = 1.0 if sin_pos < 0 else 0.0
        if abs(sin_pos) < 0.1:
            stance[:] = 1.0
        return stance, sin_pos

    def _sample_commands(self):
        self.commands[0] = np.random.uniform(*cfg.CmdRanges.LIN_VEL_X)
        self.commands[1] = np.random.uniform(*cfg.CmdRanges.LIN_VEL_Y)
        self.commands[2] = np.random.uniform(*cfg.CmdRanges.ANG_VEL_YAW)

    def _compute_ref_dof(self, sin_pos):
        ref = np.zeros(cfg.NUM_ACTIONS, dtype=np.float64)
        s1 = cfg.TARGET_JOINT_POS_SCALE
        s2 = 2 * s1
        sin_l = sin_pos if sin_pos < 0 else 0
        if abs(sin_pos) < 0.1:
            sin_l = 0
        sin_r = sin_pos if sin_pos > 0 else 0
        if abs(sin_pos) < 0.1:
            sin_r = 0

        ref[2] = -sin_l * s1 + cfg.DEFAULT_ANGLES[2]
        ref[3] = sin_l * s2 + cfg.DEFAULT_ANGLES[3]
        ref[4] = sin_l * s1 + cfg.DEFAULT_ANGLES[4]
        ref[7] = -sin_r * s1 + cfg.DEFAULT_ANGLES[7]
        ref[8] = sin_r * s2 + cfg.DEFAULT_ANGLES[8]
        ref[9] = sin_r * s1 + cfg.DEFAULT_ANGLES[9]
        return ref

    def _compute_rewards(self, sin_pos, ref_dof, quat, omega, base_lin_vel, eu_ang, contact):
        rewards = {}
        r_total = 0.0

        # 1. joint pos tracking
        diff = self.dof_pos - ref_dof
        r = math.exp(-2 * np.linalg.norm(diff)) - 0.2 * min(np.linalg.norm(diff), 0.5)
        rewards["joint_pos"] = r * cfg.RewardScales.JOINT_POS

        # 2. feet distance
        foot_pos = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_link")]
        foot_pos2 = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_link")]
        foot_dist = np.linalg.norm(foot_pos[:2] - foot_pos2[:2])
        d_min = min(foot_dist - cfg.MIN_FEET_DIST, 0)
        d_max = max(foot_dist - cfg.MAX_FEET_DIST, 0)
        r = (math.exp(-abs(d_min) * 100) + math.exp(-abs(d_max) * 100)) / 2
        rewards["feet_distance"] = r * cfg.RewardScales.FEET_DISTANCE

        # 3. knee distance
        knee_pos = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_knee_link")]
        knee_pos2 = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_knee_link")]
        knee_dist = np.linalg.norm(knee_pos[:2] - knee_pos2[:2])
        k_min = min(knee_dist - cfg.MIN_FEET_DIST, 0)
        k_max = max(knee_dist - cfg.MAX_FEET_DIST / 2, 0)
        r = (math.exp(-abs(k_min) * 100) + math.exp(-abs(k_max) * 100)) / 2
        rewards["knee_distance"] = r * cfg.RewardScales.KNEE_DISTANCE

        # 4. foot slip
        foot_vel = np.sqrt(self.dof_vel[4] ** 2 + self.dof_vel[9] ** 2)
        r = math.sqrt(foot_vel) if contact[0] or contact[1] else 0.0
        rewards["foot_slip"] = r * cfg.RewardScales.FOOT_SLIP

        # 5. feet air time
        self.feet_air_time += cfg.DT * cfg.DECIMATION
        sim_first_contact = 0
        if contact[0] and self.feet_air_time[0] > 0:
            sim_first_contact += min(self.feet_air_time[0], 0.5)
        if contact[1] and self.feet_air_time[1] > 0:
            sim_first_contact += min(self.feet_air_time[1], 0.5)
        self.feet_air_time *= (1 - contact)
        rewards["feet_air_time"] = sim_first_contact * cfg.RewardScales.FEET_AIR_TIME

        # 6. feet contact number (gait alignment)
        stance, _ = self._get_gait_phase()
        correct = sum(1 if (c > 0.5 and s > 0.5) or (c < 0.5 and s < 0.5) else 0
                       for c, s in zip(contact, stance))
        r = correct / 2.0
        rewards["feet_contact_number"] = r * cfg.RewardScales.FEET_CONTACT_NUMBER

        # 7. orientation
        quat_mismatch = math.exp(-(abs(eu_ang[0]) + abs(eu_ang[1])) * 10)
        orientation = math.exp(-np.linalg.norm([quat[1], quat[2]]) * 20)
        r = (quat_mismatch + orientation) / 2
        rewards["orientation"] = r * cfg.RewardScales.ORIENTATION

        # 8. feet contact forces
        cf = 0.0
        rewards["feet_contact_forces"] = cf * cfg.RewardScales.FEET_CONTACT_FORCES

        # 9. default joint pos (penalize yaw/roll deviation)
        joint_diff = self.dof_pos - cfg.DEFAULT_ANGLES
        yaw_roll = abs(joint_diff[0]) + abs(joint_diff[1]) + abs(joint_diff[5]) + abs(joint_diff[6])
        yaw_roll = max(yaw_roll - 0.1, 0)
        r = math.exp(-yaw_roll * 100) - 0.01 * np.linalg.norm(joint_diff)
        rewards["default_joint_pos"] = r * cfg.RewardScales.DEFAULT_JOINT_POS

        # 10. base height
        base_height = self.data.qpos[2]
        r = math.exp(-abs(base_height - cfg.BASE_HEIGHT_TARGET) * 100)
        rewards["base_height"] = r * cfg.RewardScales.BASE_HEIGHT

        # 11. base acceleration
        root_vel_old = self.data.qvel[:6].copy()
        root_acc = np.linalg.norm(root_vel_old)
        r = math.exp(-root_acc * 3)
        rewards["base_acc"] = r * cfg.RewardScales.BASE_ACC

        # 12. vel mismatch (z lin, xy ang)
        lin_m = math.exp(-base_lin_vel[2] ** 2 * 10)
        ang_m = math.exp(-(omega[0] ** 2 + omega[1] ** 2) * 5)
        r = (lin_m + ang_m) / 2
        rewards["vel_mismatch_exp"] = r * cfg.RewardScales.VEL_MISMATCH_EXP

        # 13. tracking lin vel
        err = (self.commands[0] - base_lin_vel[0]) ** 2 + (self.commands[1] - base_lin_vel[1]) ** 2
        r = math.exp(-err * cfg.TRACKING_SIGMA)
        rewards["tracking_lin_vel"] = r * cfg.RewardScales.TRACKING_LIN_VEL

        # 14. tracking ang vel
        err = (self.commands[2] - omega[2]) ** 2
        r = math.exp(-err * cfg.TRACKING_SIGMA)
        rewards["tracking_ang_vel"] = r * cfg.RewardScales.TRACKING_ANG_VEL

        # 15. track vel hard
        l_err = math.sqrt((self.commands[0] - base_lin_vel[0]) ** 2 + (self.commands[1] - base_lin_vel[1]) ** 2)
        l_exp = math.exp(-l_err * 10)
        a_err = abs(self.commands[2] - omega[2])
        a_exp = math.exp(-a_err * 10)
        linear_err = 0.2 * (l_err + a_err)
        r = (l_exp + a_exp) / 2 - linear_err
        rewards["track_vel_hard"] = r * cfg.RewardScales.TRACK_VEL_HARD

        # 16. low speed
        abs_speed = abs(base_lin_vel[0])
        abs_cmd = abs(self.commands[0])
        if abs_cmd > 0.1:
            if base_lin_vel[0] * self.commands[0] < 0:
                r = -2.0
            elif abs_speed < 0.5 * abs_cmd:
                r = -1.0
            elif abs_speed > 1.2 * abs_cmd:
                r = 0.0
            else:
                r = 1.2
        else:
            r = 0.0
        rewards["low_speed"] = r * cfg.RewardScales.LOW_SPEED

        # 17. torques
        r = np.sum(self.torques ** 2)
        rewards["torques"] = r * cfg.RewardScales.TORQUES

        # 18. dof vel
        r = np.sum(self.dof_vel ** 2)
        rewards["dof_vel"] = r * cfg.RewardScales.DOF_VEL

        # 19. dof acc
        r = np.sum(((self.last_dof_vel - self.dof_vel) / cfg.DT / cfg.DECIMATION) ** 2)
        rewards["dof_acc"] = r * cfg.RewardScales.DOF_ACC

        # 20. collision (base contact)
        base_z = self.data.qpos[2]
        r = 1.0 if base_z < 0.5 else 0.0
        rewards["collision"] = r * cfg.RewardScales.COLLISION

        # 21. action smoothness
        t1 = np.sum((self.last_actions - self.actions) ** 2)
        t2 = np.sum((self.actions + np.zeros_like(self.actions) - 2 * self.last_actions) ** 2)
        t3 = 0.05 * np.sum(np.abs(self.actions))
        r = t1 + t2 + t3
        rewards["action_smoothness"] = r * cfg.RewardScales.ACTION_SMOOTHNESS

        # 22. feet clearance
        contact_any = contact[0] or contact[1]
        feet_z = np.mean([foot_pos[2], foot_pos2[2]]) - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z
        stance, _ = self._get_gait_phase()
        swing_mask = 1 - stance
        rew_pos = sum(1 for h, s in zip(self.feet_height, swing_mask)
                       if s > 0.5 and abs(h - cfg.TARGET_FEET_HEIGHT) < 0.01)
        self.feet_height *= (1 - np.array([contact_any, contact_any]))
        rewards["feet_clearance"] = rew_pos * cfg.RewardScales.FEET_CLEARANCE

        total = sum(rewards.values())
        return total, rewards

    def _check_termination(self, eu_ang):
        base_z = self.data.qpos[2]
        if base_z < cfg.Termination.BASE_HEIGHT_MIN:
            return True
        if abs(eu_ang[0]) > cfg.Termination.ORIENTATION_LIMIT or \
           abs(eu_ang[1]) > cfg.Termination.ORIENTATION_LIMIT:
            return True
        if self.episode_step >= cfg.MAX_STEPS:
            return True
        return False

    def _build_obs(self, sin_pos, cos_pos, quat, omega, base_lin_vel, eu_ang):
        obs = np.zeros(cfg.NUM_SINGLE_OBS, dtype=np.float32)
        obs[0] = sin_pos
        obs[1] = cos_pos
        obs[2] = self.commands[0] * cfg.ObsScales.LIN_VEL
        obs[3] = self.commands[1] * cfg.ObsScales.LIN_VEL
        obs[4] = self.commands[2] * cfg.ObsScales.ANG_VEL
        obs[5:15] = (self.dof_pos - cfg.DEFAULT_ANGLES) * cfg.ObsScales.DOF_POS
        obs[15:25] = self.dof_vel * cfg.ObsScales.DOF_VEL
        obs[25:35] = self.actions
        obs[35:38] = omega * cfg.ObsScales.ANG_VEL
        obs[38:41] = eu_ang * cfg.ObsScales.QUAT
        obs = np.clip(obs, -cfg.CLIP_OBS, cfg.CLIP_OBS)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        qpos = self.data.qpos
        qpos[2] = cfg.BASE_HEIGHT_TARGET
        for i, idx in enumerate(self.qpos_start):
            qpos[idx] = cfg.DEFAULT_ANGLES[i] + np.random.uniform(-0.05, 0.05)
        mujoco.mj_forward(self.model, self.data)

        self._sample_commands()

        self.episode_step = 0
        self.push_timer = 0.0
        self.actions[:] = 0
        self.last_actions[:] = 0
        self.feet_height[:] = 0
        self.last_feet_z = 0.05
        self.feet_air_time[:] = 0
        self.last_contacts[:] = 0
        self.contact_filt[:] = 0

        self.obs_history.clear()
        for _ in range(cfg.FRAME_STACK):
            self.obs_history.append(np.zeros(cfg.NUM_SINGLE_OBS, dtype=np.float32))

        self._get_q_dq()
        quat, omega, base_lin_vel = self._get_imu()
        eu_ang = self._get_euler(quat)
        _, sin_pos = self._get_gait_phase()
        cos_pos = math.cos(2 * math.pi * self._get_phase())

        obs = self._build_obs(sin_pos, cos_pos, quat, omega, base_lin_vel, eu_ang)
        self.obs_history.append(obs)

        full_obs = np.zeros(cfg.NUM_OBS, dtype=np.float32)
        for i in range(cfg.FRAME_STACK):
            full_obs[i * cfg.NUM_SINGLE_OBS:(i + 1) * cfg.NUM_SINGLE_OBS] = self.obs_history[i]

        self.last_dof_vel[:] = self.dof_vel

        return full_obs, {}

    def step(self, action):
        action = np.clip(action, -cfg.CLIP_ACTIONS, cfg.CLIP_ACTIONS)

        self.last_actions[:] = self.actions
        self.actions[:] = action
        self.last_dof_vel[:] = self.dof_vel

        self.push_timer += cfg.DT * cfg.DECIMATION
        if self.push_timer >= cfg.DomainRand.PUSH_INTERVAL_S:
            self.push_timer = 0.0
            push_vel = np.random.uniform(-cfg.DomainRand.MAX_PUSH_VEL_XY,
                                         cfg.DomainRand.MAX_PUSH_VEL_XY, 2)
            ang_push = np.random.uniform(-cfg.DomainRand.MAX_PUSH_ANG_VEL,
                                         cfg.DomainRand.MAX_PUSH_ANG_VEL, 3)
            self.data.qvel[:2] += push_vel
            self.data.qvel[3:6] += ang_push

        # PD control loop
        for _ in range(cfg.DECIMATION):
            self._get_q_dq()
            target_q = action * cfg.ACTION_SCALE + cfg.DEFAULT_ANGLES
            target_dq = np.zeros(cfg.NUM_ACTIONS, dtype=np.float64)
            self.torques = (target_q - self.dof_pos) * cfg.KPS + (target_dq - self.dof_vel) * cfg.KDS
            self.torques = np.clip(self.torques, -cfg.TAU_LIMIT, cfg.TAU_LIMIT)
            for i, idx in enumerate(self.qpos_start):
                self.data.ctrl[i] = self.torques[i]
            mujoco.mj_step(self.model, self.data)

        if self.viewer:
            self.viewer.render()

        self.episode_step += 1

        self._get_q_dq()
        quat, omega, base_lin_vel = self._get_imu()
        eu_ang = self._get_euler(quat)
        _, sin_pos = self._get_gait_phase()
        cos_pos = math.cos(2 * math.pi * self._get_phase())

        stance, _ = self._get_gait_phase()
        contact = np.array([stance[0], stance[1]], dtype=np.float64)
        contact[0] = 1.0 if self.data.qpos[2] < 0.65 else 0.0
        contact[1] = 1.0 if self.data.qpos[2] < 0.65 else 0.0

        terminated = self._check_termination(eu_ang)
        truncated = self.episode_step >= cfg.MAX_STEPS

        ref_dof = self._compute_ref_dof(sin_pos)
        total_reward, reward_dict = self._compute_rewards(
            sin_pos, ref_dof, quat, omega, base_lin_vel, eu_ang, contact)

        obs = self._build_obs(sin_pos, cos_pos, quat, omega, base_lin_vel, eu_ang)
        self.obs_history.append(obs)

        full_obs = np.zeros(cfg.NUM_OBS, dtype=np.float32)
        for i in range(cfg.FRAME_STACK):
            full_obs[i * cfg.NUM_SINGLE_OBS:(i + 1) * cfg.NUM_SINGLE_OBS] = self.obs_history[i]

        info = {"rewards": reward_dict}
        return full_obs, total_reward, terminated, truncated, info

    def close(self):
        if self.viewer:
            self.viewer.close()
