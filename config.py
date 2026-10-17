import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

class Config:
    # Self-contained robot description shipped in this repository.
    # Override with the PANDAMAN_XML environment variable if you keep the
    # MJCF elsewhere.
    XML_PATH = os.environ.get(
        "PANDAMAN_XML",
        os.path.join(ROOT, "assets", "robots", "pandaman", "mjcf", "pandaman.xml"),
    )
    LOG_DIR = os.path.join(ROOT, "results")

    NUM_ACTIONS = 10
    NUM_SINGLE_OBS = 41
    FRAME_STACK = 15
    NUM_OBS = FRAME_STACK * NUM_SINGLE_OBS  # 615

    DT = 0.001
    DECIMATION = 10  # policy runs at 100 Hz
    EPISODE_LENGTH_S = 24.0
    MAX_STEPS = int(EPISODE_LENGTH_S / (DT * DECIMATION))  # 2400

    CYCLE_TIME = 0.64

    TAU_LIMIT = 200.0

    ACTION_SCALE = 0.25
    CLIP_ACTIONS = 18.0
    CLIP_OBS = 18.0

    KPS = np.array([20, 20, 20, 20, 7, 20, 20, 20, 20, 7], dtype=np.float64)
    KDS = np.array([0.15, 0.15, 0.15, 0.15, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1], dtype=np.float64)

    DEFAULT_ANGLES = np.array([
        0.0, 0.0, 0.17, -0.34, -0.09,
        0.0, 0.0, -0.17, 0.34, 0.09,
    ], dtype=np.float64)

    JOINT_NAMES = [
        "left_leg_roll_joint", "left_leg_yaw_joint", "left_leg_pitch_joint",
        "left_knee_joint", "left_ankle_joint",
        "right_leg_roll_joint", "right_leg_yaw_joint", "right_leg_pitch_joint",
        "right_knee_joint", "right_ankle_joint",
    ]

    class ObsScales:
        LIN_VEL = 2.0
        ANG_VEL = 1.0
        DOF_POS = 1.0
        DOF_VEL = 0.05
        QUAT = 1.0

    class CmdRanges:
        LIN_VEL_X = [-0.3, 0.6]
        LIN_VEL_Y = [-0.3, 0.3]
        ANG_VEL_YAW = [-0.3, 0.3]

    class RewardScales:
        JOINT_POS = 2.0
        FEET_CLEARANCE = 1.0
        FEET_CONTACT_NUMBER = 1.2
        FEET_AIR_TIME = 1.0
        FOOT_SLIP = -0.05
        FEET_DISTANCE = 0.2
        KNEE_DISTANCE = 0.2
        FEET_CONTACT_FORCES = -0.01
        TRACKING_LIN_VEL = 1.2
        TRACKING_ANG_VEL = 1.1
        VEL_MISMATCH_EXP = 0.5
        LOW_SPEED = 0.2
        TRACK_VEL_HARD = 0.5
        DEFAULT_JOINT_POS = 0.5
        ORIENTATION = 1.0
        BASE_HEIGHT = 0.2
        BASE_ACC = 0.2
        ACTION_SMOOTHNESS = -0.002
        TORQUES = -1e-5
        DOF_VEL = -5e-4
        DOF_ACC = -1e-7
        COLLISION = -1.0

    BASE_HEIGHT_TARGET = 0.6
    MIN_FEET_DIST = 0.2
    MAX_FEET_DIST = 0.5
    TARGET_JOINT_POS_SCALE = 0.26
    TARGET_FEET_HEIGHT = 0.12
    TRACKING_SIGMA = 5
    MAX_CONTACT_FORCE = 700

    class DomainRand:
        PUSH_INTERVAL_S = 4
        MAX_PUSH_VEL_XY = 0.2
        MAX_PUSH_ANG_VEL = 0.4
        RANDOMIZE_FRICTION = True
        FRICTION_RANGE = [0.1, 2.0]
        RANDOMIZE_MASS = True
        MASS_RANGE = [-0.1, 0.1]

    class Termination:
        BASE_HEIGHT_MIN = 0.3
        ORIENTATION_LIMIT = 1.0

    FEET_BODY_NAMES = ["left_knee_link", "right_knee_link"]

cfg = Config()
