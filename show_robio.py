import os
import sys
import argparse
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pandaman_env_imitation import PandamanImitationEnv
from config import cfg

ROOT = os.path.dirname(os.path.abspath(__file__))
REF_TRAJ = os.path.join(ROOT, "data", "reference", "reference_traj.npy")
REF_STANCE = os.path.join(ROOT, "data", "reference", "ref_stance.npy")


def main():
    p = argparse.ArgumentParser(description="Visualize a trained model walking")
    p.add_argument("--model", type=str, required=True,
                   help="path to model zip, e.g. results/robio_ablation_s42/final_model.zip")
    p.add_argument("--ref_mode", choices=["human", "sinusoid"], default="human")
    p.add_argument("--vel_w", type=float, default=0.0)
    p.add_argument("--phase_w", type=float, default=0.0)
    p.add_argument("--speed", type=float, default=0.4,
                   help="fixed forward speed command (m/s)")
    p.add_argument("--push", action="store_true",
                   help="enable domain-randomization pushes")
    args = p.parse_args()

    env = PandamanImitationEnv(
        render_mode="human",
        ref_mode=args.ref_mode,
        ref_traj_path=REF_TRAJ if args.ref_mode == "human" else None,
        stance_path=REF_STANCE if args.ref_mode == "human" else None,
        vel_tracking_weight=args.vel_w,
        phase_align_weight=args.phase_w,
    )

    # fixed command
    env.commands[:] = [args.speed, 0.0, 0.0]

    # if push disabled, neutralize the random push in step()
    if not args.push:
        cfg.DomainRand.PUSH_INTERVAL_S = 1e9

    model = PPO.load(args.model, device="cuda")

    obs, _ = env.reset()
    env.commands[:] = [args.speed, 0.0, 0.0]
    print(f"Model: {args.model}")
    print(f"Speed cmd: {args.speed} m/s | Push: {'ON' if args.push else 'OFF'}")
    print("Walking... close the viewer window to exit.")

    step = 0
    while env.viewer is not None and env.viewer.is_alive:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        if terminated or truncated:
            print(f"Episode ended at step {step} (terminated={terminated}, "
                  f"truncated={truncated})")
            obs, _ = env.reset()
            env.commands[:] = [args.speed, 0.0, 0.0]
            step = 0

    env.close()


if __name__ == "__main__":
    main()
