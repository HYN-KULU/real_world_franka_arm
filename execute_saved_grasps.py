"""
Replay the saved joint trajectories from every
  data/shelf_packing_scenes/env_<ENV_ID>/grasps_*.npz
for every grasp that was marked successful.

Each saved trajectory is the concatenation of pre_grasp + grasp + lift joint
sequences. We reset to home before each replay, run the full concatenated
trajectory, then return home.

Edit ENV_ID below before running. All object files inside that env dir
are picked up automatically (sorted by integer object id).
"""

import argparse
import re
from pathlib import Path

import numpy as np

from frankapanda import FrankaPandaController


# --------------------------------------------------------------------------- #
#                                                                             #
#   >>>  USER EDIT POINT  <<<                                                 #
#                                                                             #
#   Match the env id used by infer_policy_value_demo.py when saving           #
#   data/shelf_packing_scenes/env_<ENV_ID>/grasps_*.npz                       #
# --------------------------------------------------------------------------- #
ENV_ID = 0


def _as_float_traj(t):
    """Object-dtype round-trip from np.savez can yield non-numeric arrays.
    Coerce to float32 (T, J)."""
    if t is None:
        return None
    a = np.asarray(t, dtype=np.float32)
    return a if a.ndim == 2 else None


def _object_id_from_path(p):
    m = re.match(r"grasps_(\d+)\.npz$", p.name)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="data/shelf_packing_scenes",
                        help="Root dir holding env_<id>/grasps_<id>.npz files.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be executed; don't move the robot.")
    args = parser.parse_args()

    env_dir = Path(args.save_dir) / f"env_{ENV_ID}"
    if not env_dir.is_dir():
        raise FileNotFoundError(f"Env dir not found: {env_dir}")

    # Pick up grasps_<int>.npz files; sort by integer object id.
    object_files = sorted(
        (p for p in env_dir.glob("grasps_*.npz") if _object_id_from_path(p) is not None),
        key=lambda p: _object_id_from_path(p),
    )
    if not object_files:
        print(f"No grasps_<id>.npz files in {env_dir}")
        return
    print(f"Found {len(object_files)} object file(s) in {env_dir}:")
    for p in object_files:
        print(f"  - {p.name}")

    controller = None
    total_done = 0

    for p in object_files:
        object_id = _object_id_from_path(p)
        data = np.load(p, allow_pickle=True)
        grasps = data["grasps"]
        trajectories = data["trajectories"]
        successes = data["successes"].astype(bool)
        target_label = str(data["target_label"]) if "target_label" in data.files else "?"
        success_idx = np.where(successes)[0]

        print(f"\n=== object_id={object_id}  target='{target_label}'  "
              f"successful={len(success_idx)}/{len(grasps)} ===")

        if len(success_idx) == 0:
            print("  no successful grasps; skipping.")
            continue

        if args.dry_run:
            for n, i in enumerate(success_idx):
                traj = _as_float_traj(trajectories[i])
                shape = None if traj is None else traj.shape
                print(f"  [dry] would execute grasp {i}: traj shape {shape}")
            continue

        if controller is None:
            controller = FrankaPandaController()

        for n, i in enumerate(success_idx):
            traj = _as_float_traj(trajectories[i])
            if traj is None or len(traj) == 0:
                print(f"  [{n+1}/{len(success_idx)}] grasp {i}: empty trajectory, skipping.")
                continue
            print(f"  [{n+1}/{len(success_idx)}] Replaying grasp {i} "
                  f"(T={len(traj)}, J={traj.shape[1]})")
            controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
            controller.move_along_trajectory(traj, controller.open_gripper_action)
            controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
            total_done += 1

    print(f"\nReplayed {total_done} successful trajectories across {len(object_files)} object(s).")


if __name__ == "__main__":
    main()
