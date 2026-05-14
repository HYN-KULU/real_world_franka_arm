"""
Sample placements conditioned on a previously-collected grasp.

What it does:
1. Loads grasps_<OBJECT_ID>.npz from data/shelf_packing_scenes/env_<ENV_ID>/
   and picks the grasp at index GRASP_ID as the conditioning pose.
2. Pulls a fresh point cloud + segmentation from the perception pipeline.
3. Builds a target mask from seg_labels using the target_label stored in
   the npz (overridable with --target_label).
4. Runs the place policy num_placements times (default 32), conditioned on
   the chosen grasp pose, batched in one forward pass.
5. Visualizes the conditioning grasp (green) + sampled placements (varied
   colors) on the live point cloud with the target cluster highlighted.

Run prerequisites:
1. Perception pipeline running with segmentation enabled:
     python -m frankapanda.perception.perception_pipeline --continuous --segment
2. visplanWM .pth on PYTHONPATH (setup_frankapanda_3dfa.sh handles this).
3. A saved grasps_<OBJECT_ID>.npz produced by sample_grasps.py.

Usage:
    Edit ENV_ID / OBJECT_ID / GRASP_ID / NUM_PLACEMENTS at the top of this
    file before running.

    conda activate frankapanda_3dfa
    python sample_placements.py
    python sample_placements.py --target_label "green and blue lego"
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from robo_utils.visualization.plotting import plot_pcd
from robo_utils.visualization.point_cloud_structures import make_gripper_visualization
from robo_utils.conversion_utils import pose_to_transformation

from frankapanda.perception import PerceptionPipeline
from frankapanda.motionplanner import MotionPlanner

from policy_value_inference import PolicyValueInference
from frankapanda import FrankaPandaController


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

VISPLAN_WM = "/home/ksaha/Research/ModelBasedPlanning/visplanWM"

DEFAULT_PLACE_CKPT = os.path.join(
    VISPLAN_WM,
    "models/flowmatch_actor/train_logs/Value_Function_Planning/sim2real_place_v3/best.pth",
)


# --------------------------------------------------------------------------- #
#                                                                             #
#   >>>  USER EDIT POINT  <<<                                                 #
#                                                                             #
#   Integer ids selecting which saved grasp to condition on. Loads file       #
#   data/shelf_packing_scenes/env_<ENV_ID>/grasps_<OBJECT_ID>.npz             #
#   and picks grasps[GRASP_ID] as the placement-policy condition.             #
#   NUM_PLACEMENTS = K placement candidates sampled per call.                 #
# --------------------------------------------------------------------------- #
ENV_ID = 0
OBJECT_ID = 1
GRASP_ID = 0
NUM_PLACEMENTS = 256


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #

def filter_downward_placements(placements):
    """
    Drop placements whose gripper local z-axis points more up than down in
    world frame, i.e. R[:, 2] . world_z = R[2, 2] > 0. Pose format is
    (7,) [x, y, z, qw, qx, qy, qz]. Returns the surviving subset (same
    container element types as input).
    """
    kept = []
    for p in placements:
        if isinstance(p, torch.Tensor):
            arr = p.detach().cpu().numpy()
        else:
            arr = np.asarray(p)
        qw, qx, qy, qz = arr[3], arr[4], arr[5], arr[6]
        # R[2, 2] for wxyz quat: 1 - 2*(qx^2 + qy^2)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
        if r22 < 0.0:
            kept.append(p)
    return kept


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #

def visualize_poses_in_pointcloud(pcd, poses, rgb=None, colors=None, length=0.05):
    if colors is None:
        colors = [(1, 0, 0)] * len(poses)
    combined_pcd = pcd.copy()
    combined_rgb = rgb.copy() if rgb is not None else None
    for i, pose in enumerate(poses):
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()
        transform = pose_to_transformation(pose, format="wxyz")
        pts, cols = make_gripper_visualization(
            rotation=transform[:3, :3],
            translation=transform[:3, 3],
            length=length,
            density=50,
            color=colors[i % len(colors)],
        )
        combined_pcd = np.vstack([combined_pcd, pts])
        if combined_rgb is not None:
            combined_rgb = np.vstack([combined_rgb, cols])
    if combined_rgb is not None:
        plot_pcd(combined_pcd, combined_rgb, base_frame=True)
    else:
        plot_pcd(combined_pcd, base_frame=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--place_ckpt", default=DEFAULT_PLACE_CKPT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--publish_port", type=int, default=1235)
    parser.add_argument("--target_label", type=str, default=None,
                        help="Segmentation label to use as target mask. Defaults to the "
                             "target_label stored in the npz.")
    parser.add_argument("--save_dir", type=str, default="data/shelf_packing_scenes",
                        help="Root dir under which env_<ENV_ID>/ lives.")
    parser.add_argument("--no_viz", action="store_true", help="Skip Open3D visualization.")
    parser.add_argument("--chunk_size", type=int, default=4,
                        help="Mini-batch size for placement forward passes (lower = less GPU RAM).")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Initialize robot controller
    controller = FrankaPandaController()

    # 1. Load saved grasps npz; pick conditioning grasp.
    npz_path = Path(args.save_dir) / f"env_{ENV_ID}" / f"grasps_{OBJECT_ID}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing saved grasps: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    saved_grasps = data["grasps"]                                          # (G, 7)
    saved_label = str(data["target_label"])
    if not (0 <= GRASP_ID < len(saved_grasps)):
        raise IndexError(
            f"GRASP_ID={GRASP_ID} out of range; npz has {len(saved_grasps)} grasps."
        )
    grasp_pose_np = saved_grasps[GRASP_ID].astype(np.float32)              # (7,) wxyz
    print(f"Loaded grasps from {npz_path}")
    print(f"  saved target_label: '{saved_label}'  num_grasps: {len(saved_grasps)}")
    print(f"  using GRASP_ID={GRASP_ID} pose: {grasp_pose_np.tolist()}")

    target_label = args.target_label if args.target_label is not None else saved_label

    # 2. Build inference (place model only).
    inference = PolicyValueInference(
        place_ckpt=args.place_ckpt,
        device=device,
    )

    # 3. Live perception capture.
    print("Connecting to perception pipeline")
    perception = PerceptionPipeline(publish_port=args.publish_port, timeout_ms=10000)
    print("Capturing point cloud")
    try:
        pdata = perception.get_point_cloud_dict()
    except TimeoutError:
        print("Timeout waiting for PCD. Is `python -m frankapanda.perception."
              "perception_pipeline --continuous --segment` running?")
        return
    pcd_np = pdata["pcd"]
    seg_labels = pdata.get("seg_labels")
    seg_label_names = pdata.get("seg_label_names")

    if seg_labels is None or seg_label_names is None:
        raise RuntimeError(
            "Perception payload missing seg_labels / seg_label_names. Re-run "
            "perception_pipeline with --segment."
        )

    seg_labels = np.asarray(seg_labels)
    per_label_counts = {n: int((seg_labels == k).sum()) for k, n in enumerate(seg_label_names)}
    print(f"  pcd: {pcd_np.shape}  seg per-label counts: {per_label_counts}")
    if target_label not in seg_label_names:
        raise RuntimeError(
            f"target_label='{target_label}' not in current seg labels {seg_label_names}"
        )
    chosen_idx = seg_label_names.index(target_label)
    mask_np = (seg_labels == chosen_idx).astype(np.float32)
    if mask_np.sum() == 0:
        raise RuntimeError(f"Target label '{target_label}' has 0 points in current frame.")
    print(f"  target mask: '{target_label}' ({int(mask_np.sum())}/{len(pcd_np)} points)")

    # Shared pcd coloring: target cluster red on gray pcd.
    viz_rgb = np.where(
        mask_np[:, None].astype(bool),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.5, 0.5], dtype=np.float32),
    )

    # 4. Pre-inference viz: pcd + mask + conditioning grasp pose (green).
    if not args.no_viz:
        print("Visualizing pcd + mask + conditioning grasp (green). Close window to continue.")
        # plot_pcd(pcd_np, viz_rgb, base_frame=True)
        visualize_poses_in_pointcloud(
            pcd_np,
            [torch.from_numpy(grasp_pose_np)],
            rgb=viz_rgb,
            colors=[(0.0, 1.0, 0.0)],
        )

    motion_planner = MotionPlanner(pcd_np)

    current_joints = controller.get_robot_joints()
    current_joints = torch.tensor(current_joints, dtype=torch.float32, device="cuda:0")
    motion_planner.visualize_world_and_robot(current_joints)

    # 5. Sample placements conditioned on the chosen grasp.
    placements = inference.infer_placements(
        pcd_np=pcd_np,
        mask_np=mask_np,
        grasp_pose=grasp_pose_np,
        num_placements=NUM_PLACEMENTS,
        chunk_size=args.chunk_size,
    )
    print(f"  sampled {len(placements)} placement candidates")
    for i, p in enumerate(placements):
        print(f"    [{i}] {p.tolist()}")

    for i in range(len(placements)):
        # PLACEMENT CORRECTION
        placements[i][0] = placements[i][0] + 0.2
        placements[i][1] = placements[i][1] + 0.18
        placements[i][2] = placements[i][2] + 0.15

    # 6a. Orientation filter: drop placements whose gripper z-axis points up.
    pre_filter_n = len(placements)
    placements = filter_downward_placements(placements)
    print(f"  orientation filter: {len(placements)}/{pre_filter_n} point downward")
    if len(placements) == 0:
        print("  No downward placements; skipping IK + viz.")
        perception.close()
        return

    # 6b. IK feasibility filter via curobo. MotionPlanner is hardcoded to cuda:0.
    print("Running IK feasibility check on placement candidates")
    placements_t = torch.stack(
        [p.float() if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.float32)
         for p in placements]
    ).to("cuda:0")                                                         # (K, 7) wxyz
    _, ik_success = motion_planner.inverse_kinematics(placements_t)
    ik_success = ik_success.cpu().bool()
    feasible_idx = torch.where(ik_success)[0].tolist()
    pre_ik_n = len(placements)
    placements = [placements[i] for i in feasible_idx]
    print(f"  IK feasible: {len(feasible_idx)}/{pre_ik_n}")
    if len(placements) == 0:
        print("  No feasible placements; skipping post-inference viz.")
        perception.close()
        return

    # 7. Post-inference viz: conditioning grasp (green) + IK-feasible placements (varied).
    if not args.no_viz:
        PLACE_COLORS = [
            (0.0, 0.0, 1.0),  # blue
            (1.0, 1.0, 0.0),  # yellow
            (1.0, 0.0, 1.0),  # magenta
            (0.0, 1.0, 1.0),  # cyan
            (1.0, 0.5, 0.0),  # orange
            (0.5, 0.0, 1.0),  # purple
            (1.0, 0.75, 0.8), # pink
        ]
        grasp_t = torch.from_numpy(grasp_pose_np)
        poses_viz = [grasp_t] + list(placements)
        colors_viz = [(0.0, 1.0, 0.0)] + [
            PLACE_COLORS[i % len(PLACE_COLORS)] for i in range(len(placements))
        ]
        print("Visualizing pcd + mask + grasp (green) + IK-feasible placements. "
              "Close window to continue.")
        visualize_poses_in_pointcloud(
            pcd_np, poses_viz, rgb=viz_rgb, colors=colors_viz,
        )

    # 8. Optional save of IK-feasible placements.
    save_path = Path(args.save_dir) / f"env_{ENV_ID}" / "placement_poses.npz"
    while True:
        ans = input(f"Save {len(placements)} IK-feasible placements to {save_path}? [y/N]: ").strip().lower()
        if ans in ("", "n", "no"):
            print("  Not saving.")
            break
        if ans in ("y", "yes"):
            placements_arr = np.stack(
                [p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
                 for p in placements]
            ).astype(np.float32)                                           # (M, 7)
            tmp_path = save_path.with_suffix(".tmp.npz")
            np.savez(
                tmp_path,
                placements=placements_arr,
                grasp_pose=grasp_pose_np,
                target_label=np.array(target_label),
                env_id=np.array(ENV_ID),
                object_id=np.array(OBJECT_ID),
                grasp_id=np.array(GRASP_ID),
                pcd=pcd_np.astype(np.float32),
            )
            os.replace(tmp_path, save_path)
            print(f"  saved {len(placements)} placements -> {save_path}")
            break
        print("  Please answer y or n.")

    perception.close()


if __name__ == "__main__":
    main()
