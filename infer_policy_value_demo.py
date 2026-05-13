"""
Minimal Demo: Real-world Policy + (optional) Value Inference, with optional
robot execution of the predicted grasp.

What it does:
1. Pulls a clustered, FPS-downsampled point cloud from the perception
   pipeline (run with `--cluster`).
2. Picks one cluster uniformly at random as the target object; builds
   a (N,) target_mask aligned to the FPS points.
3. Calls PolicyValueInference to predict grasp + place poses (and a
   value score when --use_value is set).
4. Visualizes the target cluster + predicted gripper poses with Open3D.
5. With --exec, runs the grasping portion of primitives_demo against
   the predicted grasp pose (pre-grasp -> grasp -> close -> lift ->
   home), then opens the gripper.

Run prerequisites:
1. Perception pipeline running with clustering enabled:
     python -m frankapanda.perception.perception_pipeline --continuous --cluster
2. visplanWM .pth on PYTHONPATH (setup_frankapanda_3dfa.sh handles this).
3. For --exec: robot powered on and reachable by FrankaPandaController.

Usage:
    conda activate frankapanda_3dfa
    # policy only
    python infer_policy_value_demo.py
    # policy + value
    python infer_policy_value_demo.py --use_value
    # policy + execute grasp on the real robot
    python infer_policy_value_demo.py --exec
"""

import argparse
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from robo_utils.visualization.plotting import plot_pcd
from robo_utils.visualization.point_cloud_structures import make_gripper_visualization
from robo_utils.conversion_utils import (
    pose_to_transformation,
    move_pose_along_local_z,
    rotate_pose_around_local_z,
)

from frankapanda import FrankaPandaController
from frankapanda.perception import PerceptionPipeline
from frankapanda.motionplanner import MotionPlanner

from policy_value_inference import PolicyValueInference


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

VISPLAN_WM = "/home/ksaha/Research/ModelBasedPlanning/visplanWM"

DEFAULT_GRASP_CKPT = os.path.join(
    VISPLAN_WM,
    "models/flowmatch_actor/train_logs/Value_Function_Planning/sim2real_grasp_v3/best.pth",
)
DEFAULT_PLACE_CKPT = os.path.join(
    VISPLAN_WM,
    "models/flowmatch_actor/train_logs/Value_Function_Planning/sim2real_place_v3/best.pth",
)
DEFAULT_VALUE_CKPT = os.path.join(
    VISPLAN_WM,
    "models/flowmatch_actor/train_logs/Value_Function_Planning/sim2real_q_value_v3/best.pth",
)


# --------------------------------------------------------------------------- #
# Pose post-processing: force top-down (gripper z-axis = -world z), preserve yaw.
# --------------------------------------------------------------------------- #

def make_topdown(pose7_wxyz):
    """
    Project a (7,) pose [x, y, z, qw, qx, qy, qz] onto the top-down manifold:
    gripper local z-axis aligned with -world_z. Yaw around world z is taken
    from the input's gripper x-axis direction. Position is unchanged.
    """
    is_tensor = isinstance(pose7_wxyz, torch.Tensor)
    p = pose7_wxyz.detach().cpu().numpy() if is_tensor else np.asarray(pose7_wxyz)
    pos = p[:3]
    qw, qx, qy, qz = p[3:7]
    mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
    # Yaw from gripper x-axis projection onto world xy.
    x_world = mat[:, 0]
    yaw = float(np.arctan2(x_world[1], x_world[0]))
    # Compose: rotate 180° about x (z -> -z), then yaw about world z.
    R_target = R.from_euler("z", yaw) * R.from_euler("x", np.pi)
    qx_o, qy_o, qz_o, qw_o = R_target.as_quat()
    out = np.concatenate([pos, [qw_o, qx_o, qy_o, qz_o]]).astype(np.float32)

    out[..., 0] = out[..., 0] - 0.02
    out[..., 1] = out[..., 1] - 0.02
    out[..., 2] = out[..., 2] + 0.04

    return torch.from_numpy(out) if is_tensor else out


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #

def visualize_poses_in_pointcloud(pcd, poses, rgb=None, colors=None):
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
            length=0.05,
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
# Robot execution of the predicted grasp (taken from primitives_demo.py)
# --------------------------------------------------------------------------- #

def execute_grasp(grasp_pose: torch.Tensor, pcd_np: np.ndarray, device: torch.device):
    """
    Mirror primitives_demo.main() end-to-end, with the policy-predicted grasp
    pose substituted for the hardcoded one. Sequence:

        home -> pre_grasp -> grasp -> close -> lift -> inter -> target_shelf
             -> open -> reverse_target -> home

    Args:
        grasp_pose: (7,) torch.Tensor xyz + quat_wxyz in robot-base frame.
        pcd_np: scene point cloud used to populate the motion planner world.
        device: torch device for joint/pose tensors (motion_planner expects cuda).
    """
    controller = FrankaPandaController()
    controller.move_to_joints(controller.home_joints, controller.open_gripper_action)

    current_joints = controller.get_robot_joints()
    current_joints = torch.tensor(current_joints, dtype=torch.float32, device=device)

    motion_planner = MotionPlanner(pcd_np)

    # === Grasp pose comes from the policy (replaces primitives_demo hardcode) ===
    grasp_pose = grasp_pose.to(device=device, dtype=torch.float32)

    # Pre-grasp: back off 12 cm along local gripper z-axis.
    pre_grasp_pose = move_pose_along_local_z(grasp_pose, -0.12)
    pre_grasp_pose = torch.tensor(pre_grasp_pose, dtype=torch.float32, device=device)

    # Target shelf pose — hardcoded, matches primitives_demo.
    # TODO: edge of the shelf, not inside it.
    target_shelf_pose = torch.tensor(
        [0.559, -0.06, 0.285, 0.7071, -0.7071, 0.0, 0.0],
        dtype=torch.float32, device=device,
    )
    target_shelf_pose = torch.tensor(
        rotate_pose_around_local_z(target_shelf_pose, np.pi / 2, format='wxyz'),
        dtype=torch.float32, device=device,
    )
    
    # Lift: grasp xyz with z lifted to shelf height.
    lift_pose = grasp_pose.clone()
    lift_pose[2] = target_shelf_pose[2]

    # Intermediate retraction in front of shelf before insertion.
    inter_pose = target_shelf_pose.clone()
    inter_pose[0] = target_shelf_pose[0]
    inter_pose[1] = -0.25
    inter_pose[2] = target_shelf_pose[2]

    # Visualize the planned waypoints before executing anything.
    # cyan=pre_grasp, green=grasp, yellow=lift, orange=inter, blue=target_shelf.
    print("Visualizing intermediate poses in pcd: "
          "cyan=pre_grasp, green=grasp, yellow=lift, orange=inter, blue=target")
    visualize_poses_in_pointcloud(
        pcd_np,
        [pre_grasp_pose, grasp_pose, lift_pose, inter_pose, target_shelf_pose],
        rgb=None,
        colors=[
            (0.0, 1.0, 1.0),  # cyan
            (0.0, 1.0, 0.0),  # green
            (1.0, 1.0, 0.0),  # yellow
            (1.0, 0.5, 0.0),  # orange
            (0.0, 0.0, 1.0),  # blue
        ],
    )

    controller.open_gripper()

    pre_grasp_trajectories, pre_grasp_success = motion_planner.plan_to_goal_poses(
        current_joints=current_joints.unsqueeze(0),
        goal_poses=pre_grasp_pose.unsqueeze(0),
    )
    print(f"  pre_grasp success: {pre_grasp_success.item()}")

    grasp_trajectories, grasp_success = motion_planner.plan_to_goal_poses(
        current_joints=pre_grasp_trajectories[0, -1].unsqueeze(0),
        goal_poses=grasp_pose.unsqueeze(0),
        disable_collision_links=motion_planner.links[-5:],
        plan_config=motion_planner.along_z_axis_plan_config,
    )
    print(f"  grasp success: {grasp_success.item()}")

    lift_trajectories, lift_success = motion_planner.plan_to_goal_poses(
        current_joints=grasp_trajectories[0, -1].unsqueeze(0),
        goal_poses=lift_pose.unsqueeze(0),
        disable_collision_links=motion_planner.links[-5:],
        plan_config=motion_planner.lift_plan_config,
    )
    print(f"  lift success: {lift_success.item()}")

    inter_pose_trajectories, inter_pose_success = motion_planner.plan_to_goal_poses(
        current_joints=lift_trajectories[0, -1].unsqueeze(0),
        goal_poses=inter_pose.unsqueeze(0),
        plan_config=motion_planner.only_xy_translation_plan_config,
    )
    print(f"  inter_pose success: {inter_pose_success.item()}")

    target_trajectories, target_success = motion_planner.plan_to_goal_poses(
        current_joints=inter_pose_trajectories[0, -1].unsqueeze(0),
        goal_poses=target_shelf_pose.unsqueeze(0),
        disable_collision_links=motion_planner.links[-5:],
        plan_config=motion_planner.along_z_axis_plan_config,
    )
    print(f"  target success: {target_success.item()}")

    success = (
        pre_grasp_success.item()
        & grasp_success.item()
        & lift_success.item()
        & inter_pose_success.item()
        & target_success.item()
    )
    if not success:
        print("  ABORT: one or more plans failed.")
        controller.move_to_joints(controller.home_joints, controller.close_gripper_action)
        controller.open_gripper()
        return

    controller.move_along_trajectory(
        pre_grasp_trajectories[0].cpu().numpy(), controller.open_gripper_action
    )
    controller.move_along_trajectory(
        grasp_trajectories[0].cpu().numpy(), controller.open_gripper_action
    )
    controller.close_gripper(num_steps=80)
    controller.move_along_trajectory(
        lift_trajectories[0].cpu().numpy(), controller.close_gripper_action
    )
    controller.move_along_trajectory(
        inter_pose_trajectories[0].cpu().numpy(), controller.close_gripper_action
    )
    controller.move_along_trajectory(
        target_trajectories[0].cpu().numpy(), controller.close_gripper_action
    )
    controller.open_gripper(num_steps=80)
    controller.move_along_trajectory(
        target_trajectories[0].flip(0).cpu().numpy(), controller.open_gripper_action
    )

    controller.move_to_joints(controller.home_joints, controller.close_gripper_action)
    controller.open_gripper()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp_ckpt", default=DEFAULT_GRASP_CKPT)
    parser.add_argument("--place_ckpt", default=DEFAULT_PLACE_CKPT)
    parser.add_argument("--value_ckpt", default=DEFAULT_VALUE_CKPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--publish_port", type=int, default=1235)
    parser.add_argument("--use_value", action="store_true",
                        help="Also load and run the value network; print + use the scalar score.")
    parser.add_argument("--no_viz", action="store_true",
                        help="Skip Open3D visualization (headless).")
    parser.add_argument("--no_exec", dest="execute", action="store_false",
                        help="Skip robot execution (default: execute the predicted grasp).")
    parser.set_defaults(execute=True)
    parser.add_argument("--target_label", type=str, default=None,
                        help="Pick this segmentation label (e.g. 'blue block') as the target. "
                             "If unset and seg_labels are present, picks the first non-empty label. "
                             "Falls back to a random DBSCAN cluster if no seg_labels in payload.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Inference object (loads value net only if --use_value).
    inference = PolicyValueInference(
        grasp_ckpt=args.grasp_ckpt,
        place_ckpt=args.place_ckpt,
        value_ckpt=args.value_ckpt if args.use_value else None,
        device=device,
        use_value=args.use_value,
    )

    # 2. Perception — pipeline does bounds + clustering + FPS to num_points.
    print("Connecting to perception pipeline")
    perception = PerceptionPipeline(publish_port=args.publish_port, timeout_ms=10000)
    print("Capturing point cloud")
    try:
        data = perception.get_point_cloud_dict()
    except TimeoutError:
        print("Timeout waiting for PCD. Is `python -m frankapanda.perception."
              "perception_pipeline --continuous --segment` (or --cluster) running?")
        return
    pcd_np = data["pcd"]
    rgb_np = data["rgb"]
    seg_labels = data.get("seg_labels")
    seg_label_names = data.get("seg_label_names")
    cluster_labels = data.get("cluster_labels")

    # 3. Build target mask. Prefer segmentation labels when present.
    if seg_labels is not None and seg_label_names is not None:
        seg_labels = np.asarray(seg_labels)
        per_label_counts = {n: int((seg_labels == k).sum()) for k, n in enumerate(seg_label_names)}
        print(f"  pcd: {pcd_np.shape}  seg per-label counts: {per_label_counts}")
        if args.target_label is not None:
            if args.target_label not in seg_label_names:
                raise RuntimeError(
                    f"--target_label='{args.target_label}' not in {seg_label_names}"
                )
            chosen_idx = seg_label_names.index(args.target_label)
        else:
            # First non-empty label.
            chosen_idx = next(
                (k for k in range(len(seg_label_names)) if (seg_labels == k).any()), None
            )
            if chosen_idx is None:
                raise RuntimeError("All segmentation labels are empty on this frame.")
        chosen_name = seg_label_names[chosen_idx]
        mask_np = (seg_labels == chosen_idx).astype(np.float32)
        print(f"  chosen seg label: {chosen_idx} ('{chosen_name}') "
              f"({int(mask_np.sum())}/{len(pcd_np)} points)")
    elif cluster_labels is not None:
        cluster_labels = np.asarray(cluster_labels)
        num_clusters = int(cluster_labels.max() + 1)
        print(f"  pcd: {pcd_np.shape}  clusters: {num_clusters} (no seg_labels — falling back)")
        chosen_cid = int(np.random.randint(num_clusters))
        mask_np = (cluster_labels == chosen_cid).astype(np.float32)
        print(f"  chosen cluster: {chosen_cid} ({int(mask_np.sum())}/{len(pcd_np)} points)")
    else:
        raise RuntimeError(
            "Pipeline payload has neither seg_labels nor cluster_labels. Re-run "
            "perception_pipeline with --segment or --cluster."
        )

    # 4. Visualize the chosen cluster (red) on full pcd (gray).
    # if not args.no_viz:
    #     viz_rgb = np.where(
    #         mask_np[:, None].astype(bool),
    #         np.array([1.0, 0.0, 0.0], dtype=np.float32),
    #         np.array([0.5, 0.5, 0.5], dtype=np.float32),
    #     )
    #     plot_pcd(pcd_np, viz_rgb, base_frame=True)

    # 5. Inference.
    result = inference.infer(pcd_np, mask_np)
    grasp_pose_robot = result["grasp_pose"]
    place_pose_robot = result["place_pose"]

    # Force top-down grasp: gripper z-axis -> -world z, yaw preserved from policy.
    grasp_pose_robot = make_topdown(grasp_pose_robot)
    print(f"  grasp pose (xyz+quat_wxyz, robot frame, top-down): {grasp_pose_robot.tolist()}")
    print(f"  place pose (xyz+quat_wxyz, robot frame, raw): {place_pose_robot.tolist()}")
    if result["value_score"] is not None:
        print(f"  value score (sigmoid): {result['value_score']:.4f}")

    # 6. Visualize grippers + cluster highlight.
    # if not args.no_viz:
    #     print("Visualizing: red=target cluster, gray=rest, green=grasp, blue=place")
    #     viz_rgb = np.where(
    #         mask_np[:, None].astype(bool),
    #         np.array([1.0, 0.0, 0.0], dtype=np.float32),
    #         np.array([0.5, 0.5, 0.5], dtype=np.float32),
    #     )
    #     visualize_poses_in_pointcloud(
    #         pcd_np,
    #         [grasp_pose_robot, place_pose_robot],
    #         rgb=viz_rgb,
    #         colors=[(0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
    #     )

    # 7. Optional execution of the predicted grasp on the real robot.
    if args.execute:
        print("Executing predicted grasp on robot")
        execute_grasp(grasp_pose_robot, pcd_np, device)

    perception.close()


if __name__ == "__main__":
    main()
