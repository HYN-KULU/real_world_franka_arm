"""
Minimal Demo: Real-world Policy + Value Inference (no robot exec)

What it does:
1. Grabs a single point cloud from the dual-Kinect perception pipeline.
2. Downsamples to 4096 points (farthest-point).
3. Applies the sim2real preprocessing (per-sample center + unit-radius
   scale) the policy/value networks were trained on.
4. Loads the latest sim2real grasp, place, and value checkpoints.
5. Forwards through grasp policy, then place policy (conditioned on
   the grasp), then the value network on (pcd, [grasp, place]).
6. Prints the predicted poses and value score; visualizes the predicted
   gripper poses in the point cloud.

Run prerequisites:
1. Perception pipeline running:
     python -m frankapanda.perception.perception_pipeline --continuous
2. PYTHONPATH/.pth set up so `models.flowmatch_actor.*` and
   `visplan.submodules.robo_utils.*` resolve out of visplanWM. The
   setup_frankapanda_3dfa.sh script drops a visplanWM.pth into the env
   site-packages.

Usage:
    conda activate frankapanda_3dfa
    python infer_policy_value_demo.py
"""

import argparse
import os
import time

import numpy as np
import open3d as o3d
import torch

# robo_utils (frankapanda's editable install — same API as visplanWM's submodule)
from robo_utils.visualization.plotting import plot_pcd
from robo_utils.visualization.point_cloud_structures import make_gripper_visualization
from robo_utils.conversion_utils import (
    furthest_point_sample,
    pose_to_transformation,
)

# Perception client.
from frankapanda.perception import PerceptionPipeline

# visplanWM model code, resolved via the .pth that points sys.path at
# /home/ksaha/Research/ModelBasedPlanning/visplanWM/
from models.flowmatch_actor.modeling.policy_grasp_place.denoise_actor_3d_packing import (
    DenoiseActor as GraspPlaceActor,
)
from models.flowmatch_actor.modeling.policy.value_network import ValueNetwork


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

NUM_POINTS = 4096

# Policy ctor kwargs — must match what train_for_shelf_packing_grasp_place.py
# and visplan/action_sampling.py:_POLICY_KWARGS use for the cascaded
# grasp/place pair.
_POLICY_KWARGS = dict(
    embedding_dim=120,
    num_attn_heads=8,
    nhist=1,
    num_shared_attn_layers=4,
    relative=False,
    rotation_format="quat_wxyz",
    denoise_timesteps=10,
    denoise_model="rectified_flow",
    lv2_batch_size=1,
    pcd_input_channels=4,
)

VALUE_KWARGS = dict(
    embedding_dim=120,
    num_attn_heads=8,
    num_shared_attn_layers=4,
)


# --------------------------------------------------------------------------- #
# Checkpoint loader (mirrors visplan.shelf_packing_base.load_checkpoint_for_eval)
# --------------------------------------------------------------------------- #

def load_checkpoint_for_eval(checkpoint_path: str, model: torch.nn.Module) -> torch.nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    try:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as e:
        if "numpy.core.multiarray.scalar" in str(e):
            blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        else:
            raise
    msn, unx = model.load_state_dict(blob["weight"], strict=False)
    if msn:
        print(f"  [{os.path.basename(checkpoint_path)}] missing keys: {len(msn)}")
    if unx:
        print(f"  [{os.path.basename(checkpoint_path)}] unexpected keys: {len(unx)}")
    if not msn and not unx:
        print(f"  [{os.path.basename(checkpoint_path)}] all keys matched")
    del blob
    return model


def build_models(grasp_ckpt: str, place_ckpt: str, value_ckpt: str, device: torch.device):
    print("Building grasp_only policy")
    grasp_kwargs = dict(_POLICY_KWARGS)
    grasp_kwargs["trajectory_length"] = 1
    grasp_kwargs["grasp_condition_dim"] = None
    grasp_model = GraspPlaceActor(**grasp_kwargs)
    grasp_model = load_checkpoint_for_eval(grasp_ckpt, grasp_model).to(device).eval()

    print("Building place_conditional policy")
    place_kwargs = dict(_POLICY_KWARGS)
    place_kwargs["trajectory_length"] = 1
    place_kwargs["grasp_condition_dim"] = 7
    place_model = GraspPlaceActor(**place_kwargs)
    place_model = load_checkpoint_for_eval(place_ckpt, place_model).to(device).eval()

    print("Building value network")
    value_model = ValueNetwork(**VALUE_KWARGS)
    value_model = load_checkpoint_for_eval(value_ckpt, value_model).to(device).eval()

    return grasp_model, place_model, value_model


# --------------------------------------------------------------------------- #
# Sim2real per-sample preprocessing (center + max-radius scale)
# Matches sim2real_augment.center_and_scale (deterministic, no augmentation).
# +0.615 is a no-op after centering, so we skip it for real PCDs that
# already live in robot-base frame.
# --------------------------------------------------------------------------- #

def center_and_scale(pcd_xyz: torch.Tensor, eps: float = 1e-6):
    centroid = pcd_xyz.mean(dim=0)                       # (3,)
    centered = pcd_xyz - centroid
    radius = centered.norm(dim=-1).max()
    scale = torch.clamp(radius, min=eps)
    return centered / scale, centroid, scale


def unscale_action_xyz(xyz_norm: torch.Tensor, centroid: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return xyz_norm * scale + centroid


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
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp_ckpt", default=DEFAULT_GRASP_CKPT)
    parser.add_argument("--place_ckpt", default=DEFAULT_PLACE_CKPT)
    parser.add_argument("--value_ckpt", default=DEFAULT_VALUE_CKPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--publish_port", type=int, default=1235)
    parser.add_argument("--no_viz", action="store_true",
                        help="Skip Open3D visualization (headless).")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Models
    grasp_model, place_model, value_model = build_models(
        args.grasp_ckpt, args.place_ckpt, args.value_ckpt, device
    )

    # 2. Perception
    print("Connecting to perception pipeline")
    perception = PerceptionPipeline(publish_port=args.publish_port, timeout_ms=10000)
    print("Capturing point cloud")
    try:
        pcd_np, rgb_np = perception.get_point_cloud()
    except TimeoutError:
        print("Timeout waiting for PCD. Is `python -m frankapanda.perception."
              "perception_pipeline --continuous` running?")
        return
    print(f"  raw pcd: {pcd_np.shape}  rgb: {rgb_np.shape}")

    # z-bounds now applied upstream in perception_pipeline (bounds dict);
    # pcd arrives clean within [table, roof].

    # DBSCAN cluster the masked pcd into object groups.
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
    cluster_labels = np.asarray(pcd_o3d.cluster_dbscan(eps=0.02, min_points=20, print_progress=False))
    num_clusters = int(cluster_labels.max() + 1) if cluster_labels.size and cluster_labels.max() >= 0 else 0
    num_noise = int((cluster_labels == -1).sum())
    print(f"  dbscan: {num_clusters} clusters, {num_noise} noise points / {len(pcd_np)} total")

    # Reassign noise points to nearest cluster (no -1 allowed).
    if num_noise > 0 and num_clusters > 0:
        non_noise = cluster_labels != -1
        kdtree = o3d.geometry.KDTreeFlann(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_np[non_noise]))
        )
        non_noise_labels = cluster_labels[non_noise]
        for ni in np.where(~non_noise)[0]:
            _, idx, _ = kdtree.search_knn_vector_3d(pcd_np[ni], 1)
            cluster_labels[ni] = non_noise_labels[idx[0]]
        print(f"  reassigned {num_noise} noise points to nearest cluster")
    elif num_clusters == 0:
        raise RuntimeError("DBSCAN found no clusters; loosen eps / lower min_points.")

    # Pick a random cluster as the target object; build per-point mask.
    chosen_cid = int(np.random.randint(num_clusters))
    full_mask = (cluster_labels == chosen_cid).astype(np.float32)
    print(f"  chosen cluster: {chosen_cid} ({int(full_mask.sum())}/{len(pcd_np)} points)")

    # Visualize the full pcd with the chosen cluster highlighted (red=target, gray=rest).
    viz_rgb = np.where(
        full_mask[:, None].astype(bool),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.5, 0.5], dtype=np.float32),
    )
    plot_pcd(pcd_np, viz_rgb, base_frame=True)

    # 3. Downsample to NUM_POINTS via FPS (robo_utils wraps open3d).
    t0 = time.time()
    pcd_fps_np = furthest_point_sample(pcd_np, num_points=NUM_POINTS)
    print(f"  fps -> {pcd_fps_np.shape} in {time.time() - t0:.2f}s")

    # Map each FPS point back to its nearest neighbor in pcd_np to carry mask through.
    full_tree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_np))
    )
    fps_indices = np.empty(len(pcd_fps_np), dtype=np.int64)
    for i, p in enumerate(pcd_fps_np):
        _, idx, _ = full_tree.search_knn_vector_3d(p, 1)
        fps_indices[i] = idx[0]
    fps_mask_np = full_mask[fps_indices]                            # (N,)

    pcd_fps = torch.from_numpy(pcd_fps_np).float()                 # (N, 3) robot frame

    # 4. Per-sample center + scale (the sim2real normalization).
    pcd_norm, centroid, scale = center_and_scale(pcd_fps)
    print(f"  centroid={centroid.tolist()}  scale={scale.item():.4f}")
    pcd_norm_dev = pcd_norm.unsqueeze(0).to(device)                 # (1, N, 3) — value net input (xyz only)

    # 4b. Build 4-channel pcd (xyz + target_mask) for grasp/place policies.
    # Mask = 1.0 on target-object points, 0.0 elsewhere. Only xyz is normalized.
    target_mask = torch.from_numpy(fps_mask_np).to(pcd_norm.dtype).unsqueeze(-1)  # (N, 1)
    pcd_norm_masked_dev = torch.cat(
        [pcd_norm, target_mask], dim=-1
    ).unsqueeze(0).to(device)                                       # (1, N, 4)
    print(f"  target_mask: {int(target_mask.sum())}/{target_mask.numel()} fps points marked (cluster {chosen_cid})")

    # 5. Grasp policy forward.
    with torch.no_grad():
        grasp_out = grasp_model(
            gt_trajectory=None, pcd=pcd_norm_masked_dev,
            proprioception=None, run_inference=True,
        )                                                            # (1, 1, 8)
    grasp_pose_norm = grasp_out[0, 0].cpu()                          # (8,) pos+quat_wxyz+gripper
    grasp_pos_robot = unscale_action_xyz(
        grasp_pose_norm[:3], centroid, scale,
    )
    grasp_pose_robot = torch.cat([grasp_pos_robot, grasp_pose_norm[3:7]])  # (7,)
    print(f"  grasp pose (robot frame, xyz+quat_wxyz): {grasp_pose_robot.tolist()}")

    # 6. Place policy forward, conditioned on the grasp pose.
    # The place model expects `grasp_cond` in the SAME normalized frame
    # the PCD lives in — same center+scale.
    grasp_cond_norm = torch.cat([grasp_pose_norm[:3], grasp_pose_norm[3:7]]).unsqueeze(0)  # (1, 7)
    grasp_cond_dev = grasp_cond_norm.to(device)
    with torch.no_grad():
        place_out = place_model(
            gt_trajectory=None, pcd=pcd_norm_masked_dev,
            proprioception=None, run_inference=True,
            grasp_cond=grasp_cond_dev,
        )                                                            # (1, 1, 8)
    place_pose_norm = place_out[0, 0].cpu()
    place_pos_robot = unscale_action_xyz(
        place_pose_norm[:3], centroid, scale,
    )
    place_pose_robot = torch.cat([place_pos_robot, place_pose_norm[3:7]])
    print(f"  place pose (robot frame, xyz+quat_wxyz): {place_pose_robot.tolist()}")

    # 7. Value network — takes pcd (B,N,3) + actions (B,2,8) in the same
    # normalized frame the policy used. Use the raw 8-D pose (with the
    # mode-hardcoded gripper channel) directly.
    actions_norm = torch.stack([grasp_pose_norm, place_pose_norm], dim=0).unsqueeze(0)  # (1,2,8)
    actions_norm_dev = actions_norm.to(device)
    with torch.no_grad():
        score = value_model(pcd=pcd_norm_dev, actions=actions_norm_dev).cpu()
    print(f"  value score (sigmoid): {score.item():.4f}")

    # 8. Visualize predicted gripper poses on the (unnormalized) PCD.
    # Color FPS pcd by target_mask (red=target cluster, gray=rest); overlay
    # grippers via make_gripper_visualization (green=grasp, blue=place).
    if not args.no_viz:
        print("Visualizing: red=target cluster, gray=rest, green=grasp, blue=place")
        fps_viz_rgb = np.where(
            fps_mask_np[:, None].astype(bool),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )
        visualize_poses_in_pointcloud(
            pcd_fps_np,
            [grasp_pose_robot, place_pose_robot],
            rgb=fps_viz_rgb,
            colors=[(0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        )


if __name__ == "__main__":
    main()
