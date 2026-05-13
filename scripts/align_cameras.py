"""
Compute alignment transformation from camera 1 to camera 0 using ICP.

Subscribes to the perception pipeline publisher (run with --no_downsample)
to receive per-camera bounds-filtered point clouds. ICP gives a residual
correction relative to the current cam1_to_cam0; composes and saves the
updated cam1_to_cam0 and inverse cam0_to_cam1 transformations.

Usage:
    # Terminal 1
    python frankapanda/perception/perception_pipeline.py --no_downsample
    # Terminal 2
    python scripts/align_cameras.py
"""
import argparse
import os
import pickle
import numpy as np
import open3d as o3d
import zmq
from robo_utils.conversion_utils import invert_transformation


def compute_icp_alignment(source_pcd, target_pcd, threshold=0.01, visualize=False):
    """
    Compute ICP alignment from source to target point cloud.

    Args:
        source_pcd: Open3D point cloud (source)
        target_pcd: Open3D point cloud (target)
        threshold: Distance threshold for ICP
        visualize: Whether to visualize the alignment result

    Returns:
        4x4 transformation matrix that transforms source to align with target
    """
    print(f"Source point cloud: {len(source_pcd.points)} points")
    print(f"Target point cloud: {len(target_pcd.points)} points")

    print("Estimating normals...")
    source_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    trans_init = np.identity(4)

    print(f"Running ICP with threshold={threshold}...")
    reg_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    print("\nICP Registration Result:")
    print(reg_result)
    print(f"Fitness: {reg_result.fitness}")
    print(f"Inlier RMSE: {reg_result.inlier_rmse}")

    if visualize:
        source_temp = source_pcd.copy()
        target_temp = target_pcd.copy()
        source_temp.paint_uniform_color([1, 0, 0])
        target_temp.paint_uniform_color([0, 0, 1])
        source_temp.transform(reg_result.transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    return reg_result.transformation


def receive_from_pipeline(subscribe_port, timeout_ms):
    """Subscribe to perception pipeline publisher, return one message payload."""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://localhost:{subscribe_port}")
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

    print(f"Subscribed to perception pipeline on port {subscribe_port}")
    print(f"Waiting for per-camera point clouds (timeout: {timeout_ms}ms)...")
    try:
        data = pickle.loads(socket.recv())
    finally:
        socket.close()
        context.term()
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICP alignment from perception pipeline")
    parser.add_argument('--subscribe_port', type=int, default=1235,
                        help='ZMQ port to subscribe to (perception pipeline publish port)')
    parser.add_argument('--timeout_ms', type=int, default=120000,
                        help='Receive timeout in milliseconds')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='ICP correspondence distance threshold (meters)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize ICP result with Open3D')
    args = parser.parse_args()

    data = receive_from_pipeline(args.subscribe_port, args.timeout_ms)

    if not data.get('no_downsample'):
        raise RuntimeError(
            "Received payload missing 'no_downsample' marker. "
            "Run perception_pipeline.py with --no_downsample."
        )

    cam0_pcd_array = data['cam0_pcd']
    cam1_pcd_array = data['cam1_pcd']
    print(f"Camera 0: {cam0_pcd_array.shape[0]} points")
    print(f"Camera 1: {cam1_pcd_array.shape[0]} points")

    cam0_pcd = o3d.geometry.PointCloud()
    cam0_pcd.points = o3d.utility.Vector3dVector(cam0_pcd_array)

    cam1_pcd = o3d.geometry.PointCloud()
    cam1_pcd.points = o3d.utility.Vector3dVector(cam1_pcd_array)

    # ICP: cam1 -> cam0. cam1 already had existing cam1_to_cam0 applied upstream
    # (in capture_single_camera.py), so this yields a *residual* correction.
    print("\n" + "="*60)
    print("Computing residual alignment: Camera 1 -> Camera 0")
    print("="*60)
    residual_cam1_to_cam0 = compute_icp_alignment(
        cam1_pcd, cam0_pcd, threshold=args.threshold, visualize=args.visualize
    )
    print(f"\nResidual transformation:\n{residual_cam1_to_cam0}")

    alignment_dir = os.path.join("data", "camera_alignments")
    os.makedirs(alignment_dir, exist_ok=True)
    cam1_to_cam0_file = os.path.join(alignment_dir, "cam1_to_cam0.npy")
    cam0_to_cam1_file = os.path.join(alignment_dir, "cam0_to_cam1.npy")

    # Compose residual with existing alignment (capture_single_camera applied it).
    if os.path.exists(cam1_to_cam0_file):
        existing = np.load(cam1_to_cam0_file)
        cam1_to_cam0 = residual_cam1_to_cam0 @ existing
        print(f"\nComposed residual with existing alignment from {cam1_to_cam0_file}")
    else:
        cam1_to_cam0 = residual_cam1_to_cam0
        print(f"\nNo existing alignment at {cam1_to_cam0_file}; using residual as full alignment.")

    cam0_to_cam1 = invert_transformation(cam1_to_cam0)

    print(f"\nFinal cam1_to_cam0:\n{cam1_to_cam0}")
    print(f"\nFinal cam0_to_cam1:\n{cam0_to_cam1}")

    np.save(cam1_to_cam0_file, cam1_to_cam0)
    np.save(cam0_to_cam1_file, cam0_to_cam1)

    print("\n" + "="*60)
    print("Saved transformations:")
    print(f"  cam1_to_cam0: {cam1_to_cam0_file}")
    print(f"  cam0_to_cam1: {cam0_to_cam1_file}")
    print("="*60)
