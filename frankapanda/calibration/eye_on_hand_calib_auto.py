"""
Automatically collect eye-on-hand calibration data with the wrist ZED camera.

Each saved sample is a dict containing the gripper pose, marker pose, marker
corners/ids, ZED intrinsics/distortion, and marker size.
"""
import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from deoxys.utils import YamlConfig

from eye_on_hand_calib import (
    detect_zed_marker_pose,
    get_zed_left_calib,
    open_zed_camera,
    read_zed_rgb,
)
from robot_ik_controller import RobotIKController


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOINT_POSITION_CFG = PROJECT_ROOT / "configs" / "joint-position_controller.yml"
DEFAULT_INITIAL_JOINT_POSITIONS = [-0.10850218325121361, -0.4985320863472843, 0.1970343456310138, -2.091813034191466, 0.027444257930834704, 1.8022506055261363, 0.15217534011715142]


def quat_to_rot_matrix(quat, order="xyzw"):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Args:
        quat: Quaternion values.
        order: Either "xyzw" or "wxyz". scipy Rotation.as_quat() returns "xyzw";
            PyTorch3D quaternion_to_matrix expects "wxyz".
    """
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    if order == "xyzw":
        x, y, z, w = quat
    elif order == "wxyz":
        w, x, y, z = quat
    else:
        raise ValueError("order must be either 'xyzw' or 'wxyz'")

    norm = np.linalg.norm([w, x, y, z])
    if norm == 0.0:
        raise ValueError("Cannot convert a zero-norm quaternion to a rotation matrix")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def collect_auto_data(
        cam_id,
        num_movements,
        initial_joint_positions,
        debug=False,
        zed_serial_number=None,
        zed_camera_id=None,
        zed_resolution="HD720",
        zed_fps=30,
        zed_frames=5,
        pos_range=0.06,
        rot_range=0.5,
        settle_time=0.2,
        reset_each_sample=True):
    controller_cfg = YamlConfig(str(JOINT_POSITION_CFG)).as_easydict()
    robot = RobotIKController(
        controller_cfg=controller_cfg,
        controller_type="JOINT_POSITION",
        impedance_control=False,
        use_bullet=True,
        binary_grasping=False,
    )

    import pyzed.sl as sl

    zed = open_zed_camera(
        sl,
        serial_number=zed_serial_number,
        camera_id=zed_camera_id,
        resolution=zed_resolution,
        fps=zed_fps,
    )
    camera_matrix, dist_coeffs = get_zed_left_calib(zed)
    data = []
    pbar = None
    try:
        attempt_idx = 0
        pbar = tqdm(total=num_movements)
        while len(data) < num_movements:
            sample_idx = len(data)
            random_delta_pos = np.random.uniform(-pos_range, pos_range, size=(3,))
            random_delta_axis_angle = np.random.uniform(-rot_range, rot_range, size=(3,))

            print(f"[attempt {attempt_idx}] target delta pos: {random_delta_pos}")
            print(f"[attempt {attempt_idx}] target delta axis-angle: {random_delta_axis_angle}")

            reset_pos = np.array([0.485,-0.04095,0.3])
            # reset_quat is xyzw, matching scipy Rotation.as_quat().
            reset_quat = np.array([ 9.99392905e-01, -4.00813458e-05,  2.46601391e-02,  2.46109102e-02])
            target_rot = quat_to_rot_matrix(reset_quat, order="xyzw")
            # print(robot.eef_pose)
            # print(reset_quat)
            # print(target_rot)
            moved = robot.control(
                target_pos=reset_pos,
                target_rot=target_rot,
                grasping_action=0.0,
                wait_times=50,
            )

            current_pose = np.asarray(robot.eef_pose, dtype=np.float64)
            current_pos = current_pose[:3, 3]
            current_rot = current_pose[:3, :3]
            target_pos = current_pos + random_delta_pos
            target_rot = (
                Rotation.from_rotvec(random_delta_axis_angle)
                * Rotation.from_matrix(current_rot)
            ).as_matrix()

            moved = robot.control(
                target_pos=target_pos,
                target_rot=target_rot,
                grasping_action=0.0,
                wait_times=80,
            )
            attempt_idx += 1
            if not moved:
                print("\033[91m" + "Skipping sample because joint target was not reached." + "\033[0m")
                continue

            time.sleep(settle_time)
            gripper_pose = np.asarray(robot.eef_pose, dtype=np.float64)
            print(f"Gripper pos: {gripper_pose[:3, 3].flatten()}")

            rgb = read_zed_rgb(zed, sl, frames=zed_frames)
            save_prefix = f"data/eye_on_hand_wrist_zed_auto_{cam_id}_{sample_idx:03d}"
            detection = detect_zed_marker_pose(
                rgb=rgb,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                debug=debug,
                save_prefix=save_prefix,
            )

            if detection is not None:
                data.append({
                    "gripper_pose": gripper_pose,
                    "marker_pose": detection["marker_pose"],
                    "corners": detection["corners"],
                    "ids": detection["ids"],
                    "camera_matrix": detection["camera_matrix"],
                    "dist_coeffs": detection["dist_coeffs"],
                    "marker_size_m": detection["marker_size_m"],
                    "marker_length_m": detection["marker_length_m"],
                    "random_delta_pos": random_delta_pos,
                    "random_delta_axis_angle": random_delta_axis_angle,
                })
                print("\033[92m" + f"Saved sample {len(data)}" + "\033[0m")
                pbar.update(1)
            else:
                print("\033[91m" + "No valid marker pose; sample not saved." + "\033[0m")
    finally:
        try:
            if pbar is not None:
                pbar.close()
        except Exception:
            pass
        try:
            zed.close()
        except Exception:
            pass

    os.makedirs("data", exist_ok=True)
    filepath = f"data/eye_on_hand_wrist_zed_auto_{cam_id}_data_larger_range.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"Recorded {len(data)} valid data points.")
    print(f"Saved data to {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-id", type=int, default=0, help="Label used in saved filenames")
    parser.add_argument("--zed-serial-number", type=int, default=None)
    parser.add_argument("--zed-camera-id", type=int, default=None)
    parser.add_argument("--zed-resolution", type=str, default="HD720")
    parser.add_argument("--zed-fps", type=int, default=30)
    parser.add_argument("--zed-frames", type=int, default=5)
    parser.add_argument("--num-movements", type=int, default=30)
    parser.add_argument("--pos-range", type=float, default=0.03)
    parser.add_argument("--rot-range", type=float, default=0.9)
    parser.add_argument("--settle-time", type=float, default=0.2)
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    collect_auto_data(
        cam_id=args.cam_id,
        num_movements=args.num_movements,
        initial_joint_positions=DEFAULT_INITIAL_JOINT_POSITIONS,
        debug=args.debug,
        zed_serial_number=args.zed_serial_number,
        zed_camera_id=args.zed_camera_id,
        zed_resolution=args.zed_resolution,
        zed_fps=args.zed_fps,
        zed_frames=args.zed_frames,
        pos_range=args.pos_range,
        rot_range=args.rot_range,
        settle_time=args.settle_time,
        reset_each_sample=not args.no_reset,
    )


if __name__ == "__main__":
    main()
