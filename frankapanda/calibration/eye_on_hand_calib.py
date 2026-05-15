"""
Uses Deoxys to control the robot and collect data for calibration.
"""
import numpy as np
import os, pickle
import cv2
import time
import argparse
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from robot_controller import FrankaOSCController
from marker_detection import estimate_transformation


def get_zed_left_calib(zed):
    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    left = calib.left_cam

    camera_matrix = np.array(
        [
            [left.fx, 0.0, left.cx],
            [0.0, left.fy, left.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    for attr in ("disto", "distortion"):
        if hasattr(left, attr):
            dist_coeffs = np.asarray(getattr(left, attr), dtype=np.float32).reshape(-1, 1)
            break

    return camera_matrix, dist_coeffs


def open_zed_camera(sl, serial_number=None, camera_id=None, resolution="HD720", fps=30):
    resolution_map = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1200": sl.RESOLUTION.HD1200,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "SVGA": sl.RESOLUTION.SVGA,
        "VGA": sl.RESOLUTION.VGA,
        "AUTO": sl.RESOLUTION.AUTO,
    }

    res_key = resolution.upper()
    if res_key not in resolution_map:
        raise ValueError(f"Unsupported resolution {resolution}. Choices: {list(resolution_map)}")

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = resolution_map[res_key]
    init.camera_fps = fps
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode = sl.DEPTH_MODE.NONE

    if serial_number is not None:
        print("Opening wrist ZED serial:", serial_number)
        try:
            init.set_from_serial_number(serial_number)
        except AttributeError:
            input_type = sl.InputType()
            input_type.set_from_serial_number(serial_number)
            init.input = input_type
    elif camera_id is not None:
        print("Opening wrist ZED camera id:", camera_id)
        input_type = sl.InputType()
        input_type.set_from_camera_id(camera_id)
        init.input = input_type
    else:
        print("Opening default wrist ZED camera")

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open wrist ZED: {status}")

    cam_info = zed.get_camera_information()
    print("Opened wrist ZED model:", cam_info.camera_model)
    print("Opened wrist ZED serial:", cam_info.serial_number)
    return zed


def read_zed_rgb(zed, sl, frames=5):
    runtime = sl.RuntimeParameters()
    left_mat = sl.Mat()
    rgb = None

    for _ in range(max(1, frames)):
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            left_bgra = left_mat.get_data()
            bgr = left_bgra[:, :, :3].copy()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        time.sleep(0.02)

    return rgb


def detect_zed_marker_pose(rgb, camera_matrix, dist_coeffs, debug=False, save_prefix=None):
    """
    Detect the ArUco marker in wrist ZED RGB, estimate marker pose in the ZED
    camera frame, and save the same style visualization as test_franka_zed.py.
    """
    if rgb is None:
        print("\033[91m" + "No ZED RGB frame captured." + "\033[0m")
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    vis_on_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.aruco.drawDetectedMarkers(vis_on_rgb, corners, ids)

    def save_capture_images():
        if save_prefix is None:
            return
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
        cv2.imwrite(f"{save_prefix}_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{save_prefix}_vis_on_rgb.png", vis_on_rgb)

    if ids is None or len(ids) == 0:
        print("\033[91m" + "No markers detected in wrist ZED RGB." + "\033[0m")
        save_capture_images()
        return None

    print("\033[92m" + f"Detected {len(ids)} markers: ids={ids.flatten()}" + "\033[0m")
    transform_matrix = estimate_transformation(corners, ids, camera_matrix, dist_coeffs)
    if transform_matrix is None:
        print("estimate_transformation returned None.")
        save_capture_images()
        return None

    t = transform_matrix[:3, 3]
    rmat = transform_matrix[:3, :3]
    rot = Rotation.from_matrix(rmat)
    quat = rot.as_quat()
    euler = rot.as_euler("xyz", degrees=True)

    print("Marker pose (wrist ZED camera frame):")
    print(f" translation (m): {t}")
    print(f" quaternion (xyzw): {quat}")
    print(f" euler xyz (deg): {euler}")

    cv2.putText(
        vis_on_rgb,
        f"t=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        vis_on_rgb,
        f"eul=[{euler[0]:.1f},{euler[1]:.1f},{euler[2]:.1f}]",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    if debug:
        cv2.imshow("Wrist ZED Marker Detection", vis_on_rgb)
        cv2.waitKey(0)

    save_capture_images()
    return transform_matrix


def smooth_move_by(robot, delta_pos, delta_axis_angle):
    """
    Execute one large random delta as several smaller move_by calls.

    The total sampled range stays large, but each sub-target stays close to the
    smooth range that works well on the robot.
    """
    delta_pos = np.asarray(delta_pos, dtype=np.float64).reshape(3)
    delta_axis_angle = np.asarray(delta_axis_angle, dtype=np.float64).reshape(3)

    max_chunk_pos = 0.02
    max_chunk_rot = 0.15
    pos_chunks = int(np.ceil(np.max(np.abs(delta_pos)) / max_chunk_pos))
    rot_chunks = int(np.ceil(np.max(np.abs(delta_axis_angle)) / max_chunk_rot))
    num_chunks = max(1, pos_chunks, rot_chunks)

    chunk_delta_pos = delta_pos / num_chunks
    chunk_delta_axis_angle = delta_axis_angle / num_chunks

    print(f"Moving in {num_chunks} smooth chunks")
    for chunk_idx in range(num_chunks):
        print(
            f"  chunk {chunk_idx + 1}/{num_chunks}: "
            f"dp={chunk_delta_pos}, dr={chunk_delta_axis_angle}"
        )
        robot.move_by(
            chunk_delta_pos,
            chunk_delta_axis_angle,
            num_steps=40,
            num_additional_steps=20,
            max_delta_pos=0.012,
            pos_tolerance=0.006,
            rot_tolerance=0.08,
        )


def move_robot_and_record_data(
        cam_id,
        cam_type="zed",
        num_movements=3, 
        debug=False,
        initial_joint_positions=None,
        zed_serial_number=None,
        zed_camera_id=None,
        zed_resolution="HD720",
        zed_fps=30,
        zed_frames=5):
    """
    Move the robot to random poses and record the necessary data.
    """
    
    # Initialize the robot
    robot = FrankaOSCController(
        tip_offset=np.zeros(3),     # Set the default to 0 to disable accounting for the tip
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
    try:
        for movement_idx in tqdm(range(num_movements)):
            # Generate a random target delta pose
            random_delta_pos = np.random.uniform(-0.06, 0.06, size=(3,))
            random_delta_axis_angle = np.random.uniform(-0.5, 0.5, size=(3,))
            print(f"Total random delta pos: {random_delta_pos}")
            print(f"Total random delta axis-angle: {random_delta_axis_angle}")
            robot.reset(joint_positions=initial_joint_positions)
            # import pdb; pdb.set_trace()
            smooth_move_by(robot, random_delta_pos, random_delta_axis_angle)

            time.sleep(0.2)
            gripper_pose = robot.eef_pose
            print(f"Gripper pos: {gripper_pose[:3, 3].flatten()}")

            rgb = read_zed_rgb(zed, sl, frames=zed_frames)
            save_prefix = f"data/eye_on_hand_wrist_zed_{cam_id}_{movement_idx:03d}"
            marker_pose = detect_zed_marker_pose(
                rgb=rgb,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                debug=debug,
                save_prefix=save_prefix,
            )
            if marker_pose is not None:
                data.append((
                    gripper_pose,   # gripper pose in base
                    marker_pose,    # tag pose in wrist ZED camera
                ))
    finally:
        try:
            zed.close()
        except Exception:
            pass
        print(f"Recorded {len(data)} data points.")

    os.makedirs("data", exist_ok=True)
    filepath = f"data/eye_on_hand_wrist_zed_{cam_id}_data.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved data to {filepath}")
    return filepath


def manually_record_data(
        cam_id,
        num_samples=10,
        debug=False,
        zed_serial_number=None,
        zed_camera_id=None,
        zed_resolution="HD720",
        zed_fps=30,
        zed_frames=5):
    """
    Manually move the gripper to poses where the wrist ZED sees the marker.
    For each accepted marker detection, record:
        (gripper_pose_in_base, marker_pose_in_wrist_zed)
    """
    robot = FrankaOSCController(
        tip_offset=np.zeros(3),
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
    sample_idx = 0

    try:
        while sample_idx < num_samples:
            print()
            print(f"Sample {sample_idx + 1}/{num_samples}")
            print("Move the gripper so the wrist ZED can see the marker.")
            user_input = input("Press Enter to capture, 's' to skip, or 'q' to finish: ").strip().lower()

            if user_input == "q":
                break
            if user_input == "s":
                sample_idx += 1
                continue

            attempt_idx = 0
            while True:
                save_prefix = (
                    f"data/eye_on_hand_wrist_zed_{cam_id}_"
                    f"{sample_idx:03d}_attempt{attempt_idx:02d}"
                )
                rgb = read_zed_rgb(zed, sl, frames=zed_frames)
                marker_pose = detect_zed_marker_pose(
                    rgb=rgb,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    debug=debug,
                    save_prefix=save_prefix,
                )

                if marker_pose is not None:
                    gripper_pose = robot.eef_pose
                    print(f"Gripper pos: {gripper_pose[:3, 3].flatten()}")
                    gripper_rot = Rotation.from_matrix(gripper_pose[:3, :3])
                    print(f"Gripper quaternion (xyzw): {gripper_rot.as_quat()}")
                    print(f"Gripper euler xyz (deg): {gripper_rot.as_euler('xyz', degrees=True)}")
                    data.append((
                        gripper_pose,   # gripper pose in base
                        marker_pose,    # tag pose in wrist ZED camera
                    ))
                    print("\033[92m" + f"Saved sample {sample_idx + 1}" + "\033[0m")
                    sample_idx += 1
                    break

                retry_input = input(
                    "Marker not found. Move/adjust and press Enter to retry, "
                    "'s' to skip this sample, or 'q' to finish: "
                ).strip().lower()
                if retry_input == "q":
                    sample_idx = num_samples
                    break
                if retry_input == "s":
                    sample_idx += 1
                    break
                attempt_idx += 1
    finally:
        try:
            zed.close()
        except Exception:
            pass

    os.makedirs("data", exist_ok=True)
    filepath = f"data/eye_on_hand_wrist_zed_manual_{cam_id}_data.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"Recorded {len(data)} data points.")
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
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cam_id = args.cam_id

    manually_record_data(
        cam_id=cam_id,
        num_samples=args.num_samples,
        debug=args.debug,
        zed_serial_number=args.zed_serial_number,
        zed_camera_id=args.zed_camera_id,
        zed_resolution=args.zed_resolution,
        zed_fps=args.zed_fps,
        zed_frames=args.zed_frames,
    )
    

if __name__ == "__main__":
    main()
