"""
Capture one ZED RGB frame, detect an ArUco marker, estimate its pose, and save
the RGB/visualization images.
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
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
        print("Opening ZED serial:", serial_number)
        try:
            init.set_from_serial_number(serial_number)
        except AttributeError:
            input_type = sl.InputType()
            input_type.set_from_serial_number(serial_number)
            init.input = input_type
    elif camera_id is not None:
        print("Opening ZED camera id:", camera_id)
        input_type = sl.InputType()
        input_type.set_from_camera_id(camera_id)
        init.input = input_type
    else:
        print("Opening default ZED camera")

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED: {status}")

    cam_info = zed.get_camera_information()
    print("Opened model:", cam_info.camera_model)
    print("Opened serial:", cam_info.serial_number)
    return zed


def read_zed_rgb(zed, sl, frames):
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


def detect_aruco_markers(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if image.ndim == 3:
        vis_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        vis_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.aruco.drawDetectedMarkers(vis_image, corners, ids)
    cv2.imshow("ArUco Marker Detection", vis_image)
    cv2.waitKey(0 if debug else 1)

    return corners, ids, vis_image


def draw_marker_pose(vis_image, corners, ids, camera_matrix, dist_coeffs):
    marker_pose = None

    if ids is None or len(ids) == 0:
        print("No ArUco markers detected in the captured frame.")
        return marker_pose, vis_image

    print(f"Detected {len(ids)} markers: ids={ids.flatten()}")

    transform_matrix = estimate_transformation(corners, ids, camera_matrix, dist_coeffs)
    if transform_matrix is None:
        print("estimate_transformation returned None.")
        return marker_pose, vis_image

    marker_pose = transform_matrix
    t = transform_matrix[:3, 3]
    rmat = transform_matrix[:3, :3]
    rot = Rotation.from_matrix(rmat)
    quat = rot.as_quat()
    euler = rot.as_euler("xyz", degrees=True)

    print("Marker pose (camera frame):")
    print(f" translation (m): {t}")
    print(f" quaternion (xyzw): {quat}")
    print(f" euler xyz (deg): {euler}")

    cv2.putText(
        vis_image,
        f"t=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        vis_image,
        f"eul=[{euler[0]:.1f},{euler[1]:.1f},{euler[2]:.1f}]",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    return marker_pose, vis_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-number", type=int, default=None)
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--resolution", type=str, default="HD720")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    import pyzed.sl as sl

    # Match test_franka.py: create the controller, but do not command motion here.
    FrankaOSCController(tip_offset=np.zeros(3))

    zed = open_zed_camera(
        sl,
        serial_number=args.serial_number,
        camera_id=args.camera_id,
        resolution=args.resolution,
        fps=args.fps,
    )

    try:
        camera_matrix, dist_coeffs = get_zed_left_calib(zed)
        rgb = read_zed_rgb(zed, sl, args.frames)
    finally:
        zed.close()

    if rgb is None:
        print("No RGB frame captured.")
        return

    corners, ids, vis_on_rgb = detect_aruco_markers(rgb, debug=args.debug)
    _, vis_on_rgb = draw_marker_pose(vis_on_rgb, corners, ids, camera_matrix, dist_coeffs)

    os.makedirs(args.out_dir, exist_ok=True)
    serial_label = str(args.serial_number) if args.serial_number is not None else "zed"
    rgb_out = args.out_dir / f"last_zed_rgb_{serial_label}.png"
    vis_out = args.out_dir / f"last_zed_vis_on_rgb_{serial_label}.png"

    cv2.imwrite(str(rgb_out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(vis_out), vis_on_rgb)
    print("Saved RGB frame to", rgb_out)
    print("Saved visualization on RGB to", vis_out)

    cv2.imshow("Last ZED RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow("Last ZED Marker Detection", vis_on_rgb)
    cv2.waitKey(0 if args.debug else 1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
