"""
Uses Deoxys to control the robot and collect data for calibration.
"""
import time
import os
import numpy as np
import cv2

from scipy.spatial.transform import Rotation

from pyk4a import PyK4A
from pyk4a.calibration import CalibrationType

from robot_controller import FrankaOSCController
from marker_detection import get_kinect_ir_frame, detect_aruco_markers, estimate_transformation


def main():
    cam_id = 0

    # create controller (disable tip offset like in collect_data.py)
    robot = FrankaOSCController(tip_offset=np.zeros(3))

    # send robot to home/reset (best-effort)
    # try:
    #     if hasattr(robot, "go_home"):
    #         robot.go_home()
    #     elif hasattr(robot, "goto_home"):
    #         robot.goto_home()
    #     else:
    #         robot.reset()
    # except Exception:
    #     try:
    #         home_joints = [-0.74921682, 0.13623207, 0.37435664, -2.00871515, -0.54053575, 2.19774203, 2.34971468]
    #         robot.reset(joint_positions=home_joints)
    #     except Exception as e:
    #         print("Failed to move robot to home:", e)
    #         return

    # time.sleep(1.0)

    # # small relative move to ensure a changed pose (optional)
    # try:
    #     delta_pos = [np.array([0.08, 0.28, -0.03])]  # down 3 cm
    #     delta_axis_angle = [np.array([0.0, 0.0, 0.0])]
    #     robot.move_by(delta_pos, delta_axis_angle, num_steps=400, num_additional_steps=0)
    # except Exception:
    #     pass

    # time.sleep(0.5)

    # Initialize Kinect and get calibration (use DEPTH intrinsics as in collect_data.py)
    k4a = PyK4A(device_id=cam_id)
    try:
        k4a.start()
    except Exception as e:
        print("Failed to start PyK4A:", e)
        return

    try:
        camera_matrix = k4a.calibration.get_camera_matrix(CalibrationType.DEPTH)
        dist_coeffs = k4a.calibration.get_distortion_coefficients(CalibrationType.DEPTH)
    except Exception:
        camera_matrix = None
        dist_coeffs = None

    last_color = None
    last_ir = None

    # warm up / drain a few frames and capture a few valid frames (use timeouts)
    for i in range(3):
        try:
            _ = k4a.get_capture(timeout_ms=200)
        except Exception:
            pass

    for i in range(5):
        try:
            cap = k4a.get_capture(timeout_ms=200)
        except TypeError:
            # older pyk4a may not support timeout_ms parameter
            cap = k4a.get_capture()
        except Exception:
            cap = None

        if cap is None:
            continue

        # color may be None depending on config
        color = getattr(cap, "color", None)
        # use helper for IR as in collect_data.py (returns single-channel image)
        ir = get_kinect_ir_frame(k4a)

        if color is not None:
            if color.ndim == 3 and color.shape[-1] == 4:
                color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
            last_color = color.copy()
        if ir is not None:
            if ir.dtype != np.uint8:
                ir_disp = cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                ir_disp = ir.copy()
            last_ir = ir_disp

    # stop camera promptly to avoid queue buildup during subsequent operations
    try:
        k4a.stop()
    except Exception:
        pass

    if last_color is None and last_ir is None:
        print("No frames captured.")
        return

    # Prefer IR for marker detection (as collect_data.py does)
    img_for_detection = last_ir if last_ir is not None else cv2.cvtColor(last_color, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids = detect_aruco_markers(img_for_detection, debug=False)

    marker_pose = None

    # Prepare visualization on the exact image used for detection (IR -> BGR)
    if img_for_detection.ndim == 2:
        vis_on_ir = cv2.cvtColor(img_for_detection, cv2.COLOR_GRAY2BGR)
    else:
        vis_on_ir = img_for_detection.copy()

    if ids is not None and len(ids) > 0:
        print(f"Detected {len(ids)} markers: ids={ids.flatten()}")

        # draw boxes on the detection image; handle normalized coords and different shapes
        for c in corners:
            pts = np.asarray(c).reshape(-1, 2).astype(np.float32)  # (4,2)
            # If coordinates look normalized (<=1), scale to image size
            if pts.max() <= 1.01:
                h_det, w_det = img_for_detection.shape[:2]
                pts[:, 0] = pts[:, 0] * w_det
                pts[:, 1] = pts[:, 1] * h_det
            # ensure integer pixel order (x,y)
            pts_i = np.int32(pts[:, :2])
            cv2.polylines(vis_on_ir, [pts_i], isClosed=True, color=(0, 255, 0), thickness=2)

        # estimate pose using the same function as collect_data.py
        if camera_matrix is not None and dist_coeffs is not None:
            transform_matrix = estimate_transformation(corners, ids, camera_matrix, dist_coeffs)
            if transform_matrix is not None:
                marker_pose = transform_matrix
                t = transform_matrix[:3, 3]
                R = transform_matrix[:3, :3]
                r = Rotation.from_matrix(R)
                quat = r.as_quat()  # x,y,z,w
                euler = r.as_euler("xyz", degrees=True)
                print("Marker pose (camera frame):")
                print(f" translation (m): {t}")
                print(f" quaternion (xyzw): {quat}")
                print(f" euler xyz (deg): {euler}")
                txt = f"t=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]"
                cv2.putText(vis_on_ir, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                txt2 = f"eul=[{euler[0]:.1f},{euler[1]:.1f},{euler[2]:.1f}]"
                cv2.putText(vis_on_ir, txt2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                print("estimate_transformation returned None.")
        else:
            print("No camera intrinsics available; cannot estimate pose.")
    else:
        print("No ArUco markers detected in the captured frame.")

    # Save IR, color and visualization-on-IR (do not display)
    os.makedirs("data", exist_ok=True)
    if last_color is not None:
        out_color = f"data/last_color_cam{cam_id}.png"
        cv2.imwrite(out_color, last_color)
        print("Saved color frame to", out_color)
    if last_ir is not None:
        out_ir = f"data/last_ir_cam{cam_id}.png"
        cv2.imwrite(out_ir, last_ir)
        print("Saved IR frame to", out_ir)

    vis_ir_out = f"data/last_vis_on_ir_cam{cam_id}.png"
    cv2.imwrite(vis_ir_out, vis_on_ir)
    print("Saved visualization on IR to", vis_ir_out)


if __name__ == "__main__":
    main()
