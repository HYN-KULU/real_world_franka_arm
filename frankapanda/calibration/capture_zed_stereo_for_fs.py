from pathlib import Path
import argparse
import cv2
import numpy as np


def get_zed_calib(zed):
    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    left = calib.left_cam

    K = np.array([
        [left.fx, 0.0, left.cx],
        [0.0, left.fy, left.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    baseline_m = None

    if hasattr(calib, "stereo_transform"):
        st = calib.stereo_transform
        if hasattr(st, "get_translation"):
            t = st.get_translation()
            if hasattr(t, "get"):
                baseline_m = abs(float(t.get()[0]))
            elif hasattr(t, "x"):
                baseline_m = abs(float(t.x))
            else:
                baseline_m = abs(float(t[0]))

    if baseline_m is None:
        raise RuntimeError("Could not read ZED baseline")

    return K, baseline_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/fs_test")
    parser.add_argument("--serial-number", type=int, default=None)
    parser.add_argument("--camera-id", type=int, default=None)
    parser.add_argument("--resolution", type=str, default="HD720")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()

    import pyzed.sl as sl

    resolution_map = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1200": sl.RESOLUTION.HD1200,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "SVGA": sl.RESOLUTION.SVGA,
        "VGA": sl.RESOLUTION.VGA,
        "AUTO": sl.RESOLUTION.AUTO,
    }

    res_key = args.resolution.upper()
    if res_key not in resolution_map:
        raise ValueError(
            f"Unsupported resolution {args.resolution}. "
            f"Choices: {list(resolution_map.keys())}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = resolution_map[res_key]
    init.camera_fps = args.fps
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode = sl.DEPTH_MODE.NONE
    # if args.serial_number is not None:
    #     print("Opening serial:", args.serial_number)
    #     init.set_from_serial_number(args.serial_number)
    # elif args.camera_id is not None:
    #     print("Opening camera id:", args.camera_id)
    #     init.input.set_from_camera_id(args.camera_id)
    # else:
    #     print("Opening default camera")

    #     print("Using resolution:", res_key)
    #     print("Using fps:", args.fps)

    input_type = sl.InputType()

    if args.serial_number is not None:
        print("Opening serial:", args.serial_number)
        input_type.set_from_serial_number(args.serial_number)

    init.input = input_type

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED: {status}")

    cam_info = zed.get_camera_information()
    print("Opened model:", cam_info.camera_model)
    print("Opened serial:", cam_info.serial_number)

    runtime = sl.RuntimeParameters()
    left_mat = sl.Mat()
    right_mat = sl.Mat()

    K, baseline_m = get_zed_calib(zed)

    left = None
    right = None

    for _ in range(args.frames):
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT)

            left_bgra = left_mat.get_data()
            right_bgra = right_mat.get_data()

            left = left_bgra[:, :, :3].copy()
            right = right_bgra[:, :, :3].copy()

    zed.close()

    if left is None or right is None:
        raise RuntimeError("Failed to capture stereo pair")

    cv2.imwrite(str(out_dir / "left.png"), left)
    cv2.imwrite(str(out_dir / "right.png"), right)

    with open(out_dir / "K.txt", "w") as f:
        f.write(
            f"{K[0,0]} {K[0,1]} {K[0,2]} "
            f"{K[1,0]} {K[1,1]} {K[1,2]} "
            f"{K[2,0]} {K[2,1]} {K[2,2]}\n"
        )
        f.write(f"{baseline_m}\n")

    print("Saved:")
    print(out_dir / "left.png")
    print(out_dir / "right.png")
    print(out_dir / "K.txt")
    print("baseline_m:", baseline_m)


if __name__ == "__main__":
    main()