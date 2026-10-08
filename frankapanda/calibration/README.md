# Camera Calibration

This folder holds the tools for finding where each camera sits relative to the Franka robot. Two setups are supported:

| Setup | Where the camera is | Where the ArUco marker is | Result |
|---|---|---|---|
| **Eye-to-hand** (side camera) | Fixed on a stand next to the robot | Mounted on the gripper | `T_base_cam`: camera pose in the robot base frame |
| **Eye-in-hand** (wrist camera) | Mounted on the gripper / wrist | Fixed on the table | `T_ee_cam`: camera pose in the end-effector frame |

All commands below are run from the **repository root** (`real_world_franka_arm/`), because the scripts use relative paths such as `configs/...` and `data/...`.

---

## 0. Before you start: measure the marker and set its size

> [!IMPORTANT]
> **Measure the printed ArUco marker and enter its side length in the code before you collect any data.** The pose estimate scales directly with this number. If it is wrong, every translation is wrong by the same factor and the calibration will be off, even though detection still looks fine.

1. Print the marker. `aruco-581.svg` in this folder is ID 581 from `DICT_ARUCO_ORIGINAL`, the dictionary used by the detector. Turn off "fit to page" or any scaling when printing.
2. With a ruler or calipers, measure the **side length of the black square only**. Do not include the white border. Use meters.
3. Put that value in `marker_size` inside `estimate_transformation()` in [marker_detection.py](marker_detection.py):

   ```python
   # marker_detection.py, estimate_transformation()
   marker_size = 0.042  # In meters  <-- replace with YOUR measured side length
   ```

   Every collection script (Kinect, ZED side camera, and ZED wrist camera) uses this one function, so this is the only place to change it.
4. Mount the marker flat and rigidly. If it bends or slides, the calibration will be noisy.

**Eye-to-hand only:** the solver also needs to know where the marker sits on the gripper. The offsets in `estimate_tag_pose()` in [solve_calibration.py](solve_calibration.py) assume a marker that is 0.055 m wide (`0.0275 = 0.055 / 2` moves from the marker corner to its center). If your marker or its mount is different, measure again and update `t_tag_to_hand` (see step 1.3).

### Quick check

Before collecting data, run the single-frame test on a ZED camera:

```bash
python frankapanda/calibration/test_franka_zed.py --serial-number <ZED_SERIAL> --debug
```

It detects the marker, prints its pose, and saves `data/last_zed_vis_on_rgb_<serial>.png`. Put the marker at a known distance from the camera, for example 0.50 m measured with a tape. The printed `z` should match that distance to within about 1 cm. If it is consistently too large or too small, `marker_size` is wrong.

### Finding reset joint positions

Both collection procedures start from a hand-chosen robot configuration. Use the Franka Desk to put the arm in guiding mode, move it by hand to the pose you want, and then print the joints:

```bash
python frankapanda/calibration/print_current_joints.py
```

Copy the printed Python list into the collection script, as described below.

---

## 1. Eye-to-hand calibration (fixed side camera)

The marker is on the gripper. The robot moves the gripper to random poses around a reset pose while the fixed camera watches. Each sample stores `(gripper_pose_in_base, marker_pose_in_camera)`. The solver then fits the rigid transform between marker positions in the camera frame and the same positions in the base frame.

### 1.1 Mount the marker on the gripper

Attach the marker rigidly to the gripper, facing the camera you are calibrating.

### 1.2 Collect data

1. For **each camera**, move the arm so the gripper marker is near the **center** of that camera's image and clearly visible. Record the joints with `print_current_joints.py`.
2. Add those joints to the `initial_joint_positions` dict in `main()`, keyed by `cam_id`, and set `cam_id`:
   - Azure Kinect (IR image): [collect_data.py](collect_data.py)
   - ZED: [collect_data_zed.py](collect_data_zed.py)
3. If needed, change the random sampling range in `move_robot_and_record_data()` (`random_delta_pos`, `random_delta_axis_angle`). The marker must stay in view for every sample. Wider ranges give a better-conditioned solution.
4. Run the script:

   ```bash
   python frankapanda/calibration/collect_data.py          # Kinect
   # or
   python frankapanda/calibration/collect_data_zed.py      # ZED
   ```

   Data is saved to `data/cam<cam_id>_data.pkl`. Aim for about 50 movements. Samples where the marker is not detected are dropped.

### 1.3 Solve

1. In [solve_calibration.py](solve_calibration.py), check `estimate_tag_pose()`:
   - `finger_to_hand`: offset from the Franka hand flange to the fingertip frame. The 0.1034 m value comes from the Franka Hand manual.
   - `t_tag_to_hand`: translation from the hand frame to the marker **center**. Measure it on your mount. The second line adds half the marker width (see step 0).
   - `R_tag_to_hand`: rotation of the marker relative to the hand.
2. Set `cam_id` in `__main__` and run:

   ```bash
   python frankapanda/calibration/solve_calibration.py
   ```

   This prints `T` (camera to base) and the **average reprojection error** in meters, and saves `data/calibration_results/cam<cam_id>_calibration.npz` (key `T`).

   A good result has an error of a few millimeters. If the error is large, check `marker_size` first, then `t_tag_to_hand`.

3. Optional: if you are unsure of the marker offset on the gripper, [solve_calibration_grid_search.py](solve_calibration_grid_search.py) searches over `t_tag_to_hand` and keeps the value with the lowest reprojection error.

---

## 2. Eye-in-hand calibration (wrist ZED camera)

The marker is fixed to the table and does not move. The wrist camera moves with the gripper. Each sample stores the gripper pose in the base frame and the marker pose in the camera frame. The unknown is the fixed transform from the end effector to the camera, which is a classic `AX = XB` hand-eye problem.

### 2.1 Fix the marker on the table

Tape the marker flat to the table, roughly below the robot's working pose. **It must not move during collection.** If it moves, the data is invalid.

### 2.2 Choose the reset pose

The automatic script resets to a hard-coded Cartesian pose before each sample. In `collect_auto_data()` in [eye_on_hand_calib_auto.py](eye_on_hand_calib_auto.py), set:

```python
reset_pos  = np.array([0.485, -0.04095, 0.3])          # eef position in base frame (m)
reset_quat = np.array([...])                            # eef orientation, xyzw
```

Pick a pose where the marker is near the center of the wrist image. One way to find it is to guide the arm there by hand and read `robot.eef_pose`.

### 2.3 Collect data

Automatic: random poses around the reset pose.

```bash
python frankapanda/calibration/eye_on_hand_calib_auto.py \
    --zed-serial-number <WRIST_ZED_SERIAL> \
    --num-movements 30 \
    --pos-range 0.03 \
    --rot-range 0.9
```

- `--pos-range` (m) and `--rot-range` (rad, axis-angle) set how far each sample moves from the reset pose. **Rotation diversity matters most for eye-in-hand.** Use as much rotation as you can while the marker stays in view. Poses where the IK target is not reached, or where the marker is not detected, are skipped and sampled again.
- Output is saved to `data/eye_on_hand_wrist_zed_auto_<cam_id>_data_larger_range.pkl`. Each entry is a dict with `gripper_pose` (4x4, eef in base), `marker_pose` (4x4, marker in camera), the detected `corners`/`ids`, the ZED `camera_matrix`/`dist_coeffs`, and the marker size used.
- Debug images for each sample are saved to `data/eye_on_hand_wrist_zed_auto_<cam_id>_<idx>_{rgb,vis_on_rgb}.png`. Check a few of them to confirm the detections are correct.

Manual alternative: move the arm by hand in guiding mode and press Enter to capture each sample. This was `manually_record_data()` in `eye_on_hand_calib.py`. Its output is a list of `(gripper_pose, marker_pose)` tuples.

> [!NOTE]
> `eye_on_hand_calib_auto.py` imports the ZED helper functions (`open_zed_camera`, `get_zed_left_calib`, `read_zed_rgb`, `detect_zed_marker_pose`) from `eye_on_hand_calib.py`. That module must exist next to it, and `detect_zed_marker_pose` must return a dict (see the fields above).

### 2.4 Solve

Use OpenCV's hand-eye solver. For eye-in-hand, pass the gripper poses **as-is**, meaning gripper to base. Do **not** invert them as you would for eye-to-hand. `solve_hand_eye_calibration(path, eye_to_hand=False)` in [evaluate_calibration.py](evaluate_calibration.py) does this for the tuple format. For the dict format produced by the auto script:

```python
import pickle, numpy as np, cv2

data = pickle.load(open("data/eye_on_hand_wrist_zed_auto_0_data_larger_range.pkl", "rb"))
G = [d["gripper_pose"] for d in data]   # T_base_ee
M = [d["marker_pose"]  for d in data]   # T_cam_marker

R, t = cv2.calibrateHandEye(
    R_gripper2base=[g[:3, :3] for g in G], t_gripper2base=[g[:3, 3] for g in G],
    R_target2cam=[m[:3, :3] for m in M],  t_target2cam=[m[:3, 3] for m in M],
    method=cv2.CALIB_HAND_EYE_TSAI,
)
T_ee_cam = np.eye(4); T_ee_cam[:3, :3] = R; T_ee_cam[:3, 3] = t.ravel()

# Sanity check: the marker is static, so T_base_marker should be the same for every sample.
P = np.array([(g @ T_ee_cam @ m)[:3, 3] for g, m in zip(G, M)])
print(T_ee_cam)
print("marker position spread (m):", np.linalg.norm(P - P.mean(0), axis=1).mean())
```

A good calibration gives a marker position spread of a few millimeters. If it is large, check `marker_size` and look for bad detections in the debug images. It can also help to compare methods (`CALIB_HAND_EYE_PARK`, `_DANIILIDIS`, ...).

To get the camera pose in the base frame at any moment, use `T_base_cam = robot.eef_pose @ T_ee_cam`.

---

## 3. Multi-camera alignment (optional)

After each fixed camera is calibrated to the base, small residual errors can remain between cameras. [align_multiple_cameras.py](align_multiple_cameras.py) refines the extrinsics by running ICP between the camera point clouds. Put several objects with distinct shapes in the shared workspace first, so ICP has enough geometry to lock onto.

---

## Troubleshooting

- **Translations off by a constant scale:** `marker_size` is wrong. See step 0.
- **No markers detected:** wrong dictionary (it must be `DICT_ARUCO_ORIGINAL`), the marker is too small in the image, there is glare, or the Kinect IR image is saturated.
- **Large error that does not improve with more data:** the marker moved or flexed during collection, or (for eye-to-hand) `t_tag_to_hand` / `R_tag_to_hand` are wrong.
- **Eye-in-hand result is unstable:** there is not enough rotation between samples. Increase `--rot-range`.
