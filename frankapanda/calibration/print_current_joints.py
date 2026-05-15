"""
Print the current Franka joint positions without commanding robot motion.
"""
import argparse
import time

import numpy as np

from robot_controller import FrankaOSCController


def wait_for_robot_state(robot, timeout_s):
    start_time = time.time()
    while robot.robot_interface.state_buffer_size == 0:
        if time.time() - start_time > timeout_s:
            raise TimeoutError("Timed out waiting for robot state.")
        print("Waiting for robot state...")
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--interface-cfg", default="configs/charmander.yml")
    args = parser.parse_args()

    robot = FrankaOSCController(
        interface_cfg=args.interface_cfg,
        controller_cfg=None,
        tip_offset=np.zeros(3),
    )
    wait_for_robot_state(robot, args.timeout)

    joints = np.asarray(robot.joint_positions, dtype=np.float64)
    print("Current robot joints:")
    print(joints)
    print("As Python list:")
    print(joints.tolist())


if __name__ == "__main__":
    main()
