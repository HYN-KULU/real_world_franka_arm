import numpy as np
import pyrobotiqgripper as rq
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig

config_path = "configs"

robot_interface = FrankaInterface(
    f"{config_path}/charmander.yml",
    use_visualizer=False
)

controller_type = "JOINT_POSITION"
controller_cfg = YamlConfig(
    f"{config_path}/joint-position-controller.yml"
).as_easydict()

gripper = rq.RobotiqGripper()
gripper.activate()
gripper.calibrate(closemm=0, openmm=40)
gripper.open()

# These are home joints:
target_joint_positions = [
    -1.3159,
    -0.4246,
     0.1067,
    -2.7110,
    -0.0562,
     2.3219,
     0.7518,
]

action = target_joint_positions + [0.0]    # Franka gripper byte unused; Robotiq driven separately

while True:

    if len(robot_interface._state_buffer) > 0:

        if np.max(np.abs(np.array(robot_interface._state_buffer[-1].q) - np.array(target_joint_positions))) < 2e-3:
            break

    robot_interface.control(
        controller_type=controller_type,
        action=action,
        controller_cfg=controller_cfg,
    )

gripper.close()

robot_interface.close()
