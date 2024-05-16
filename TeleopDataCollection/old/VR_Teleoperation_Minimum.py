import numpy as np

from frankapy import FrankaArm

import time

from Teleoperation_Data_Collection import Teleoperation

FRANKA_IP = "172.26.91.11"
OCULUS_IP = "172.26.33.175"

if __name__ == "__main__":
	# Reset Franka
	USE_ROBOHAND = False # Set to True if the 5 fingered hand is used.
	fa = FrankaArm()
	fa.reset_joints()
	fa.close_gripper()
	fa.goto_gripper(0.04)

	# Initalize the VR controller:
	if USE_ROBOHAND:
		TeleopController = Teleoperation(franka_IP = FRANKA_IP, Oculus_IP = OCULUS_IP, Hand_IP="172.26.50.162")
	else:
		TeleopController = Teleoperation(franka_IP = FRANKA_IP, Oculus_IP = OCULUS_IP)
	TeleopController.start()


	input("Press enter to kill teleoperation")
	print('Killing Teleop')
	TeleopController.stop()
