import numpy as np
from save_data import DataRecorder, cv2_dispay
from frankapy import FrankaArm
from frankapy import FrankaConstants as FC

from robomail.motion import GotoPoseLive
import UdpComms as U
import threading
from copy import deepcopy
from typing import List, Tuple, Dict
import time
from autolab_core.transformations import euler_matrix
import rospy
import cv2


SAVE_DIR = '/home/selamg/beadsight_stonehenge/data/ssd/full_dataset/'
FOLDER_NAME = 'run/'   

cv2_dispay.start()

class Teleoperation:
	def __init__(self, 
				 franka_IP, 
				 Oculus_IP, 
				 camera_numbers:List[int],
				 camera_sizes:List[Tuple[int, int]],
				 Hand_IP = None, 
				 portTX = 8000, 
				 portRX=8001, 
				 hand_portTX=8010, 
				 hand_portRX=8011, 
				 girpper_offset = [0, 0, 0], 
				 scaleing_factor = 1, #scaling factor to apply to motions
				 scaleing_center =[0.5, 0, 0], # location to scale about (m, in franka frame)
				 min_gripper_width = 0.03,   #0.007, for usb
				 record_data=True,
				 Hz = 10,
				 print_pos = False) -> None:
		
		# crate the communcation socket between franka and the oculus
		# This is a seperate function so that it can be overriden for testing.
		self.sock = self.oculus_socket(franka_IP, Oculus_IP, portTX, portRX)

		if Hand_IP is None:
			self.use_robo_hand = False
		else:
			self.hand_sock = U.UdpComms(udpIP=franka_IP, sendIP = Hand_IP, portTX=hand_portTX, portRX=hand_portRX, enableRX=True, suppressWarnings=True)
			self.use_robo_hand = True

		default_impedances = np.array(FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES)
		default_impedances[3:] = default_impedances[3:] # DONT # reduce the rotational stiffnesses, default in gotopose live
		new_impedances = np.copy(default_impedances)
		# new_impedances[:3] = 0.5*default_impedances[:3] # reduce the translational stiffnesses
		self.step_size = 0.05 # max step size for the pose controller
		self.PoseController = GotoPoseLive(cartesian_impedances=new_impedances.tolist(), step_size=self.step_size)
		self.fa = self.PoseController.fa
		self.new_object_list = [] 
		self.inventory_list = []
		self.detected_objects = {}
		self.running = False
		self.paused = False
		self.gripper_offset = girpper_offset # gripper offset for if using custom grippers.
		self.scaleing_factor = scaleing_factor # scaling factor to apply to motions
		self.scaleing_center = np.array(scaleing_center)
		if record_data:
			self.data_recorder = DataRecorder(SAVE_DIR + FOLDER_NAME, 
												camera_numbers=camera_numbers, 
												camera_sizes=camera_sizes,
												position_dim=7, 
												velocity_dim=7, 
												overwrite=False)
		self.record = False
		self.manual_control = False
		self.manual_pose = deepcopy(FC.HOME_POSE)
		# self.grasped = False
		self.message_index = 0
		self.min_gripper_width = min_gripper_width
		self.record_data = record_data
		self.Hz = Hz
		self.print_pos = print_pos

	@staticmethod
	def oculus_socket(franka_IP, Oculus_IP, portTX, portRX):
		# start the oculus communication thread
		return U.UdpComms(udpIP=franka_IP, sendIP = Oculus_IP, portTX=portTX, portRX=portRX, enableRX=True, suppressWarnings=True)	

	# Generates a message to be sent to the oculus when a new object is created.
	def new_object_message(self, new_object_list):
		# new_object_list: list of all new objects. Keys to the object dictionary.
		message = ""
		for new_object in new_object_list:
			object_ID = int(new_object)
			message += '_newItem\t' + self.detected_objects[object_ID].object_type \
			+ '\t' + str(new_object) + '\t' + self.detected_objects[object_ID].size \
			+ '\t' + self.detected_objects[object_ID].color + '\n'
		return message
	
	# generates a message to be sent to the oculus for an object in the object dictonary.
	def object_message(self, object_ID):
		# object_ID: ID of the object to be sent. int.
		pos = self.detected_objects[object_ID].position
		vel = self.detected_objects[object_ID].velocity
		rot = self.detected_objects[object_ID].rotation
		avel = self.detected_objects[object_ID].ang_velocity
		return str(object_ID) + '\t' + str(-pos[1]) + ',' + str(pos[2]) + ',' + str(pos[0]-0.6) + '\t' \
		+ str(-vel[1]) + ',' + str(vel[2]) + ',' + str(vel[0]) + '\t' \
		+ str(rot[1]) + ',' + str(-rot[2]) + ',' + str(-rot[0]) + ',' + str(rot[3]) + '\t' \
		+ str(avel[1]) + ',' + str(-avel[2]) + ',' + str(-avel[0])


	def oculus_communciation(self, current_position:np.ndarray, current_rot:np.ndarray, finger_width:float) -> Tuple[bool, Tuple[np.ndarray, np.ndarray, float]]:
		""""
		Sends the current state of the scene to the oculus, and recieves the goal position, rotation, and gripper width.
		:param current_position: Current position of the hand. (m, in franka frame)
		:param current_rot: Current rotation of the hand. (quaternion, in franka frame)
		:param finger_width: Current gripper width. (m)
		:return: communication success, (goal_position, goal_rotation, gripper_width)
		"""
		send_string = str(self.message_index) + '\n' # Begin the me>ssage to send to the oculus

		# add delete commands to the message
		for item in self.inventory_list:
			if not (int(item)) in self.detected_objects.keys():
				send_string += "_deleteItem" + '\t' + item + '\n'

		# Determine if any of the objects in the scene have not been sent to unity, and generate new object messages.
		for item in self.detected_objects.keys():
			if not(str(item) in self.inventory_list) and not(str(item) in self.new_object_list):
				self.new_object_list.append(str(item))
		if len(self.new_object_list) != 0: # There are new objects in the scene!
			send_string += self.new_object_message(self.new_object_list)
	
		# For each object in the scene, genearte an object message and add it to the send string.
		for game_object in self.detected_objects:
			send_string += self.object_message(game_object) + '\n'

		# Get the hand and gripper positions to send to the oculus.
		send_hand_position = current_position + self.gripper_offset
		# finger_width = self.fa.get_gripper_width()

		send_string += '_hand\t' + str(-send_hand_position[1]) + ',' + str(send_hand_position[2]) + ',' + str(send_hand_position[0]-0.6) +'\t'\
			+ str(current_rot[2]) + ',' + str(-current_rot[3]) + ',' + str(-current_rot[1]) + ',' + str(current_rot[0]) + '\t' + str(finger_width)

		self.sock.SendData(send_string) # Send this string to other application
		self.message_index += 1

		data = self.sock.ReadReceivedData() # read data

		if data != None: # if NEW data has been received since last ReadReceivedData function call
			inventory, unknown_objects, ee_pose, gripper_data = data.split('\n')
			gripper_data = gripper_data[:-1] # remove unnessesary tab
			self.inventory_list = inventory.split('\t')[1:]
			self.new_object_list = unknown_objects.split('\t')[1:]
			
			# Extract the goal position, rotation, and width from the oculus message.
			goal_position, goal_rotation = ee_pose.split('\t')
			goal_position = np.array(goal_position[1:-1].split(', ')).astype(np.float64)
			goal_position = np.array([goal_position[2] + 0.6, -goal_position[0], goal_position[1] + 0.02])
			goal_position -= self.gripper_offset
			goal_rotation = np.array(goal_rotation[1:-1].split(', ')).astype(np.float64)
			goal_rotation = np.array([goal_rotation[3], -goal_rotation[2], goal_rotation[0], -goal_rotation[1]])

			# adjust the goal width
			if not self.use_robo_hand:
				gripper_data = float(gripper_data)*2

			return True, (goal_position, goal_rotation, gripper_data)

		else:
			return False, (None, None, None)


	def run(self):
		print('running teleoperation')
		goal_pose = self.fa.get_pose() # pose to move to. Default to current pose.
		last_gripper_width = self.fa.get_gripper_width()
		rate = rospy.Rate(self.Hz)
		while self.running:
			robo_data = self.fa.get_robot_state()
			current_pose = robo_data['pose']*self.fa._tool_delta_pose
			cur_joints = robo_data['joints']
			cur_vel = robo_data['joint_velocities']
			finger_width = fa.get_gripper_width()
			jacobian = self.fa.get_jacobian(cur_joints)

			# current_pose = self.fa.get_pose() # get the current pose of the hand.
			current_position = current_pose.translation
			current_rot = current_pose.quaternion

			# communicate with the oculus, and get the goal position, rotation, and gripper width.
			recieve_success, recieved_data = self.oculus_communciation(current_position, current_rot, finger_width) 		
			
			if recieve_success:
				goal_position, goal_rotation, gripper_data = recieved_data

				# print('--------------------------------')
				# print('goal_position: ', goal_position)
				# print('goal_rotation: ', goal_rotation)
				# print('gripper_data: ', gripper_data)
				
				# adjust the goal position by the scaling factor
				goal_position = (goal_position - self.scaleing_center)*self.scaleing_factor + self.scaleing_center

				# print(gripper_data, type(gripper_data))
				# print(self.use_robo_hand)
				if self.use_robo_hand:
					goal_width = 0
					self.hand_sock.SendData(gripper_data)
				else:
					goal_width = gripper_data

				# Hardcode rotation to none:
				goal_rotation = deepcopy(FC.HOME_POSE.quaternion)


				# The pose controller clips the goal position so that the 
				# move command doesn't cause the robot to move too much in one 
				# step. Clip the goal position before sending it to the pose 
				# controller to avoid this and make the saved action more accurate.
				delta_motion = goal_position - current_pose.translation
				if (np.linalg.norm(delta_motion) > self.step_size):
					goal_position = (delta_motion/np.linalg.norm(delta_motion))*self.step_size + current_pose.translation

				# set the goal_pose rotation and translation to the goal values
				goal_rotation_mat = goal_pose.rotation_from_quaternion(goal_rotation)
				goal_pose.rotation = goal_rotation_mat
				goal_pose.translation = goal_position


				# print('goal translation', goal_pose.translation)

				if self.manual_control:
					goal_pose = deepcopy(self.manual_pose)
				
				if self.print_pos:
					print("starting position:", np.round(current_pose.translation, 3))

				# print('goal_pose: ', goal_pose.translation)

				 # save the gripper data
				current_relative_rotation = (current_pose*FC.HOME_POSE.inverse()).euler_angles
				goal_relative_rotation = (goal_pose*FC.HOME_POSE.inverse()).euler_angles
				current_velocity_arm = jacobian@cur_vel # get the current velocity of the arm using the jacobian
				

				current_pose_info = np.concatenate((current_pose.translation, current_relative_rotation, np.array([finger_width])))
				goal_pose_info = np.concatenate((goal_pose.translation, goal_relative_rotation, np.array([goal_width])))
				velocity_info = np.concatenate((current_velocity_arm, np.array([0])))

				if self.record:
					self.data_recorder.record_data(current_pose_info, goal_pose_info, velocity_info)

				# Send the goal goal_pose to the goal_pose controller
				# print('send to pose controller position', goal_pose.translation)
					
				self.PoseController.step(goal_pose, current_pose, ros_sleep=False) # step the pose controller to the goal pose
				
				# print("goal_width: ", goal_width)
				# Move the gripper, clip to make sure it is within range, and doesn't get stuck on the usb.

				if goal_width <= self.min_gripper_width:
					if last_gripper_width == self.min_gripper_width:
						pass
					else:
						self.fa.goto_gripper(self.min_gripper_width, block=False, speed=0.15, force = 10)
						last_gripper_width = self.min_gripper_width
				else:
					self.fa.goto_gripper(goal_width, block=False, speed=0.15, force = 10)
					last_gripper_width = goal_width

				# Move the gripper.
				# if goal_width < 0.01: # if the grasp is less than 1cm, use grasp mode. 
				# 	if self.grasped and goal_width >= finger_width:
				# 		pass # do nothing, since the object is already grapsed
				# 	else:
				# 		self.grasped = True
				# 		self.fa.goto_gripper(goal_width, grasp = True, block=False, speed=0.15, force = 60) # , force = 10)
				# 		print('grasp', goal_width)
				# else:
				# 	self.grasped = False
				# 	self.fa.goto_gripper(goal_width, block=False, speed=0.15, force = 60) # , force = 10)
				# 	print('move', goal_width)
				rate.sleep() # sleep for the rest of the time to keep the loop at the desired frequency
					
				


	def start(self):
		# start run() in a new thread
		# self.PoseController.start()
		self.running = True
		self.thread = threading.Thread(target=self.run, daemon=True)
		self.thread.start()

	def stop(self):
		# stop runing the thread by setting the 'running' flag to false, then waiting for 
		# the tread to terminate. Finally, stop any ongoing skill.
		# self.PoseController.stop()
		self.data_recorder.close()
		self.running = False
		self.data_recorder #TODO: what is this lol 
		self.thread.join()
		print('Stoped Teleoperation')

	# sets the detected object dictionary.
	def set_detected_objects(self, detected_objects):
		# detected_objects: Dictionary (key: int, value: Object) of objects found in the scene.
		self.detected_objects = detected_objects

	def save_epoch(self, epoch_number):
		if self.record_data:
			# self.data_recorder.write_to_file(epoch_number)
			self.pause_recording()
			time.sleep(0.2) # sleep to make sure the data recorder has paused
			self.data_recorder.write_to_file()

	def clear_data(self):
		if self.record_data:
			self.pause_recording()
			time.sleep(0.2)
			self.data_recorder.reset_episode(delete_last_episode=True)
	
	def start_recording(self):
		if self.record_data:
			self.record = True
	
	def pause_recording(self):
		if self.record_data:
			self.record = False

	def set_manual_control(self, manual_pose):
		self.manual_pose = manual_pose
		self.manual_control = True		

	def pause(self):
		"""
		Stops the teleoperation thread, but does not stop the pose controller.
		The pose controller will continue to run, but the goal pose will not be updated.
		"""
		if self.paused:
			print('Teleoperation is already paused. Pause will pass')
			return
		
		self.running = False
		self.paused = True
		self.thread.join()
		print('Stoped Teleoperation')

	def resume(self):
		"""
		Resumes the teleoperation thread.
		"""
		if not self.paused:
			print('Teleoperation is not paused. Resume will pass')
			return
		
		self.running = True
		self.paused = False
		self.thread = threading.Thread(target=self.run, daemon=True)
		print('start thread')
		self.thread.start()
	
	def end_manual_control(self):
		self.manual_control = False
		self.manual_pose = deepcopy(FC.HOME_POSE)

# create a fake teleoperation class for testing. Inherit from the real teleoperation class.
# need to only change the oculus_communication function and the oculus_socket function.
class FakeTeleoperation(Teleoperation):
	def __init__(self, 
			  	 camera_numbers,
				 camera_sizes,
				 franka_IP = None, 
				 Oculus_IP = None, 
				 Hand_IP = None, 
				 portTX = 8000, 
				 portRX=8001, 
				 hand_portTX=8010, 
				 hand_portRX=8011, 
				 girpper_offset = [0, 0, 0], 
				 scaleing_factor = 1, #scaling factor to apply to motions
				 scaleing_center =[0.5, 0, 0], # location to scale about (m, in franka frame)
				 ) -> None:
		"""
		Fake teleoperation class for testing. Inherits from the real teleoperation class.
		None of the inputs are used, and the oculus_communication function is overriden.
		"""

		# call the init function of the parent class
		super().__init__(franka_IP, Oculus_IP, camera_numbers, camera_sizes)
		self.command_pose = deepcopy(FC.HOME_POSE)
		self.command_width = 0.04

		# flag to use a predefined motion
		self.use_predefined_motion = False

	@staticmethod
	def oculus_socket(franka_IP, Oculus_IP, portTX, portRX):
		# return a fake socket object
		return None
	
	def set_command_pose(self, command_pose=None, command_width=None):
		self.use_predefined_motion = False # Don't use a predefined motion
		# set the command pose and width
		if command_pose is not None:
			self.command_pose = command_pose
		if command_width is not None:
			self.command_width = command_width

	def set_command_pose_info(self, command_position=None, command_rotation=None, command_width=None):
		self.use_predefined_motion = False # Don't use a predefined motion
		# set the command pose and width
		if command_position is not None:
			self.command_pose.translation = command_position
		if command_rotation is not None:
			self.command_pose.rotation = command_rotation
		if command_width is not None:
			self.command_width = command_width

	def oculus_communciation(self, current_position:np.ndarray, current_rot:np.ndarray, finger_width:float) -> Tuple[bool, Tuple[np.ndarray, np.ndarray, float]]:
		""""
		Returns the current command pose and width.
		:return: communication success, (goal_position, goal_rotation, gripper_width)
		"""
		if self.use_predefined_motion:
			return True, self.predefined_motion()
		return True, (self.command_pose.translation, self.command_pose.rotation, self.command_width)				

	def set_predefined_motion(self, motion_function):
		"""
		Sets a predefined motion function to be used instead of the oculus communication.
		:param motion_function: function to be called to get the next pose and width. Should return (position, width, gripper_width)
		"""
		# set the flag to use a predefined motion
		self.use_predefined_motion = True
		self.predefined_motion = motion_function

class SinusoidalMotion:
	def __init__(self, amplitude, frequency, offset, phase_shift = None):
		"""
		Generates a sinusoidal motion.
		:param amplitude: amplitude of the sinusoid, either [x, y, z], [x, y, z, g], or [x, y, z, r, p, y, g]
		:param frequency: frequency of the sinusoid, either [x, y, z], [x, y, z, g], or [x, y, z, r, p, y, g]
		:param offset: offset of the sinusoid, either [x, y, z], [x, y, z, g], or [x, y, z, r, p, y, g]
		:param phase_shift: phase of the sinusoid, either [x, y, z], [x, y, z, g], or [x, y, z, r, p, y, g]
		"""
		# check that the input is valid
		if phase_shift is None:
			self.phase_shift = np.zeros(len(amplitude))
		else:
			self.phase_shift = phase_shift
		
		# verify that the input is valid
		assert len(amplitude) == len(frequency) == len(offset) == len(self.phase_shift), "Amplitude, frequency, offset, and phase_shift must all have the same length"
		assert len(amplitude) in [3, 4, 7], "Amplitude, frequency, offset, and phase_shift must have length 3, 4, or 7"

		self.amplitude = amplitude
		self.frequency = frequency
		
		self.offset = offset
		self.start_time = time.time()
	
	def __call__(self):
		"""
		Returns the next pose and gripper width from the sinusoidal motion.
		:return: position, rotation, gripper_width
		"""
		time_elapsed = time.time() - self.start_time
		sine_value = self.amplitude*np.sin(2*np.pi*self.frequency*time_elapsed + self.phase_shift) + self.offset
		
		position = sine_value[:3]
		rotation = np.copy(FC.HOME_POSE.rotation)

		if len(sine_value) == 3:
			width = 0.04
		elif len(sine_value) == 4:
			width = sine_value[3]
		elif len(sine_value) == 7:
			# rotate the pose by the rotation matrix specified by the euler angles
			rotation = euler_matrix(sine_value[3], sine_value[4], sine_value[5])[:3, :3]@rotation
			width = sine_value[6]
		else:
			raise ValueError("Invalid sine value length")

		return position, rotation, width

FRANKA_IP = "172.26.89.208"
OCULUS_IP = "172.26.24.38"
cam_nums = [1, 2, 3, 4, 5, 6]
cam_sizes = [(1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (800, 1280)]


if __name__ == "__main__":
	# with RemoteInputHost() as remote_io:
	# Reset Franka
	FAKE = False # Set to True if the oculus isn't being used (testing).
	fa = FrankaArm()
	fa.reset_joints()
	fa.close_gripper()
	fa.goto_gripper(0.04)

	# Initalize the VR controller:
	if FAKE:
		TeleopController = FakeTeleoperation(camera_numbers = cam_nums,
											camera_sizes = cam_sizes)
		motion = SinusoidalMotion(amplitude=np.array([0.1, 0.1, 0.1]), frequency=np.array([0.085, 0.1, 0.115]), offset=np.array([0.5, 0, 0.3]))
		TeleopController.set_predefined_motion(motion)
	else:
		TeleopController = Teleoperation(franka_IP = FRANKA_IP, 
											camera_numbers = cam_nums,
											camera_sizes = cam_sizes,
											Oculus_IP = OCULUS_IP, 
											scaleing_factor=1, 
											scaleing_center=[0.5, 0, 0], 
											record_data=True)

	TeleopController.set_manual_control(FC.HOME_POSE)
	TeleopController.start()

	reset_pose = deepcopy(FC.HOME_POSE)
	reset_pose.translation = np.array([0.6, 0, 0.4])

	MAX_EPOCHS = 100
	for i in range(MAX_EPOCHS):
		TeleopController.start_recording() # creates useless files, so that we can render the cameras.
		if input("Press enter to take control, or q to quit") == 'q':
			break
		TeleopController.end_manual_control()
		TeleopController.print_pos = False
		if input("Press enter to start recording, or q to quit") == 'q':
			break
		TeleopController.print_pos = False
		TeleopController.clear_data()
		TeleopController.resume()
		TeleopController.start_recording()
		input("Press enter to end run")
		print('pausing')
		TeleopController.pause_recording()
		TeleopController.pause()
		break_loop = False
		while True:
			in_message = input("Press enter to reset the arm and begin saving, q to quit, or d to not save this run")
			if in_message == 'q':
				TeleopController.stop()
				break_loop = True
				break
			elif in_message == 'd':
				save_epoch = False
				break
			elif in_message == '':
				save_epoch = True
				break

		if break_loop:
			break

		TeleopController.set_manual_control(reset_pose)
		TeleopController.fa.open_gripper()
		TeleopController.resume()

		print('rest holder: ', np.random.randint(0, 11, size=2))
		print('reset portion: ', np.random.randint(0, 11, size=2))
		if save_epoch:
			print('Saving run ' + str(i) + " RESET SCEENE")
			TeleopController.save_epoch(i)
		else:
			print('Not saving run ' + str(i) + " RESET SCEENE")
			TeleopController.clear_data()
		
		

	input("Press enter to kill teleoperation")
	print('Killing Teleop')

	TeleopController.clear_data()
	TeleopController.stop()
	cv2_dispay.stop()
	cv2_dispay.join()

	print('stopped')
