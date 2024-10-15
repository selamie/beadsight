import numpy as np
from torchvision import transforms
from process_data_cage import MASK_VERTICIES, CROP_PARAMS, make_masks
from typing import Dict, List, Tuple
import cv2
import torch
import os
from dataset import NormalizeDiffusionActionQpos
from HardwareTeleop.multiprocessed_cameras import SaveVideosMultiprocessed
import shutil

BEAD_HORIZON = 5
#from rospy import Rate
from copy import deepcopy
import time
import json
# from rospy import Rate

def monitor_cameras(frames: Dict[str, np.ndarray]): #, gelsight_frame: np.ndarray = None):
    print('show cams')
    out_size = (900, 1800, 3)
    
    n_col = int(np.ceil(np.sqrt(len(frames)))) # 3
    n_row = int(np.ceil(len(frames) / n_col))

    # Create a grid of images
    tile_size = (int(out_size[0]/n_row), int(out_size[1]/n_col))
    grid = np.zeros(out_size, dtype=np.uint8)

    running_idx = 0
    for i, (name, frame) in enumerate(frames.items()):
        if not 'beadsight' in name:
            frame = frame[CROP_PARAMS[name]['i']:CROP_PARAMS[name]['i']+CROP_PARAMS[name]['h'], 
                        CROP_PARAMS[name]['j']:CROP_PARAMS[name]['j']+CROP_PARAMS[name]['w']]
        if name == '6': # rotate the 6th camera
            frame = np.rot90(frame).copy()
        row = i // n_col
        col = i % n_col
        scale_factor = min(tile_size[0]/frame.shape[0], tile_size[1]/frame.shape[1])
        frame = cv2.resize(frame.copy(), (0, 0), fx=scale_factor, fy=scale_factor)
        # if row == n_row - 1 and gelsight_frame is not None: # sqeeze the gelsight into the bottom row
        #     grid[row*tile_size[0]:row*tile_size[0]+frame.shape[0], 
        #         running_idx:running_idx+frame.shape[1]] = frame
        #     running_idx += frame.shape[1]
        grid[row*tile_size[0]:row*tile_size[0]+frame.shape[0], 
            col*tile_size[1]:col*tile_size[1]+frame.shape[1]] = frame
    
    
    cv2.imshow('cameras', grid)
    cv2.waitKey(1)

image_size = (400, 480)

class PreprocessData:
    """Preprocesses the data for the ACT model. Behaves like the dataset class 
    used in the training loop, but does not inherit from torch.utils.data.Dataset."""

    def __init__(self, norm_stats, camera_names):
        self.image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        # self.normalizer = NormalizeActionQpos(norm_stats)
        self.normalizer = NormalizeDiffusionActionQpos(norm_stats)
        self.masks = make_masks(image_size=image_size, verticies=MASK_VERTICIES)
        self.camera_names = camera_names

    def process_data(self, 
                     images: Dict[str, np.ndarray], 
                     beadsight_frames: List[np.ndarray], 
                     qpos: np.ndarray) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        
        normed_bead = []
        for image in beadsight_frames:
            image = torch.tensor(image, dtype=torch.float32)/255.0
            image = torch.einsum('h w c -> c h w', image) # change to c h w
            image = self.image_normalize(image)
            normed_bead.append(image)

        if normed_bead != []:
            beadcat = torch.concat(normed_bead,axis=0)
            beadcat = beadcat.unsqueeze(0)
        else:
            beadcat = None

        all_images = []
        for cam_name in self.camera_names:

            if cam_name == "beadsight" and beadcat != None:
                all_images.append(beadcat)

            elif cam_name in images.keys():
                # crop the images
                crop = CROP_PARAMS[cam_name]

                # crop the image and resize
                image = images[cam_name]
                image = image[crop['i']:crop['i']+crop['h'], crop['j']:crop['j']+crop['w']]
                image = cv2.resize(image, (image_size[1], image_size[0]))

                #apply the masks
                if int(cam_name) in self.masks and self.masks[int(cam_name)] is not None:
                    image = cv2.bitwise_and(image, image, mask=self.masks[int(cam_name)])

                # convert to tensor and normalize
                image = torch.tensor(image, dtype=torch.float32)/255.0
                image = torch.einsum('h w c -> c h w', image) # change to c h w
                image = self.image_normalize(image)
                all_images.append(image.unsqueeze(0))

            else:
                raise ValueError(f"Camera name {cam_name} not found in images")

        # get rid of velocities
        if qpos.shape[0] == 7:
            qpos = np.concatenate([qpos[:3], qpos[6:]])
        qpos, _ = self.normalizer(qpos, qpos)
        qpos_data = torch.from_numpy(qpos).float().unsqueeze(0)

        return all_images, qpos_data        

def visualize(images, qpos, actions, ground_truth=None):

    import matplotlib.pyplot as plt
    # Create a figure and axes
    fig = plt.figure(figsize=(10, 10), layout='tight')
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    axs_left = subfigs[0].subplots(len(images), 1)
    if len(images) > 1:
        for i, image in enumerate(images):
            print(image.shape)
            axs_left[i].imshow(image)     
    else:
        axs_left.imshow(images[0])

    # Make a 3D scatter plot of the actions in the right subplot. Use cmaps to color the points based on the index
    c = np.arange(len(actions))
    ax2 = subfigs[1].add_subplot(111, projection='3d')
    # ax2.scatter(actions[:, 0], actions[:, 1], actions[:, 2], c='b', marker='o')
    sc = ax2.scatter(actions[:, 0], actions[:, 1], actions[:, 2], c=c, cmap='viridis', marker='x')
    if ground_truth is not None:
        ax2.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], c=np.arange(len(ground_truth)), cmap = 'viridis', marker='o')
    ax2.scatter(qpos[0], qpos[1], qpos[2], c='r', marker='o')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Actions and Qpos')
    cbar = fig.colorbar(sc, ax=ax2, label='Time', shrink=0.5)

    # Set the axis limits
    center = np.array([0.5, 0, 0.2])
    radius = 0.15
    ax2.set_xlim(center[0] - radius, center[0] + radius)
    ax2.set_ylim(center[1] - radius, center[1] + radius)
    ax2.set_zlim(center[2] - radius, center[2] + radius)

    plt.show()    
    

import threading

class AsyncInput:
    def __init__(self, command, is_async = True):
        self.command = command
        self.responce = None
        if is_async:
            self.thread = threading.Thread(target=self._get_input, daemon=True)
            self.thread.start()
        else:
            self._get_input()

    def _get_input(self):
        self.responce = input(self.command)

if __name__ == '__main__':
    num_episodes = 1100
    grip_closed = False

    ADD_NOISE = True
    noise_std = 0.0025
    noise_mean = 0
    replan_horizon = 8
    timeout_steps = 1000

    weights_dir = 'data/weights/resnet18_epoch3500_05-09-26_2024-10-10__stonehenge_ablate'
    # weights_dir = "/home/selamg/beadsight/data/weights/clip_epoch3500_04-12-47_2024-10-10__stonehenge_clip_frozen"
    save_path = "/home/selamg/beadsight/data/ssd/experiment_results/"
    
    norm_stats_dir = "/home/selamg/beadsight/data/norm_stats/stonehenge_norm_stats.json"

    # EXPECTED_CAMERA_NAMES = ['1','2','3','4','5','6','beadsight'] 
    EXPECTED_CAMERA_NAMES = ['1','2','3','4','5','6']

    SAVE_VIDEO = True

    # offset = np.array([0, 0, -0.01])

    # for images only:
    

    use_real_robot = True
    if use_real_robot:
        from frankapy import FrankaArm
        from frankapy import FrankaConstants as FC
        from robomail.motion import GotoPoseLive
        from rospy import Rate

        # from simple_gelsight import GelSightMultiprocessed, get_camera_id
        from cameras_and_beadsight import CamerasAndBeadSight        
        print("starting")
        import time
        fa = FrankaArm()
        fa.reset_joints()
        print("resetting joints")
        fa.open_gripper()
        move_pose = FC.HOME_POSE
        # move_pose.translation = np.array([0.6, 0, 0.35])
        # fa.goto_pose(move_pose)

        default_impedances = np.array(FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES)
        new_impedances = np.copy(default_impedances)
        new_impedances[3:] = np.array([0.5, 2, 0.5])*new_impedances[3:] # reduce the rotational stiffnesses, default in gotopose live
        new_impedances[:3] = np.array([0.5, 0.5, 1])*default_impedances[:3] # reduce the translational stiffnesse

        pose_controller = GotoPoseLive(cartesian_impedances=new_impedances.tolist(), step_size=0.05)
        pose_controller.set_goal_pose(move_pose)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {device}")
        
        camera_nums = [1, 2, 3, 4, 5, 6]
        camera_sizes = [(1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (800, 1280)]
        cameras = CamerasAndBeadSight(device=36,bead_horizon=BEAD_HORIZON) #check cam test to find devicenum
        min_gripper_width = 0.007

    else:
        assert not SAVE_VIDEO, "Save video doesn't work with the fake robot"
        import h5py
        # data_dir = "/media/selamg/DATA/beadsight/data/full_dataset/run/episode_0/episode_0.hdf5"
        data_dir = "/home/selamg/beadsight/data/ssd/full_dataset/run_0/episode_0/episode_0.hdf5"
        with h5py.File(data_dir, 'r') as root:
            all_qpos_7 = root['/observations/position'][()]
            all_qpos = np.empty([all_qpos_7.shape[0], 4])
            all_qpos[:, :3] = all_qpos_7[:, :3]
            all_qpos[:, 3] = all_qpos_7[:, 6]
            gt_actions = root['/goal_position'][()]
            num_episodes = root.attrs['num_timesteps']
            #num_episodes = 50

            all_images = {}
            for cam in root.attrs['camera_names']:
                if cam == 'beadsight':
                    #how do I construct a rolling stacked object...
                    video_images = []
                    video_path = os.path.join(os.path.dirname(data_dir), f'cam-{cam}.avi')
                    cap = cv2.VideoCapture(video_path)
                    for i in range(num_episodes):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        video_images.append(frame)
                    
                    all_beadsight_images = np.array(video_images)
                    cap.release()
                
                video_images = []
                video_path = os.path.join(os.path.dirname(data_dir), f'cam-{cam}.avi')
                cap = cv2.VideoCapture(video_path)
                for i in range(num_episodes):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    video_images.append(frame)
                
                all_images[cam] = np.array(video_images)
                cap.release()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from predict_robot_actions import diffuse_robot


    with open(norm_stats_dir, 'r') as f:
        norm_stats = json.load(f)

    # convert the norm stats to numpy arrays
    for key in norm_stats:
        norm_stats[key] = np.array(norm_stats[key])

    preprocess = PreprocessData(norm_stats, EXPECTED_CAMERA_NAMES)

    print('start load')
    model_dict = torch.load(weights_dir, map_location=device)
    print('finish load')

    run = 0
    if use_real_robot:
        rate = Rate(10)
    while True:
        # if we want to save the video, create a new folder to save it too.
        save_beadsight_images = np.zeros([num_episodes, 480, 480, 3], dtype=np.uint8) # hardcoded, but whatever...

        existing_folders = [int(f.split('_')[-1]) for f in os.listdir(save_path) if f.startswith('run')]
        new_folder = max(existing_folders, default=0) + 1
        run_save_folder = os.path.join(save_path, f'run_{new_folder}')
        os.makedirs(run_save_folder)
        print(f'saving run to {run_save_folder}')

        if SAVE_VIDEO:
            file_names = [run_save_folder + f"/cam_{cam_num}.mp4" for cam_num in camera_nums]
            print(file_names)
            fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
            video_recorder = SaveVideosMultiprocessed(file_names, camera_sizes, fourcc_mp4)
        
        run += 1
        print(f'starting run {run}')
        # noise = OUNoise(3, theta=0.1, sigma=0.0005)
        if use_real_robot:
            fa.open_gripper()
            move_pose = FC.HOME_POSE
            move_pose.translation = np.array([0.44, -0.06, 0.3])

            print(fa.get_pose().translation)
            input("Press enter to continue")

            while np.linalg.norm(fa.get_pose().translation - move_pose.translation) > 0.04:
                pose_controller.step(goal_pose = move_pose)
                time.sleep(0.05)
                print('moving to home')

            input("Press enter to continue")

            integral_term = np.zeros(3)
            moved_gripper = False
            last_position = np.copy(move_pose.translation)

        total_timesteps = 0
        user_input = AsyncInput("Press enter to continue, v to visualize, or q to quit, or s to save", is_async=True)
        for i in range(num_episodes):
            print(i)
            # images = {key: all_images[key][i] for key in all_images}
            # gelsight_data = all_gelsight_data[i]

            if use_real_robot:
                images = cameras.get_next_frames() 
                beadframes = cameras.bead_buffer
            
            else:
                images = {key: all_images[key][i] for key in all_images}
                if 'beadsight' in all_images.keys():
                    if i < BEAD_HORIZON:
                        beadframes = [all_images['beadsight'][i]]
                        for c in range(BEAD_HORIZON-1):
                            beadframes.append(all_images['beadsight'][i].copy())  
                    elif i >= BEAD_HORIZON:
                        beadframes = []
                        for c in range(BEAD_HORIZON-1):
                            beadframes.append(all_images['beadsight'][i-c])  
                        beadframes.append(all_images['beadsight'][i])
                else:
                    beadframes = []

            save_beadsight_images[total_timesteps] = beadframes[-1] # record the most recent beadframe

            # show the images
            disp_images = images.copy() # create a shallow copy of the dictionary
            disp_images.pop('beadsight') # remove beadsight
            if SAVE_VIDEO:
                video_recorder(list(disp_images.values())) # save the recorded images (don't want beadsight)
            for bead_num, bead_image in enumerate(beadframes):
                disp_images[f"beadsight{bead_num}"] = bead_image # add all of the beadsight images in
            monitor_cameras(disp_images) 

            if use_real_robot:
                robo_data = fa.get_robot_state()
                current_pose = robo_data['pose']*fa._tool_delta_pose
                
                # cur_joints = robo_data['joints']
                # cur_vel = robo_data['joint_velocities']
                finger_width = fa.get_gripper_width()
                # if grip_closed:
                #     finger_width = 0.0
                qpos = np.zeros(4)
                qpos[:3] = current_pose.translation
                qpos[3] = finger_width
                print("qpos", qpos)
            else:
                qpos = all_qpos[i]
                current_pose = None

            image_data, qpos_data = preprocess.process_data(images, beadframes, qpos)

            print('normalized qpos values:', qpos_data)

            # get the action from the model
            qpos_data = qpos_data.to(device)
            image_data = [img.to(device) for img in image_data]
            for im in image_data:
                print(im.shape)
            start = time.time()
            # For gel only, image data should be of form: 
            norm_actions = diffuse_robot(qpos_data,image_data,EXPECTED_CAMERA_NAMES,model_dict,
                         pred_horizon=20,device=device)

            print('norm_actions', norm_actions)
            end = time.time()
            print('inference time', end-start)
            _, actions = preprocess.normalizer.unnormalize(qpos, norm_actions)

            actions = actions.squeeze().detach().cpu().numpy()

            # visualize the data
            vis_images = [image_data[j].squeeze().detach().cpu().numpy().transpose(1, 2, 0) for j in range(len(image_data))]
            #TODO: hardcoded:
            vis_images = vis_images[:-1]

            if not use_real_robot:
                if i %100 == 0:
                    visualize(vis_images, qpos, actions, ground_truth=gt_actions[i:i+len(actions)])
                continue

            if i >= timeout_steps/replan_horizon:
                print('Run timed out. Press enter continue')
                user_input.thread.join()
                command = input("Press q to quit, or s to save the run")
                while command != 'q' and command != 's':
                    command = input('run timed out. Press q to quit, or s to save')
                break
            if user_input.responce == 'q' or user_input.responce == 's':
                command = deepcopy(user_input.responce)
                break
            elif user_input.responce == 'v':
                visualize(vis_images, qpos, actions)
                user_input = AsyncInput("Press enter to continue, v to visualize, or q to quit, or s to save", is_async=True)
            elif user_input.responce is not None:
                user_input = AsyncInput("Press enter to continue, v to visualize, or q to quit, or s to save", is_async=True)

            print('actions', actions)
            print('current pose', current_pose.translation)
            for action_idx, move_action in enumerate(actions[:replan_horizon]):
                total_timesteps += 1
                print("move_action", move_action)
                if action_idx != 0: # don't need to run the first time, because we got it above.
                    beadsight_buffer = cameras.get_and_update_bead_buffer() #update the buffer at rate of action execution
                    save_beadsight_images[total_timesteps] = beadsight_buffer[-1] # last item is the most recent picture.
                # move the robot:
                if np.linalg.norm(last_position - current_pose.translation) < 0.0025 and not moved_gripper: # if the robot hasn't moved much, add to the integral term (to deal with friction)
                    integral_term += (move_action[:3] - current_pose.translation)*0.25
                else:
                    integral_term = np.zeros(3)
                    

                # integral_term = np.zeros(3)
                integral_term[2] = min(0, integral_term[2]) # don't wind up donwards
                print("integral_term", integral_term)
                last_position = np.copy(current_pose.translation)

                move_pose = FC.HOME_POSE
                if ADD_NOISE:
                    move_pose.translation = move_action[:3] + integral_term + np.random.normal(noise_mean, noise_std, 3)
                else:
                    move_pose.translation = move_action[:3] + integral_term
                # move_pose.translation = ensembled_action[:3] 
                
                current_pose = fa.get_pose()
                pose_controller.step(move_pose, current_pose)
                grip_command = move_action[3]
                grip_command = np.clip(grip_command, 0, 0.08)
                if grip_command <= min_gripper_width:
                    if not grip_closed:
                        grip_closed = True
                        fa.goto_gripper(min_gripper_width)
                        moved_gripper = True
                        print("closing gripper")
                    else:
                        moved_gripper = False
                else:
                    grip_closed = False
                    fa.goto_gripper(grip_command, block=False, speed=0.15, force = 10)
                    moved_gripper = True
                
                print("moving to", move_pose.translation, grip_command)

                rate.sleep()



        if command == 's':
            # save the beadsight data
            np.save(run_save_folder + "/beadsight_obs.npy", save_beadsight_images[:total_timesteps])
            video_recorder.close()
            
            while True:
                was_successful = input('Was the run successful? (y/n)')
                if was_successful == 'y' or was_successful == 'n':
                    break

            run_stats = {"num_timesteps": total_timesteps, 
                         "replan_horizon": replan_horizon, 
                         "ADD_NOISE": ADD_NOISE, 
                         "noise_std": noise_std,
                         "noise_mean": noise_mean,
                         "was_successful": was_successful,
                         "weight_path": weights_dir}
            
            with open(run_save_folder + '/run_stats.json', 'w') as f:
                json.dump(run_stats, f)

        else:
            video_recorder.close()
            # delete the save data folder:
            input(f'Deleting failed run data. Press enter to delete {run_save_folder}')
            shutil.rmtree(run_save_folder)

        if input('Press d to quit, or enter to continue') == 'd':
            break


        print('next run')
    print("done")
