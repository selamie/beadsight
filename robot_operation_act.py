import numpy as np
from torchvision import transforms
from process_data_cage import MASK_VERTICIES, CROP_PARAMS, make_masks
from typing import Dict, List, Tuple
import cv2
import torch
import os
from ACT.utils import NormalizeDeltaActionQpos
from HardwareTeleop.multiprocessed_cameras import SaveVideosMultiprocessed
import shutil
from ACT.load_ACT import load_ACT

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
        self.normalizer = NormalizeDeltaActionQpos(norm_stats)
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
        qpos = self.normalizer.normalize_qpos(qpos)
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
    timeout_steps = 500

    temporal_ensemble = True
    K = 0.25


    model_path = "/home/selamg/beadsight/data/ssd/weights/ACT/pretrain_vision_only_2/policy_last.ckpt"
    save_path = "/home/selamg/beadsight/data/ssd/experiment_results/ACT/pretrained_vision_only"
    args_file = "/home/selamg/beadsight/data/ssd/weights/ACT/pretrain_vision_only_2/args.json"

    # model_path = "/home/selamg/beadsight/data/ssd/weights/ACT/pretrain_both_8/policy_last.ckpt"
    # save_path = "/home/selamg/beadsight/data/ssd/experiment_results/ACT/pretrain_both"
    # args_file = "/home/selamg/beadsight/data/ssd/weights/ACT/pretrain_both_8/args.json"

    # model_path = "/home/selamg/beadsight/data/ssd/weights/ACT/no_pretrain_vision_only_1/policy_last.ckpt"
    # save_path = "/home/selamg/beadsight/data/ssd/experiment_results/ACT/no_pretrain_vision_only"
    # args_file = "/home/selamg/beadsight/data/ssd/weights/ACT/no_pretrain_vision_only_1/args.json"

    # model_path = "/home/selamg/beadsight/data/ssd/weights/ACT/pretrain_both_frozen_3/policy_last.ckpt"
    # args_file = "/home/selamg/beadsight/data/ssd/weights/ACT/pretrain_both_frozen_3/args.json"
    # save_path = "/home/selamg/beadsight/data/ssd/experiment_results/ACT/pretrain_both_frozen"


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
        cameras = CamerasAndBeadSight(device=6,bead_horizon=BEAD_HORIZON) #check cam test to find devicenum
        min_gripper_width = 0.006

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

    args = json.load(open(args_file, 'r'))
    modified_args = args.copy() # we have the backbone weights in the saved weights, so we don't need to load them seperatly
    modified_args['beadsight_backbone_path'] = "none" 
    modified_args['vision_backbone_path'] = "none"

    act = load_ACT(model_path, args_file, modified_args).to(device)
    act.eval()


    # convert the norm stats to numpy arrays
    norm_stats = {k: np.array(v) for k, v in args['norm_stats'].items()}
    print('normstats', norm_stats)

    horizon = args['chunk_size']

    preprocess = PreprocessData(norm_stats, args['camera_names'])

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
        action_history = np.zeros([num_episodes + horizon, num_episodes, 4]) # prediction_time, time_preiction_was_made, action
        confidences = []
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
            total_timesteps += 1

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
            # for im in image_data:
            #     print(im.shape)

            start_time = time.time()
            with torch.no_grad():
                deltas:torch.Tensor = act(qpos_data, image_data)
            print('act inference time', time.time() - start_time)
            
            # unnormalize the actions and qpos 
            deltas = deltas.squeeze().detach().cpu().numpy()
            unnormalized_deltas = preprocess.normalizer.unnormalize_delta(deltas)
            # print("unnormalized_deltas", unnormalized_deltas)

            all_actions = unnormalized_deltas + qpos
            all_actions[:, 3] = np.clip(unnormalized_deltas[:, 3], 0, 0.08) # use output grip width and clip it
            print('predicted actions', all_actions)

            action_history[i:i+horizon, i] = all_actions[:] 

            if temporal_ensemble:
                ensembled_action = np.zeros(4)
                total_weight = 0
                time_step_actions = []
            
            
                confidences.insert(0, 1) # add the confidence to the front of the list (this time step)
                if len(confidences) > horizon:
                    confidences.pop() # remove the last confidence from the list 
                
                for t in range(min(i+1, horizon)):
                    ensembled_action += confidences[t]*action_history[i, i-t, :]
                    total_weight += confidences[t]
                    confidences[t] *= np.exp(-K) # update the confidence
                    time_step_actions.append(action_history[i, i-t, :])

                time_step_actions = np.array(time_step_actions)
                ensembled_action /= total_weight
                print("time_step_actions", time_step_actions)
                # print('confidences', confidences)
                print('current_delta', unnormalized_deltas[0])
                print("current_pose", current_pose.translation)
                print("ensembled_action", ensembled_action)
                print('ensembled_delta', ensembled_action[:3] - qpos[:3])
                # print('current_action', all_actions[0])
                print("K", K)


            # visualize the data
            vis_images = [image_data[j].squeeze().detach().cpu().numpy().transpose(1, 2, 0) for j in range(len(image_data))]
            
            #TODO: hardcoded:
            vis_images = vis_images[1:]

            # if not use_real_robot:
            #     if i %100 == 0:
            #         visualize(vis_images, qpos, all_actions, ground_truth=time_step_actions)
            #     continue

            if i >= timeout_steps:
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
                visualize(vis_images, qpos, all_actions, ground_truth=time_step_actions)
                user_input = AsyncInput("Press enter to continue, v to visualize, or q to quit, or s to save", is_async=True)
            elif user_input.responce is not None:
                user_input = AsyncInput("Press enter to continue, v to visualize, or q to quit, or s to save", is_async=True)

            # move the robot:
            if np.linalg.norm(last_position - current_pose.translation) < 0.0025 and not moved_gripper: # if the robot hasn't moved much, add to the integral term (to deal with friction)
                integral_term += (ensembled_action[:3] - current_pose.translation)*0.25
            else:
                integral_term = np.zeros(3)

            integral_term = np.clip(integral_term, -0.03, 0.03)
            # integral_term = np.zeros(3)
            
            
            last_position = np.copy(current_pose.translation)

            integral_term[2] = max(integral_term[2], 0) # don't wind up the z term in the negative direction (contact with the table)
            print("integral_term", integral_term)
            move_pose = FC.HOME_POSE
            if ADD_NOISE:
                move_pose.translation = ensembled_action[:3] + integral_term + np.random.normal(noise_mean, noise_std, 3)
            else:
                move_pose.translation = ensembled_action[:3] + integral_term
            # move_pose.translation = ensembled_action[:3] 
                
            # current_pose = fa.get_pose()
            pose_controller.step(move_pose, current_pose)
            grip_command = ensembled_action[3]
            grip_command = np.clip(grip_command, 0, 0.08)
            start_command_time = time.time()
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
                
            print('command time', time.time() - start_command_time)

            rate.sleep()
            print('total time', time.time() - start_time)



        if command == 's':
            # save the beadsight data
            np.save(run_save_folder + "/beadsight_obs.npy", save_beadsight_images[:total_timesteps])
            video_recorder.close()
            
            while True:
                was_successful = input('Was the run successful? (y/n)')
                if was_successful == 'y' or was_successful == 'n':
                    break

            run_stats = {"num_timesteps": total_timesteps, 
                         "temporal_ensemble": temporal_ensemble,
                         "K": K,
                         "ADD_NOISE": ADD_NOISE, 
                         "noise_std": noise_std,
                         "noise_mean": noise_mean,
                         "was_successful": was_successful,
                         "training_args": args}
            
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
