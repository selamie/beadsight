import numpy as np
from multiprocessed_cameras import SaveVideosMultiprocessed
from cameras_and_beadsight import CamerasAndBeadSight
import h5py
import os
import shutil
import time
import cv2
from tqdm import tqdm
from multiprocessing import Pool
from typing import List, Tuple, Dict, Union
import threading
import datetime

CROP_PARAMS = {
    1: {'i': 0, 'j': 312, 'h': 1080, 'w': 1296, 'size': (1080, 1920), 'fliplr': False, 'flipud': False},
    2: {'i': 108, 'j': 775, 'h': 755, 'w': 906, 'size': (1080, 1920), 'fliplr': False, 'flipud': False},
    3: {'i': 324, 'j': 768, 'h': 595, 'w': 714, 'size': (1080, 1920), 'fliplr': False, 'flipud': False},
    4: {'i': 360, 'j': 648, 'h': 560, 'w': 672, 'size': (1080, 1920), 'fliplr': False, 'flipud': False},
    5: {'i': 150, 'j': 350, 'h': 595, 'w': 714, 'size': (1080, 1920), 'fliplr': False, 'flipud': False},
    6: {'i': 212, 'j': 425, 'h': 375, 'w': 450, 'size': (800, 1200), 'fliplr': True, 'flipud': False},
}

class ImageDisplayThread(threading.Thread):
    def __init__(self, window_name):
        super(ImageDisplayThread, self).__init__()
        self.window_name = window_name
        self.stopped = False
        self.image = None

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # resize to 900x1800
        cv2.resizeWindow(self.window_name, 1800, 900)
        while not self.stopped:
            if self.image is not None:
                cv2.imshow(self.window_name, self.image)
                cv2.waitKey(30)
        cv2.destroyWindow(self.window_name)

    def stop(self):
        self.stopped = True

    def update_image(self, image):
        self.image = image

cv2_dispay = ImageDisplayThread('Cameras')


def monitor_cameras(frames: Dict[str, np.ndarray]):
    # print('show cams')
    out_size = (900, 1800, 3)
    # if gelsight_frame is not None:
    #     gelsight_frame = (visualize_gelsight_data(gelsight_frame)*255).astype(np.uint8)
    
    n_col = int(np.ceil(np.sqrt(len(frames)))) # 3
    n_row = int(np.ceil(len(frames) / n_col))

    # Create a grid of images
    tile_size = (int(out_size[0]/n_row), int(out_size[1]/n_col))
    grid = np.zeros(out_size, dtype=np.uint8)

    for i, (name, frame) in enumerate(frames.items()):
        if name != 'beadsight':
            frame = frame[CROP_PARAMS[int(name)]['i']:CROP_PARAMS[int(name)]['i']+CROP_PARAMS[int(name)]['h'], 
                        CROP_PARAMS[int(name)]['j']:CROP_PARAMS[int(name)]['j']+CROP_PARAMS[int(name)]['w']]
            if name == '6': # rotate the 6th camera
                frame = np.rot90(frame).copy()
            if CROP_PARAMS[int(name)]['fliplr']:
                frame = np.fliplr(frame).copy()
            if CROP_PARAMS[int(name)]['flipud']:
                frame = np.flipud(frame).copy()
        row = i // n_col
        col = i % n_col
        scale_factor = min(tile_size[0]/frame.shape[0], tile_size[1]/frame.shape[1])
        frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        grid[row*tile_size[0]:row*tile_size[0]+frame.shape[0], 
            col*tile_size[1]:col*tile_size[1]+frame.shape[1]] = frame
    
    cv2_dispay.update_image(grid)


class DataRecorder:
    def __init__(self, save_folder_path, 
                 camera_numbers=[1, 2, 3, 4, 5, 6], 
                 camera_sizes: List[Tuple[int, int]] = [(1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (800, 1280)], 
                 position_dim=7, 
                 velocity_dim=7, 
                 overwrite=False,
                 beadsight_size = (480, 480),
                 max_time_steps=1000,
                 fps = 30):
                            
        self.max_time_steps = max_time_steps
        self.fps = fps
        # Ensure that the save path ends with a '/'
        if save_folder_path[-1] != '/':
            save_folder_path += '/'
        self.save_path = save_folder_path
        # Check if the save path exists. If it deosn't, create it.
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Ensure that files are not overwritten (unless overwrite=True)
        elif not overwrite:
            Warning(f'Folder {self.save_path} already exists. New file path will be generated.')
            n = 0
            while os.path.exists(self.save_path):
                self.save_path = save_folder_path[:-1] + f'_{n}/' # remove the '/' and add the suffix
                n += 1
            os.makedirs(self.save_path)
            print(f'New save path: {self.save_path}')
        
        else:
            print(f'Folder {self.save_path} already exists. Files will be overwritten.')

        self.cameras = CamerasAndBeadSight(camera_numbers, camera_sizes, frame_rate=fps)
        
        self.camera_names = [str(cam_num) for cam_num in camera_numbers]
        self.camera_names.append("beadsight")
        self.camera_sizes = camera_sizes
        self.camera_sizes.append(beadsight_size)
        self.position_dim = position_dim
        self.velocity_dim = velocity_dim
        
        self.episode_index = -1

        self.current_times = [] #times in UTC seconds for later analysis

        self.reset_episode()
        
    def reset_episode(self, delete_last_episode = False):
        self.position: List[np.ndarray] = []
        self.velocity: List[np.ndarray] = []
        self.goal_position: List[np.ndarray] = []

        self.time_steps: int = 0

        if delete_last_episode:
            shutil.rmtree(self.save_dir)
        else:
            self.episode_index += 1

        self.save_dir = os.path.join(self.save_path, f'episode_{self.episode_index}')
        os.makedirs(self.save_dir)

        self.video_paths = [os.path.join(self.save_dir, f'cam-{cam_name}.avi') for cam_name in self.camera_names]
        fourcc = cv2.VideoWriter_fourcc(*'HFYU')

        if self.episode_index > 0: # not first episode:
            self.save_images.close()
        
        self.save_images = SaveVideosMultiprocessed(self.video_paths, self.camera_sizes, fourcc, self.fps)

        
    def record_data(self, position:np.ndarray, goal_position:np.ndarray, velocity:np.ndarray):
        start_time = time.time()
        
        assert self.time_steps < self.max_time_steps, "max_time_steps reached"
        
        # save the position, velocity, and goal position. Make them immutable so
        # that they can't be modified by other code:
        self.position.append(position)
        self.velocity.append(velocity)
        self.goal_position.append(goal_position)
        self.position[-1].setflags(write=False)
        self.velocity[-1].setflags(write=False)
        self.goal_position[-1].setflags(write=False)

        self.current_times.append(datetime.datetime.now().timestamp())


        # # Save the camera data
        frames = self.cameras.get_next_frames()
        self.save_images(list(frames.values()))
        
        monitor_cameras(frames)
        
        self.time_steps += 1
        # print(f"record time: {time.time() - start_time:.3f} secs")
        
    def write_to_file(self):
        t0 = time.time()
        file_path = os.path.join(self.save_path, f'episode_{self.episode_index}/episode_{self.episode_index}.hdf5')

        with h5py.File(file_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = False
            # root.attrs['use_gelsight'] = self.use_gelsight
            root.attrs['camera_names'] = self.camera_names
            root.attrs['num_timesteps'] = self.time_steps
            root.attrs['realtimes'] = self.current_times
            root.attrs['position_dim'] = self.position_dim
            root.attrs['velocity_dim'] = self.velocity_dim
            root.attrs['image_sizes'] = self.camera_sizes

            root.attrs['position_doc'] = "x, y, z, roll, pitch, yaw, gripper_width. Rotation is with respect to the default orientation of the gripper."
            root.attrs['velocity_doc'] = "x_dot, y_dot, z_dot, roll_dot, pitch_dot, yaw_dot, gripper_vel (set to 0)."
            obs = root.create_group('observations')

            # save the paths to the videos:
            root.attrs['video_paths'] = self.video_paths
            
            obs.create_dataset('position', (self.time_steps, self.position_dim),
                               data=self.position,
                               chunks=(1, self.position_dim))
            obs.create_dataset('velocity', (self.time_steps, self.velocity_dim),
                               data=self.velocity,
                               chunks=(1, self.velocity_dim))
            root.create_dataset('goal_position', (self.time_steps, self.position_dim),
                                data=self.goal_position,
                                chunks=(1, self.position_dim))

        print(f'Saving: {time.time() - t0:.1f} secs\n') 
        
        # clear the episode data
        self.reset_episode()

    def __del__(self):
        self.cameras.close()
        self.save_images.close()

def main():
    # Simple video record test
    cv2_dispay.start()

    save_path = '/home/selamg/beadsight/data/ssd/testdir'
    sizes = [(1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (1080, 1920), (800, 1280)]
    recorder = DataRecorder(save_path, camera_numbers=[1, 2, 3, 4, 5, 6], camera_sizes=sizes, fps=30)
    
    for j in range(5):
        for i in range(100):
            print(i)
            last_time = time.time()
            recorder.record_data(np.random.rand(7), np.random.rand(7), np.random.rand(7))
            print("total record time: ", time.time() - last_time)
            # time.sleep(01)
        print('start write')
        recorder.write_to_file()
        print('finished write')

        
if __name__ == '__main__':
    main()
    exit()
    # run_multicam_loop()

