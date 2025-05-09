import os
import h5py
import numpy as np
import shutil
import cv2
from matplotlib import pyplot as plt
from typing import List, Dict
from multiprocessing import Pool
import json
# from utils import gelsight_norm_stats

CROP_PARAMS = {
    '1': {'i': 0, 'j': 312, 'h': 1080, 'w': 1296, 'size': (1080, 1920)},
    '2': {'i': 108, 'j': 775, 'h': 755, 'w': 906, 'size': (1080, 1920)},
    '3': {'i': 324, 'j': 768, 'h': 595, 'w': 714, 'size': (1080, 1920)},
    '4': {'i': 360, 'j': 648, 'h': 560, 'w': 672, 'size': (1080, 1920)},
    '5': {'i': 150, 'j': 350, 'h': 595, 'w': 714, 'size': (1080, 1920)},
    '6': {'i': 212, 'j': 425, 'h': 375, 'w': 450, 'size': (800, 1200)},
    # 'beadsight' : {'i': 0, 'j': 0, 'h': 480, 'w': 480, 'size': (480, 480)}
}

MASK_VERTICIES = {'5': [[0, 0.77], [0.0625, 1], [0, 1]],
                  '6': [[0, 0.79], [0.0875, 1], [0, 1]]}

# mask out background of images (for where curtain was taken off)
def make_masks(image_size, verticies):
    masks = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
    for cam in verticies:
        mask = np.ones((image_size[0], image_size[1]), dtype=np.uint8)
        pts = np.array(verticies[cam])*image_size
        pts = np.fliplr(pts.astype(np.int32)) # cv2 requires the points to be in the format (x, y), not (i, j)
        cv2.fillPoly(mask, [pts], 0)
        masks[cam] = mask
    return masks
        
def uncompress_data(source_folder, save_path, image_size = [400, 480], masks: Dict[str, np.ndarray] = {}, use_rot = False):
    # First, copy the hdf files to the save_path
    # Find the hdf5 files in the source folder
    print("starting", source_folder)
    try:
        h5py_files = []
        for file in os.listdir(source_folder):
            if file.endswith('.hdf5'):
                h5py_files.append(file)

        assert len(h5py_files) == 1, f"Expected 1 hdf5 file, but found {len(h5py_files)}"


        # open the new hdf5 file
        with h5py.File(os.path.join(source_folder, h5py_files[0]), 'r') as old:
            with h5py.File(save_path, 'w') as new:
                # copy the attributes
                new.attrs['camera_names'] = old.attrs['camera_names']
                new.attrs['image_height'] = image_size[0]
                new.attrs['image_width'] = image_size[1]
                new.attrs['realtimes'] = old.attrs['realtimes']
                new.attrs['num_timesteps'] = old.attrs['num_timesteps']
                new.attrs['sim'] = old.attrs['sim']

                position = old['observations/position']
                velocity = old['observations/velocity']
                action = old['goal_position']
                if not use_rot:
                    new.attrs['position_dim'] = 4
                    new.attrs['velocity_dim'] = 4
                    new.attrs['position_doc'] = "x, y, z, gripper"
                    new.attrs['velocity_doc'] = "x_dot, y_dot, z_dot, gripper_vel"
                    position = np.delete(position, [3, 4, 5], axis=1)
                    velocity = np.delete(velocity, [3, 4, 5], axis=1)
                    action = np.delete(action, [3, 4, 5], axis=1)
                else:
                    new.attrs['position_dim'] = old.attrs['position_dim']
                    new.attrs['velocity_dim'] = old.attrs['velocity_dim']
                    new.attrs['position_doc'] = old.attrs['position_doc']
                    new.attrs['velocity_doc'] = old.attrs['velocity_doc']
                


                # copy the datasets
                new.create_dataset('action', data=action, chunks=(1, action.shape[1]))
                
                obs = new.create_group('observations')
                obs.create_dataset('qpos', data=position, chunks=(1, position.shape[1]))
                obs.create_dataset('qvel', data=velocity, chunks=(1, velocity.shape[1]))

                
                # uncompress the images
                # save each camera image
                image_group = obs.create_group('images')
                for cam_name in old.attrs['camera_names']:
                    # print('starting cam', cam_name) #debug
                    # open the video file
                    video_path = os.path.join(source_folder, f'cam-{cam_name}.avi')
                    cap = cv2.VideoCapture(video_path)
                    # get the number of frames
                    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    assert num_frames == old.attrs['num_timesteps'], f"Number of frames in video {num_frames} does not match number of timesteps in hdf5 file {old.attrs['num_timesteps']}"
                    
                    if cam_name == 'beadsight':
                        images = np.empty((num_frames, 480, 480, 3), dtype=np.uint8)
                        for i in range(num_frames):
                            ret, frame = cap.read()
                            images[i] = frame

                    else:
                        crop = CROP_PARAMS[cam_name]
                        images = np.empty((num_frames, image_size[0], image_size[1], 3), dtype=np.uint8)
                        # loop through the frames and save them in the hdf5 file
                        for i in range(num_frames):
                            ret, frame = cap.read()
                            # crop the frame
                            frame = frame[crop['i']:crop['i']+crop['h'], crop['j']:crop['j']+crop['w']]
                            # resize the frame and save
                            frame = cv2.resize(frame, (image_size[1], image_size[0]))
                            # apply the mask 
                            if cam_name in masks and masks[cam_name] is not None:
                                frame = cv2.bitwise_and(frame, frame, mask=masks[cam_name])
                            images[i] = frame

                    cap.release()


                    
                    # save the images in the hdf5 file
                    image_group.create_dataset(name=f'{cam_name}', dtype='uint8', 
                                            chunks=(1, image_size[0], image_size[1], 3),
                                            data=images)       
    except Exception as e:
        print(e)
        print(source_folder, save_path)

            
def process_folder(source_folders, save_folder, image_size = [400, 480], masks = {}):
    # find all the episodes in the source folders recursively
    h5py_files = []
    for source_folder in source_folders:
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if file.endswith('.hdf5'):
                    h5py_files.append(os.path.join(root, file))

    save_paths = []
    episode_folders = []
    for i, h5py_file in enumerate(h5py_files):

        save_paths.append(os.path.join(save_folder, f'episode_{i}.hdf5'))
        episode_folders.append(os.path.dirname(h5py_file)) # the episode folder will be the parent of the h5py file
    
    # uncompress the data, multiprocessed
    with Pool(processes=12) as p:
        p.starmap(uncompress_data, zip(episode_folders, save_paths, [image_size]*len(save_paths), [masks]*len(save_paths)))

    # for i in range(len(episode_folders)):
    #     print(i, episode_folders[i], save_paths[i])
    #     uncompress_data(episode_folders[i], save_paths[i], image_size, masks)

            
if __name__ == "__main__":
    image_size = [400, 480]
    masks = make_masks(image_size, MASK_VERTICIES)

    # source_folders = ['/media/selamg/DATA/beadsight/data/full_dataset/']
    # save_folder = '/media/selamg/DATA/beadsight/data/processed_data_test/' 

    source_folders = ['/home/selamg/beadsight/data/ssd/drawer_supporting']
    save_folder = '/media/selamg/Crucial/selam/processed_drawer_supporting'
    # save_folder = '/home/selamg/beadsight/data/ssd/processed_drawer_supporting'

    process_folder(source_folders, save_folder, image_size, masks)
    #might be slow (10-30 minutes) 

    # save_norm_stats(save_folder, 100)

    # source_file = '/home/aigeorge/research/TactileACT/data/original/camara_cage_1/run_0/episode_3'
    # save_file = '/home/aigeorge/research/TactileACT/test.hdf5'
    # uncompress_data(source_file, save_file, image_size, masks, use_rot=False)


