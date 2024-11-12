import h5py
import numpy as np
import os
##NOTE: 
## difference between this and get_avg_start_pos.py is that this is all data not just first 100 eps

# EXPECTED_CAMERA_NAMES = ['1','2','3','4','5','6','beadsight'] 

# print("CAMNAMES:", EXPECTED_CAMERA_NAMES)

# data_dir = "/home/selamg/beadsight/data/ssd/full_dataset/run_2/episode_3/episode_3.hdf5"
# with h5py.File(data_dir, 'r') as root:
#     qpos = root['/observations/position'][()]
#     print(qpos[0])


def uncompress_data(source_folder):
    # Find the hdf5 files in the source folder
    print("starting", source_folder)
    all_qpos = []
    try:
        h5py_files = []
        for file in os.listdir(source_folder):
            if file.endswith('.hdf5'):
                h5py_files.append(file)

        # assert len(h5py_files) == 1, f"Expected 1 hdf5 file, but found {len(h5py_files)}"

        # open the new hdf5 files
        for i in range(len(h5py_files)):
            with h5py.File(os.path.join(source_folder, h5py_files[0]), 'r') as root:

                qpos = root['/observations/qpos'][()]
                # print
                all_qpos.append(qpos[0])
                    

    except Exception as e:
        print(e)
        print(source_folder)

    return all_qpos

def process_folder(source_folders):
    # find all the episodes in the source folders recursively
    h5py_files = []
    for source_folder in source_folders:
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if file.endswith('.hdf5'):
                    h5py_files.append(os.path.join(root, file))

    episode_folders = []
    for i, h5py_file in enumerate(h5py_files):
        episode_folders.append(os.path.dirname(h5py_file)) # the episode folder will be the parent of the h5py file
    
    all_qpos = []
    for i in range(len(episode_folders)):
         print(i, episode_folders[i])
         qpos = uncompress_data(episode_folders[i])
         all_qpos += qpos
    return all_qpos

if __name__ == "__main__":
    source_folders = '/media/selamg/Crucial/selam/processed_drawer'

    all_qpos = uncompress_data(source_folders)
    print(all_qpos)
    print('mean', np.mean(all_qpos, axis = 0))
