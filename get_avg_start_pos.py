
import numpy as np
import h5py
import os
from tqdm import tqdm

def avg_start_pos(dataset_dir, num_episodes) -> tuple:
    all_start = []
    for episode_idx in tqdm(range(num_episodes), desc="Get Average Start Pos"):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:

            # if gelsight data exists, get the average
            if 'observations/qpos' in root:
                qpos = root['observations/qpos'][()]
                first_qpos = qpos[0][:]
                all_start.append(first_qpos)
                # print("shapes", np.array(all_start).shape)


    all_start = np.array(all_start)
    print(all_start.shape)    
    avg_start_pos = all_start.mean(axis=0)
    
    
    return avg_start_pos




if __name__ == '__main__':

    datadir = '/media/selamg/Crucial/selam/processed_ishape_2'
    avg = avg_start_pos(datadir, 100)
    print(avg)

# for stonehenge: [ 0.51642333 -0.01241467  0.38239434  0.07577089]
# for i shape: [ 0.46060304 -0.03064966  0.39440889  0.07545292]



# filename = '/home/selamg/beadsight/data/ssd/processed_ishape/episode_0.hdf5'

# with h5py.File(filename, "r") as f:
#     # Print all root level object names (aka keys) 
#     # these can be group or dataset names 
#     print("Keys: %s" % f.keys())
#     # get first object name/key; may or may NOT be a group
#     a_group_key = list(f.keys())[0]
#     print(f['observations']['qpos'][0])

#     # get the object type for a_group_key: usually group or dataset
#     print(type(f[a_group_key])) 

#     # If a_group_key is a group name, 
#     # this gets the object names in the group and returns as a list
#     data = list(f[a_group_key])

#     # If a_group_key is a dataset name, 
#     # this gets the dataset values and returns as a list
#     data = list(f[a_group_key])
#     # preferred methods to get dataset values:
#     ds_obj = f[a_group_key]      # returns as a h5py dataset object
#     ds_arr = f[a_group_key][()]  # returns as a numpy array
