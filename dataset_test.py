    
from dataset import DiffusionEpisodicDataset 
from utils import get_norm_stats

import torch
import numpy as np

from torchvision import transforms

import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from network import get_resnet, replace_bn_with_gn, ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

#obs_horizon = 1
pred_horizon = 5
#action_horizon = 4

dataset_path = "/media/selamg/DATA/beadsight/data/processed_data"
# dataset_path = "/home/selamg/beadsight/data/ssd/processed_data"

dataset_dir = dataset_path

# with open(os.path.join(save_dir, 'meta_data.json'), 'r') as f:
#     meta_data: Dict[str, Any] = json.load(f)
# task_name: str = meta_data['task_name']
# num_episodes: int = meta_data['num_episodes']
# # episode_len: int = meta_data['episode_length']
# camera_names: List[str] = meta_data['camera_names']
# is_sim: bool = meta_data['is_sim']
# state_dim:int = meta_data['state_dim']

num_episodes = 13

norm_stats = get_norm_stats(dataset_dir, num_episodes)
camera_names = [1,2,3,4,5,6,"beadsight"]

train_ratio = 0.8
shuffled_indices = np.random.permutation(num_episodes)
train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
val_indices = shuffled_indices[int(train_ratio * num_episodes):]


t = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.5), #0.1, p = 0.5
    transforms.RandomResizedCrop(size=[480,480], scale=(0.8,1.0),ratio=(1,1)) #0.9, 1.0

    ])

train_dataset = DiffusionEpisodicDataset(train_indices, dataset_dir, pred_horizon, camera_names, norm_stats, image_transforms=t)


dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=2,
    num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)


batch = next(iter(dataloader))

print(batch.keys())
print("batch['beadsight'].shape:", batch['beadsight'].shape)
print("batch[6].shape:", batch[6].shape)
print("batch['agent_pos'].shape:", batch['agent_pos'].shape)
print("batch['action'].shape:", batch["action"].shape)

image = batch[4][0]
image = image.flatten(end_dim=1)
image = image.permute(1,2,0)
image = image.detach().to('cpu')
image = image.numpy()

import cv2
cv2.imshow('transformed im',image)
cv2.waitKey(0)

# weights_dir = "/home/selamg/beadsight/data/ssd/weights/clip_epoch3500_23-56-01_2024-06-01"
# weights_dir = "/media/selamg/DATA/beadsight/data/final_wts/clip_epoch3000_03-25-05_2024-06-01"

# model_dict = torch.load(weights_dir, map_location='cuda')

# nets = nn.ModuleDict(model_dict).to('cuda')

# obs_horizon = 1
# bead = batch['beadsight'][:,:obs_horizon].to('cuda')

# print("bead:", bead.shape)
# bead_features = nets['beadsight_encoder'](bead.flatten(end_dim=1))
# print("success")
# import cv2

# #done in visualize_waypts/predict_diff_actions
# bead = batch["beadsight"][0].flatten(end_dim=1).to('cpu')
# bead = bead.permute(1,2,0) #H W C
# bead = bead.detach().to('cpu')
# print("bead",bead.shape)

# beadframes = []
# start = 0
# assert bead.shape[0] % 3 == 0
# for i in range(int(bead.shape[0]/3)):
#     print(i)
#     frame = bead[:,:,start:((i+1)*3)]
#     beadframes.append(frame)
#     start += 3




#works but is a weird color bc needs to be unnormalized
# import cv2
# x = beadframes[0].permute(1,2,0).numpy()
# print(x.shape)
# cv2.imshow('test',x)
# cv2.waitKey(0)
# import pdb; pdb.set_trace()


#@markdown  - key `image`: shape (obs_horizon, 3, 96, 96)
#@markdown  - key `agent_pos`: shape (obs_horizon, 2)
#@markdown  - key `action`: shape (pred_horizon, 2)

# from gelsight:
#batch['image'].shape: torch.Size([1, 7, 3, 400, 480])
# batch['agent_pos'].shape: torch.Size([1, 4])
# batch['action'].shape: torch.Size([1, 5, 4])
# i forget why action is 5,4 but its right

# dict_keys([1, 2, 3, 4, 5, 6, 'beadsight', 'agent_pos', 'action'])
# batch['beadsight'].shape: torch.Size([2, 1, 15, 480, 480]) this is 3NxHxW
# batch[6].shape: torch.Size([2, 1, 3, 400, 480])
# batch['agent_pos'].shape: torch.Size([2, 1, 4])
# batch['action'].shape: torch.Size([2, 5, 4])

# beadsight_encoder = get_resnet('resnet18')
# beadsight_encoder = nn.Sequential(nn.Conv2d(15,3,3),beadsight_encoder)
# beadsight_encoder = replace_bn_with_gn(beadsight_encoder)


