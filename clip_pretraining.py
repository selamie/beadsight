import torchvision
from torch import nn
import torch
from typing import Tuple, Dict, Union, Callable, List
from torch.utils.data import DataLoader
import h5py
import os
import cv2
from torchvision.transforms import Normalize
import numpy as np


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

# the projection head for CLIP. I'm using resnet's approach of an average pooling layer followed by a linear layer.
class ClipProjectionHead(nn.Module):
    def __init__(self, out_dim: int, conditioning_dim: int = 0, num_channels:int = 512, normailize: bool = True):
        """
        Create a projection head for CLIP. The projection head consists of an 
        average pooling layer followed by a linear layer.
        out_dim: The output dimension of the linear layer.
        conditioning_dim: The dimension of the conditioning vector. If 0, no conditioning is used.
        num_channels: The number of channels in the feature map.
        normailize: If true, the output of the linear layer is normalized. (default: True)
        """

        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1, -1)
        self.linear = nn.Linear(num_channels + conditioning_dim, out_dim)
        self.normalize = normailize
    
    def forward(self, feature_map, conditioning=None) -> torch.Tensor:
        x = self.pooling(feature_map)
        x = self.flatten(x)
        if conditioning is not None:
            x = torch.cat((x, conditioning), dim=-1)

        x = self.linear(x)

        if self.normalize:
            x = F.normalize(x, dim=-1)

        return x

def modified_resnet18(features_per_group=16) -> nn.Module:
    """
    Get a resnet18 model with all BatchNorm layers replaced with GroupNorm.
    weights: The weights to load into the model. If None, uses default pretraiend weights.
    features_per_group: The number of features per group in the GroupNorm layer.
    return: The modified resnet18 model."""
    # get a resnet18 model
    resnet18 = getattr(torchvision.models, 'resnet18')()

    # remove the final fully connected layer and average pooling
    resnet18 = nn.Sequential(*list(resnet18.children())[:-2])

    # replace all BatchNorm with GroupNorm
    resnet18 = replace_bn_with_gn(resnet18, features_per_group=features_per_group)
    return resnet18    
def get_beadsight_clip(features_per_group=16):
    beadsight_encoder = modified_resnet18(features_per_group=features_per_group)
    beadsight_encoder = nn.Sequential(nn.Conv2d(15,3,3),beadsight_encoder)
    return beadsight_encoder



#TODO: hardcoded
BEADSIGHT_SIZE = (480,480) 
class ClipDataset(torch.utils.data.Dataset):
    """
    A dataset for training the CLIP model. This dataset will return a set of 
    images from a single episode, making sure they are at least min_distance apart. 
    The images are normalized and resized to the correct size.
    """
    def __init__(self, 
                 episode_ids: List[int], 
                 dataset_dir: str, 
                 camera_names: List[str], 
                 norm_stats: Dict[str, Union[float, np.ndarray]],
                 image_size: Tuple[int, int] = None, 
                 beadsight_size: Tuple[int, int] = BEADSIGHT_SIZE,
                 beadsight_horizon = 5,
                 min_distance = 10,
                 n_images = 7):
        
        super(ClipDataset).__init__()
        self.n_images = n_images
        self.min_distance = min_distance
        self.episode_ids = episode_ids # list of episode ids to use
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.image_size = image_size # image size in (H, W)
        self.n_cameras = len(camera_names)

        self.position_mean = norm_stats["qpos_mean"]
        self.position_std = norm_stats["qpos_std"]

        # image normalization for resnet. 
        self.image_normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.beadsight_horizon = beadsight_horizon

        # get the length of each episode
        self.episode_lengths = []
        for episode_id in self.episode_ids:
            dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
            with h5py.File(dataset_path, 'r') as root:
                self.episode_lengths.append(root.attrs['num_timesteps'])
                if self.image_size is None:
                    self.image_size = (root.attrs['image_height'], root.attrs['image_width'])
                if beadsight_size is None:
                    self.beadsight_size = (root.attrs['beadsight_height'], root.attrs['beadsight_width'])   
                    # these attrs don't currently exist
                
        # check that the episode lengths are long enough for the number of images and min_distance
        for length in self.episode_lengths:
            assert length >= n_images*min_distance*1.5, "To small of an episode length for the number of images and min_distance"


    def __len__(self):
        return len(self.episode_ids)
    

    def __getitem__(self, index):
        # gets a set of images from the selected episode, making sure they are at least min_distance apart
        timesteps = []
        while len(timesteps) < self.n_images:
            t = np.random.randint(0, self.episode_lengths[index])
            good_timestep = True
            for prev_t in timesteps:
                if abs(t - prev_t) < self.min_distance:
                    good_timestep = False

            if good_timestep:
                timesteps.append(t)

        # open the hdf5 file and get the images
        dataset_path = os.path.join(self.dataset_dir, f'episode_{self.episode_ids[index]}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            all_cam_images = []
            beadsight_images = []
            all_positions = []
            for timestep in timesteps:
                # get camera images
                timestep_cam_images = []
                
                for cam_name in self.camera_names:

                    if cam_name == "beadsight":
                        beadframes = []
                        if timestep < (self.beadsight_horizon+1):
                            for i in range(self.beadsight_horizon-timestep):
                                image = root[f'/observations/images/{cam_name}'][timestep]
                                # convert to tensor
                                image = torch.tensor(image, dtype=torch.float32)/255.0
                                image = torch.einsum('h w c -> c h w', image) # change to c h w
                                # normalize image
                                image = self.image_normalize(image)
                                beadframes.append(image)
                            for i in range(timestep):
                                image = root[f'/observations/images/{cam_name}'][i]
                                # convert to tensor
                                image = torch.tensor(image, dtype=torch.float32)/255.0
                                image = torch.einsum('h w c -> c h w', image) # change to c h w
                                # normalize image
                                image = self.image_normalize(image)
                                beadframes.append(image) 
                        else:
                            for i in range((timestep-self.beadsight_horizon)+1,timestep+1):
                                image = root[f'/observations/images/{cam_name}'][i]
                                # normalize image # ?? do we actually wanna?
                                image = torch.tensor(image, dtype=torch.float32)/255.0
                                image = torch.einsum('h w c -> c h w', image) # change to c h w
                                image = self.image_normalize(image)
                                beadframes.append(image)

                        beadcat = torch.concatenate(beadframes, axis=0)
                                                      

                    else:
                        image = root[f'/observations/images/{cam_name}'][timestep]
                        
                        # convert to tensor
                        image = torch.tensor(image, dtype=torch.float32)/255.0
                        image = torch.einsum('h w c -> c h w', image) # change to c h w

                        # normalize image
                        image = self.image_normalize(image)
                        timestep_cam_images.append(image)

                images = torch.stack(timestep_cam_images, axis=0)
                
                # get qpos and normalize
                position = root['observations/qpos'][timestep]
                position = (position - self.position_mean) / self.position_std

                # don't include the last element, which is the gripper
                position = torch.tensor(position[:3], dtype=torch.float32)

                all_cam_images.append(images)
                beadsight_images.append(beadcat)
                all_positions.append(position)

        return torch.stack(all_cam_images, axis=0), torch.stack(beadsight_images,axis=0), torch.stack(all_positions, axis=0)
    
    # Create helper get functions for evaluation
    def get_image(self, episode_idx, timestep, cam_name):
        # get an image from the hdf5 file and preprocess it.
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            image = root[f'/observations/images/{cam_name}'][timestep]
            
            # convert to tensor
            image = torch.tensor(image, dtype=torch.float32)/255.0
            image = torch.einsum('h w c -> c h w', image)

            # normalize image
            image = self.image_normalize(image)
            return image
        
    def get_position(self, episode_idx, timestep):
        # get position data from the hdf5 file and preprocess it.
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            position = root['observations/qpos'][timestep]
            position = (position - self.position_mean) / self.position_std
            position = torch.tensor(position[:3], dtype=torch.float32)
            return position

        
import torch.nn.functional as F
def clip_loss_non_vectorized(image_embeddings, gelsight_embeddings, target_matrix, logit_scale = 1.0, visualize = False):
    # Same as below, but not vectorized
    loss = torch.empty(image_embeddings.shape[2]).to(image_embeddings.device)
    visualizations = []
    for batch_idx in range(image_embeddings.shape[0]):
        image_targets = target_matrix
        gelsight_targets = target_matrix.T

        n_cameras = image_embeddings.shape[2]
        for i in range(n_cameras):
            image_logits = logit_scale * image_embeddings[batch_idx, :, i] @ gelsight_embeddings[batch_idx].T
            gelsight_logits = logit_scale * gelsight_embeddings[batch_idx] @ image_embeddings[batch_idx, :, i].T

            if visualize and batch_idx == 0:
                visualizations.append(image_logits.clone().detach().cpu().numpy()/logit_scale)

            image_loss = F.cross_entropy(image_logits, gelsight_targets)
            gelsight_loss = F.cross_entropy(gelsight_logits, image_targets)

            loss[i] = ((image_loss + gelsight_loss)/2.0).mean()

    return loss, visualizations



def clip_loss(image_embeddings:torch.Tensor, gelsight_embeddings:torch.Tensor, target_matrix:torch.Tensor, logit_scale = 1.0, visualize = False):
    """
    Calculate the loss for the CLIP model. The loss is calculated by taking the 
    dot product of the image embeddings and the gelsight embeddings (the
    embeddings are normalized in the forward pass). The dot product is then 
    scaled by logit_scale. The loss is calculated by taking the cross entropy
    loss between the dot product and the target matrix. The target matrix is
    the identity matrix. The loss is averaged over the batch and clip_N dimensions.
    image_embeddings: torch.Tensor of shape (batch, clip_N, camera, clip_dim). The image embeddings.
    gelsight_embeddings: torch.Tensor of shape (batch, clip_N, clip_dim). The gelsight embeddings.
    target_matrix: torch.Tensor of shape (clip_N, clip_N). The target matrix.
    logit_scale: float. The scale to apply to the dot product. (default: 1.0)"""

    n_cameras = image_embeddings.shape[2]
    batch_size = image_embeddings.shape[0]

    visualizations = []
    image_embeddings = image_embeddings.permute(0, 2, 1, 3) # batch, camera, clip_N, clip_dim
    gelsight_embeddings = gelsight_embeddings.unsqueeze(1) # batch, 1, clip_N, clip_dim
    image_logits = logit_scale * image_embeddings @ gelsight_embeddings.permute(0, 1, 3, 2) # dot product by multiplying by transpose
    gelsight_logits = logit_scale * gelsight_embeddings @ image_embeddings.permute(0, 1, 3, 2)

    # if visualize, save the average softmax map for each camera (only for the first batch)
    if visualize:
        visualizations = image_logits[0].clone().detach().cpu().numpy()/logit_scale
    
    # flatten the batch and camera dimensions, then calculate the loss
    image_logits = image_logits.flatten(0, 1)
    gelsight_logits = gelsight_logits.flatten(0, 1)

    # need to make the target matrix B, N, N
    image_loss = F.cross_entropy(image_logits, target_matrix.repeat(image_logits.shape[0], 1, 1), reduce=False).mean(dim=1)
    gelsight_loss = F.cross_entropy(gelsight_logits, target_matrix.T.repeat(gelsight_logits.shape[0], 1, 1), reduce=False).mean(dim=1)


    # reshape the loss to be B, N_cameras
    image_loss = image_loss.view(batch_size, n_cameras)
    gelsight_loss = gelsight_loss.view(batch_size, n_cameras)

    loss = ((image_loss + gelsight_loss)/2.0).mean(dim=0)

    # return the per-camera loss

    return loss, visualizations


from tqdm import tqdm

def clip_pretraining(train_loader: DataLoader,
                     test_loader: DataLoader,
                     device: torch.device,
                     save_dir: str,
                     save_freq: int = 100,
                     plot_freq: int = 50,
                     n_epochs: int = 1000,
                     clip_dim: int = 512,
                     features_per_group: int = 16,
                     resnet_lr: float = 1e-5,
                     projection_lr: float = 1e-4):
    
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    
    if save_dir[-1] == '/':
        save_dir = save_dir[:-1]

    # get the camera, beadsight, and state dimensions from the dataset
    dataset:ClipDataset = train_loader.dataset
    n_cameras = dataset.n_cameras -1 # accounts for beadsight
    state_size = 3

    # get resnet models for each camera
    # get a resnet18 model
    vision_encoder = modified_resnet18().to(device)

    # create a projection head
    vision_projection = ClipProjectionHead(out_dim=clip_dim).to(device)

    # get a resnet18 model for beadsight
    beadsight_encoder = get_beadsight_clip().to(device)

    # create a projection head, conditioned on state
    beadsight_projection = ClipProjectionHead(out_dim=clip_dim, conditioning_dim=state_size).to(device)

    # create a learnable parameter for the logit scale and add it to the optimizer.

    optim_params = [{"params": beadsight_encoder.parameters(), "lr": resnet_lr},
                    {"params": beadsight_projection.parameters(), "lr": projection_lr},
                    {"params": vision_encoder.parameters(), "lr": resnet_lr},
                    {"params": vision_projection.parameters(), "lr": projection_lr}]

    print('optim_params:', optim_params)

    optimizer = torch.optim.Adam(optim_params)
    
    training_losses = np.empty([n_epochs, n_cameras])
    testing_losses = np.empty([n_epochs, n_cameras])
    for epoch in tqdm(range(n_epochs)):
    # train the model
        training_loss = np.zeros(n_cameras)

        beadsight_encoder.train()
        beadsight_projection.train()
        vision_encoder.train()
        vision_projection.train()
        for batch_idx, (images, beadsight, position) in enumerate(train_loader):
            images = images.to(device)
            beadsight = beadsight.to(device)
            position = position.to(device)

            # forward pass
            
            batch_size = images.shape[0]
            clip_N = images.shape[1]
            # images are in form batch, clip_N, camera, c, h, w. We want to flatten the batch and camera dimensions
            images = images.view(-1, images.shape[3], images.shape[4], images.shape[5])
            image_embeddings = vision_projection(vision_encoder(images))
            
            # now reshape the image_embeddings to be batch, clip_N, camera, clip_dim
            image_embeddings = image_embeddings.view(batch_size, clip_N, n_cameras, clip_dim)

            # flatten the batch and clip_N dimensions
            beadsight = beadsight.view(-1, beadsight.shape[2], beadsight.shape[3], beadsight.shape[4])
            position = position.view(-1, position.shape[2])
            beadsight_embeddings = beadsight_projection(beadsight_encoder(beadsight), position)

            # reshape the beadsight_embeddings to be batch, clip_N, clip_dim
            beadsight_embeddings = beadsight_embeddings.view(batch_size, clip_N, clip_dim)

            # calculate target matrix
            target_matrix = torch.eye(clip_N).to(device)

            # calculate loss - vector of per-camera losses
            if batch_idx == 0 and epoch%plot_freq == 0: # visualize the first batch in each epoch
                loss, avg_softmax_maps = clip_loss(image_embeddings, beadsight_embeddings, target_matrix, visualize=True)
                try:
                    for cam_num, softmax_map in enumerate(avg_softmax_maps):
                        plt.figure()
                        plt.imshow(softmax_map)
                        plt.colorbar()
                        plt.title(f'Average Softmax Map, Epoch {epoch}, Cam {cam_num} - Train')
                        plt.savefig(f'{save_dir}/graphs/epoch_{epoch}_cam_{cam_num}_train.png')
                        plt.close()
                except:
                    print('Error in train plots')
                    raise
            else:
                loss, _ = clip_loss(image_embeddings, beadsight_embeddings, target_matrix, visualize=False)
            training_loss += loss.clone().detach().cpu().numpy()
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

        training_losses[epoch] = training_loss/len(train_loader)

        # test the model
        beadsight_encoder.eval()
        beadsight_projection.eval()
        vision_encoder.eval()
        vision_projection.eval()

        test_loss = np.zeros(n_cameras)
        with torch.no_grad():
            for batch_idx, (images, beadsight, position) in enumerate(test_loader):
                images = images.to(device)
                beadsight = beadsight.to(device)
                position = position.to(device)

                # forward pass
                batch_size = images.shape[0]
                clip_N = images.shape[1]
                # images are in form batch, clip_N, camera, c, h, w. We want to flatten the batch and camera dimensions
                images = images.view(-1, images.shape[3], images.shape[4], images.shape[5])
                image_embeddings = vision_projection(vision_encoder(images))
                
                # now reshape the image_embeddings to be batch, clip_N, camera, clip_dim
                image_embeddings = image_embeddings.view(batch_size, clip_N, n_cameras, clip_dim)

                # flatten the batch and clip_N dimensions
                beadsight = beadsight.view(-1, beadsight.shape[2], beadsight.shape[3], beadsight.shape[4])
                position = position.view(-1, position.shape[2])
                beadsight_embeddings = beadsight_projection(beadsight_encoder(beadsight), position)

                # reshape the beadsight_embeddings to be batch, clip_N, clip_dim
                beadsight_embeddings = beadsight_embeddings.view(batch_size, clip_N, clip_dim)

                # calculate target matrix
                target_matrix = torch.eye(clip_N).to(device)

                # calculate loss - vector of per-camera losses
                            # calculate loss - vector of per-camera losses
                if batch_idx == 0 and epoch%plot_freq == 0: # visualize the first batch in each epoch
                    loss, avg_softmax_maps = clip_loss(image_embeddings, beadsight_embeddings, target_matrix, visualize=True)
                    try:
                        for cam_num, softmax_map in enumerate(avg_softmax_maps):
                            plt.figure()
                            plt.imshow(softmax_map)
                            plt.colorbar()
                            plt.title(f'Average Softmax Map, Epoch {epoch}, Cam {cam_num} - Test')
                            plt.savefig(f'{save_dir}/graphs/epoch_{epoch}_cam_{cam_num}_test.png')
                            plt.close()
                    except:
                        print('Error in test plots')
                        raise
                else:
                    loss, _ = clip_loss(image_embeddings, beadsight_embeddings, target_matrix, visualize=False)
                test_loss += loss.clone().detach().cpu().numpy()
        testing_losses[epoch] = test_loss/len(test_loader)


        # plot the training and testing losses
        if epoch%plot_freq == 0:
            plt.figure()
            for i in range(n_cameras):
                plt.plot(training_losses[:epoch+1, i], label=f'camera {i+1} train', c=f'C{i}')
                plt.plot(testing_losses[:epoch+1, i], label=f'camera {i+1} test', linestyle='dashed', c=f'C{i}')
            plt.legend(loc='best')
            plt.title(f'Training and Testing Loss - Epoch {epoch+1}/{n_epochs}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(f'{save_dir}/graphs/training_loss.png')
            plt.close()

        # save the losses as a np file
        np.save(f'training_losses.npy', training_losses)
        np.save(f'testing_losses.npy', testing_losses)

        # save the models
        if (epoch+1) % save_freq == 0:
            torch.save(vision_encoder.state_dict(), f'{save_dir}/epoch_{epoch}_vision_encoder.pth')
            torch.save(vision_projection.state_dict(), f'{save_dir}/epoch_{epoch}_vision_projection.pth')
            torch.save(beadsight_encoder.state_dict(), f'{save_dir}/epoch_{epoch}_beadsight_encoder.pth')
            torch.save(beadsight_projection.state_dict(), f'{save_dir}/epoch_{epoch}_beadsight_projection.pth')
   

def run_clip_pretraining(n_epochs, device):
    from utils import get_norm_stats
    num_episodes = 106 #TODO: Change
    dataset_dir = "/home/selamg/processed_data"
    save_dir = "/home/selamg/clipmodels"
    camera_names = ['1', '2', '3', '4', '5', '6', 'beadsight']
    norm_stats = get_norm_stats(dataset_dir, num_episodes, use_existing=True)
    batch_size_train = 2
    batch_size_test = 2
    n_clip_images = 7
    min_distance = 10
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    train_dataset = ClipDataset(train_indices, dataset_dir, camera_names, norm_stats, n_images=n_clip_images, min_distance=min_distance)
    test_dataset = ClipDataset(val_indices, dataset_dir, camera_names, norm_stats, n_images=n_clip_images, min_distance=min_distance)

    if device == torch.device("cuda"):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=10, prefetch_factor=10, pin_memory_device='cuda')
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True, pin_memory=True, num_workers=10, prefetch_factor=10, pin_memory_device='cuda')
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

    # create directory to save models and plots
    # get all folders in the clip_models directory
    ns = [-1]
    for folder in os.listdir(save_dir):
        ns.append(int(folder))

    n = max(ns) + 1
    os.makedirs(f'{save_dir}/{n}')
    os.makedirs(f'{save_dir}/{n}/graphs')

    # save run stats:
    with open(f'{save_dir}/{n}/run_stats.txt', 'w') as f:
        f.write(f'num_episodes: {num_episodes}\n')
        f.write(f'dataset_dir: {dataset_dir}\n')
        f.write(f'camera_names: {camera_names}\n')
        f.write(f'norm_stats: {norm_stats}\n')
        f.write(f'batch_size_train: {batch_size_train}\n')
        f.write(f'batch_size_test: {batch_size_test}\n')
        f.write(f'n_clip_images: {n_clip_images}\n')
        f.write(f'min_distance: {min_distance}\n')
        f.write(f'train_indices: {train_indices}\n')
        f.write(f'val_indices: {val_indices}\n')
        
    clip_pretraining(train_dataloader, test_dataloader, device, save_dir=f'{save_dir}/{n}', clip_dim=512, features_per_group=16, n_epochs=n_epochs)


def replot_loss_graph(training_losses, testing_losses):
    """
    Plot the training and testing losses from the saved npy files.
    Applies a running average to smooth the losses.
    """
    # training_losses: N X cameras
    # testing_losses: N X cameras
    from matplotlib import pyplot as plt

    total_train = training_losses.mean(axis=1)
    total_test = testing_losses.mean(axis=1)

    # smooth the losses (running average)
    window_size = 10
    smooth_train =  np.zeros_like(total_train)
    smooth_test =  np.zeros_like(total_test)
    for i in range(len(total_train)):
        if i < window_size:
            smooth_train[i] = total_train[:i].mean()
            smooth_test[i] = total_test[:i].mean()
        else:
            smooth_train[i] = total_train[i-window_size:i].mean()
            smooth_test[i] = total_test[i-window_size:i].mean()


    plt.figure()
    plt.plot(smooth_train, label=f'Training loss', c='r')
    plt.plot(smooth_test, label=f'Testing loss', c='b')
    plt.legend(loc='best')
    plt.title(f'Training and Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":
    run_clip_pretraining(1501, device='cuda:0')
    
    # training_losses = np.load('/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/clip_models/11/epoch1450-training_losses.npy')[:1450]
    # testing_losses = np.load('/home/aigeorge/research/TactileACT/data/camera_cage_new_mount/clip_models/11/epoch1450-testing_losses.npy')[:1450]
    # replot_loss_graph(training_losses, testing_losses)