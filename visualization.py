import matplotlib.pyplot as plt
import numpy as np    
import os

class DebugController:
    def __init__(self, print=False, plot=False, epoch=0, batch=0, dataset='train', visualizations_dir = None):
        self.print = print
        self.plot = plot
        self.epoch = epoch
        self.batch = batch
        self.dataset = dataset
        self._visualizations_dir = visualizations_dir
        self.action_qpos_normalizer = None

    @property
    def visualizations_dir(self):
        if self._visualizations_dir is None:
            self._visualizations_dir = self.gen_visualizations_dir()
            return self._visualizations_dir

        else:
            return self._visualizations_dir
        
    @visualizations_dir.setter
    def visualizations_dir(self, value):
        if not os.path.exists(value):
            os.makedirs(value)
        self._visualizations_dir = value

    def gen_visualizations_dir(self) -> str:
        n = 0
        while os.path.exists(f'visualizations/{n}'):
            n += 1
        os.makedirs(f'visualizations/{n}')
        return f'visualizations/{n}'


debug = DebugController()

def visualize(images, qpos, actions, ground_truth=None):
    """
    images = [HxWxC] of length num cams + beadsight
    for image in images assert image.rank=3 (try/check) determines that this is a 3D array
    qpos = 4,
    actions = nx4
    ground_truth = nx4
    """
    #TODO: Unnormalize images
    global debug
    if debug.plot:
        save = True
    else:
        save = False
    # Create a figure and axes
    fig = plt.figure(figsize=(15, 15), layout='tight')
    gs = fig.add_gridspec(15,15,wspace=0.01)
    axs_left = fig.add_subfigure(gs[0:13,0:7]).subplots(len(images)-1,1)
    # axs_left = subfigs_top[0].subplots(len(images), 1)

    if len(images)>1:
        for i, image in enumerate(images):
            # print(image.shape)
            if image.shape[2] > 3: 
                # put an image that is 5 resized beadsight frames into plot?
                if image.shape[2]%3 != 0:
                    raise ValueError("something is really fked with ur beadsight image channels")
                num_frames = int(image.shape[2]/3.0)
                
                axs_bot = fig.add_subfigure(gs[13:15,:]).subplots(1,num_frames)
                start = 0
                for i in range(num_frames):
                    frame = image[:,:,start:((i+1)*3)]
                    start += 3
                    axs_bot[i].imshow(frame)
            else:
                axs_left[i].imshow(image)
    else:
        axs_left.imshow(images[0])
    
    # Make a 3D scatter plot of the actions in the right subplot. Use cmaps to color the points based on the index
    c = np.arange(len(actions))
    ax2 = fig.add_subplot(gs[0:13,7:15], projection='3d')
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

    
    
    if not save:
        plt.show()
    else:
        fig.savefig(f'{debug.visualizations_dir}/epoch_{debug.epoch}-{debug.dataset}.png')
        plt.close(fig)


# modified a bunch
# def visualize_data(image_data, qpos_data, action_pred_data, action_gt_data=None):
#     # spec: all inputs already unnormalized
#     # (except images)
    
#     from matplotlib import pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#     import numpy as np

#     images = [img.clone().detach().cpu().numpy() for img in image_data]
#     qpos = qpos_data.clone().detach().cpu().numpy()
#     actions_pred = action_pred_data.clone().detach().cpu().numpy()
#     #is_pad = is_pad_data.clone().detach().cpu().numpy()

#     # unnormlize actions and qpos
#     #qpos, actions_pred = debug.action_qpos_normalizer.unnormalize(qpos_norm, actions_pred)

#     if action_gt_data is not None:
#         actions_gt = action_gt_data.clone().detach().cpu().numpy()
#         #_, actions_gt = debug.action_qpos_normalizer.unnormalize(qpos_norm, actions_gt)

#     # Create a figure and axes
#     fig = plt.figure(figsize=(10, 10), layout='tight')
#     subfigs = fig.subfigures(1, 2, wspace=0.07)

#     axs_left = subfigs[0].subplots(len(images), 1)
#     for i, image in enumerate(images):
#         #axs_left[i].imshow(image.transpose(1, 2, 0))     
#         print(image.shape)
#         axs_left[i].imshow(image)
#         #images also already shifted 

#     # Make a 3D scatter plot of the actions in the right subplot. Use cmaps to color the points based on the index
#     c = np.arange(len(actions_pred))
#     ax2 = subfigs[1].add_subplot(111, projection='3d')
#     # ax2.scatter(actions[:, 0], actions[:, 1], actions[:, 2], c='b', marker='o')
#     sc = ax2.scatter(actions_pred[:, 0], actions_pred[:, 1], actions_pred[:, 2], c=c, cmap='viridis', marker='x')
#     if action_gt_data is not None:
#         # ax2.scatter(output[:, 0], output[:, 1], output[:, 2], c='g', marker='x')
#         ax2.scatter(actions_gt[:, 0], actions_gt[:, 1], actions_gt[:, 2], c=c, cmap = 'viridis', marker='o')
#     ax2.scatter(qpos[0], qpos[1], qpos[2], c='r', marker='o')
#     ax2.set_xlabel('X')
#     ax2.set_ylabel('Y')
#     ax2.set_zlabel('Z')
#     ax2.set_title('Actions and Qpos')
#     cbar = fig.colorbar(sc, ax=ax2, label='Time', shrink=0.5)

#     # Set the axis limits
#     center = np.array([0.5, 0, 0.2])
#     radius = 0.15
#     ax2.set_xlim(center[0] - radius, center[0] + radius)
#     ax2.set_ylim(center[1] - radius, center[1] + radius)
#     ax2.set_zlim(center[2] - radius, center[2] + radius)

#     # Show or save the figure
#     fig.suptitle(f'Epoch {debug.epoch}')
#     fig.savefig(f'{debug.visualizations_dir}/epoch_{debug.epoch} - {debug.dataset}.png')
#     plt.close(fig)