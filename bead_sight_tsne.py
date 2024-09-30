from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import glob 
import torch
from torchvision import transforms
from tqdm import tqdm
import h5py
from torch import nn

def plot_tsne(train_encoding:np.ndarray, test_encoding):
    n = train_encoding.shape[0]
    
    tsne = TSNE(n_components=2, random_state=10)

    # fit the t-SNE model to the latent vectors
    all_latent_vectors = np.concatenate([train_encoding, test_encoding], axis=0)

    embedded = tsne.fit_transform(all_latent_vectors)
    
    train_embeddings = embedded[:n]
    test_embeddings = embedded[n:]

    plt.figure(figsize=(8, 5))
    plt.scatter(train_embeddings[:, 0], train_embeddings[:, 1], marker='x', s=50, alpha=0.25, label='Training Embeddings')
    plt.scatter(test_embeddings[:, 0], test_embeddings[:, 1], marker = 'o', s=50, alpha=0.25, label='Testing Embeddings')

    plt.title('t-SNE Visualization of BeadSight Latent Vectors')
    # put legend outside of plot
    plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.colorbar(label='Timestamp')
    plt.grid(True)
    plt.tight_layout() 


def encode_beadsight_images(beadsight_net, beadsight_recording, batch_size = 25):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    all_images = []
    for t in range(5, beadsight_recording.shape[0]+1):
        images = torch.tensor(beadsight_recording[t-5:t], dtype=torch.float32)/255.0
        images = torch.einsum('t h w c -> t c h w', images) # change to c h w
        images = normalizer(images)
        all_images.append(torch.concatenate([image for image in images], dim = 0))

    all_encodings = []
    for b_idx in range(int(np.ceil(len(all_images)/batch_size))):
        batch_images = torch.stack(all_images[b_idx*batch_size: (b_idx + 1)*batch_size]).cuda()
        all_encodings.append(beadsight_net(batch_images).cpu().detach().numpy())

    return np.concatenate(all_encodings, axis=0)


if __name__ == "__main__":
    weights_dir = "/home/selamg/beadsight/data/ssd/weights/clip_epoch3500_23-56-01_2024-06-01"
    model_dict = torch.load(weights_dir)
    nets = nn.ModuleDict(model_dict).cuda()
    beadsight_net = nets['beadsight_encoder']
    all_beadsight_files = []
    for filename in glob.iglob('/home/selamg/beadsight/data/ssd/experiment_results/diffusion/pretrained_both/**', recursive=True):
        if "beadsight_obs.npy" in filename:
            all_beadsight_files.append(filename)

    test_encodings = []
    for filename in tqdm(all_beadsight_files):
        beadsight_data = np.load(filename)
        test_encodings.append(encode_beadsight_images(beadsight_net, beadsight_data))

    test_encodings = np.concatenate(test_encodings, axis=0)

    # train encodings:
    train_encodings = []
    for i in range(105):
        with h5py.File(f"/home/selamg/beadsight/data/ssd/processed_data/episode_{i}", 'r') as root:
            beadsight_data = root[f'/observations/images/beadsight'][()]
            train_encodings.append(encode_beadsight_images(beadsight_net, beadsight_data))
    train_encodings = np.concatenate(train_encodings, axis=0)

    plot_tsne(train_encodings, test_encodings)

    

