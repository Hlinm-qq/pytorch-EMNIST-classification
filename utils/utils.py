import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import math

def set_random_seed(SEED = 42):
    # Python's built-in random module
    random.seed(SEED)
    # NumPy's random number generator
    np.random.seed(SEED)
    # PyTorch on the CPU
    torch.manual_seed(SEED)

    # GPU operations (if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        # For multi-GPU
        torch.cuda.manual_seed_all(SEED)
        # Enable deterministic behavior for CuDNN (slower, but necessary for full reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def worker_init_fn(worker_id):
    """Function to initialize the random seed for each DataLoader worker."""
    # The DataLoader's base seed + worker_id gives a unique seed for each worker
    # worker_seed = SEED % 2**32 + worker_id
    # worker_seed = SEED
    # np.random.seed(worker_seed)
    # random.seed(worker_seed)

    # seed = torch.initial_seed() % 2**32
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed % 2**32  # 32-bit safe seed
    np.random.seed(seed)
    random.seed(seed)

def set_default_device():
    # checks our available computing resources
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU device:", torch.cuda.get_device_name(0))
        print("GPU count:", torch.cuda.device_count())
    print(torch.cuda.is_available())  # True if GPU detected

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_emnist_mapping(filepath):
    """
    Reads an EMNIST mapping file (e.g., 'emnist-byclass-mapping.txt' or 'emnist-letters-mapping.txt')
    and returns a string of all label characters in order.
    """
    labels = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                # format: index unicode
                _, unicode_val = parts
                labels.append(chr(int(unicode_val)))
            elif len(parts) == 3:
                # format: index upper_unicode lower_unicode
                _, unicode_val, lower_val = parts
                labels.append(chr(int(lower_val)))
            else:
                continue
    
    print("".join(labels))

    return "".join(labels)

def visualize_sample(dataset, 
                     save_path='emnist_sample_visualization.png', 
                     num_samples=11, 
                     split='byclass',
                     mapping="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                     ):
    
    ncol_ = int(num_samples*0.5)
    nrow_ = math.ceil(num_samples / ncol_)
    fig, axes = plt.subplots(nrow_, ncol_, figsize=(6 * ncol_, 3 * nrow_))
    fig.suptitle(f"{num_samples} EMNIST {split} Samples", fontsize=16)
    axes = axes.flatten()

    for i in range(num_samples):
        img, label_idx = dataset[i] 

        if split == 'letters':
            label_idx -= 1

        np_image = img.squeeze().numpy() 
        try:
            # Look up the character using the index
            label_char = f"Label: {mapping[label_idx]}"
        except IndexError:
            # Fallback if the index is outside the mapping length
            label_char = f"Label index: {label_idx}"

        # Plot the image on the corresponding subplot
        axes[i].imshow(np_image, cmap='grey')
        axes[i].set_title(label_char, fontsize=12)
        axes[i].axis('off')

    for j in range(num_samples, nrow_ * ncol_):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0,0,1,0.98])
    plt.savefig(save_path)
    print(f"Visualization saved successfully to: {save_path}")

    # Close the figure to free memory
    plt.close(fig)

