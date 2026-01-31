from torchvision import datasets, transforms
from torch.utils.data import Dataset


class EMNISTDataset(Dataset):
    def __init__(self, 
                 root='./',
                 is_download=True,
                 image_set='train',
                 emnist_split='byclass',
                 transform=None):
        
        self.root = root
        self.is_download = is_download
        self.emnist_split = emnist_split
        self.image_set = image_set

        # If no transform provided, use a default one
        if transform is None:
            self.transform = self.custom_transform()

        self.dataset = datasets.EMNIST(
            root=self.root,
            split=self.emnist_split,
            train=True if self.image_set == 'train' else False,
            download=self.is_download,
            transform=self.transform
        )

        # Expose useful properties
        self.classes = self.dataset.classes
        self.targets = self.dataset.targets
    
        print(f"Successfully loaded EMNIST dataset ({self.image_set} split) with {len(self)} samples.")

    def custom_transform(self):
        """Defines the default transformation pipeline."""
        transform = transforms.Compose([
            # transforms.RandomRotation([-90, -90]),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.4),
            transforms.RandomAdjustSharpness(sharpness_factor=0.7, p=0.4),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            # transforms.Lambda(lambda x: x.transpose(1, 2))
        ])

        return transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Retrieves a sample and its label by index."""
        return self.dataset[idx]

