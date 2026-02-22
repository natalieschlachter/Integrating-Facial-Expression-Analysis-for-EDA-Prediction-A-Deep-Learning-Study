import torch
import torchvision.transforms.functional as F
import random
import os
from torch.utils.data import Dataset, DataLoader




class SingleFrameDataset3D(Dataset):
    """
    This dataset loads single-frame .pt files where each file has:
      - 'tensor': shape (3, H, W)
      - 'eda': float
      - 'bvp': float
    We then reshape the image to (3, 1, H, W) to create a T=1 dimension
    so that 3D convolutions still work.
    """
    def __init__(self, file_paths):
        self.file_paths = sorted(file_paths)
        print(f"Initializing SingleFrameDataset3D with {len(self.file_paths)} files.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path)  # => { 'tensor': (3, H, W), 'bvp': float, 'eda': float }

        image_2d = data['tensor']  # shape (3, H, W)
        # Insert a dummy time dimension => (3, 1, H, W)
        image_3d = image_2d.unsqueeze(1)

        eda = torch.tensor(data['eda'], dtype=torch.float32)
        bvp = torch.tensor(data['bvp'], dtype=torch.float32)

        return image_3d, eda, bvp


class PhysioAugmentation:
    """
    Applies random augmentations to the input tensor with a probability of p.
    Augmentations include:
      - Random horizontal flip
      - Random rotation
      - Random brightness adjustment
      - Random noise injection
      - Random temporal smoothing (if T > 1)
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor, eda, bvp):
        # Random horizontal flip
        if random.random() < self.p:
            tensor = torch.flip(tensor, [-1])  

        # Random rotation (-10 to 10 degrees)
        if random.random() < self.p:
            angle = random.uniform(-10, 10)
            tensor = F.rotate(tensor, angle)

        # Random brightness adjustment
        if random.random() < self.p:
            brightness_factor = random.uniform(0.8, 1.2)
            tensor = tensor * brightness_factor
            tensor = torch.clamp(tensor, -1, 1)

        # Random noise injection
        if random.random() < self.p:
            noise = torch.randn_like(tensor) * 0.02
            tensor = tensor + noise
            tensor = torch.clamp(tensor, -1, 1)

        # Random temporal smoothing (conv1d along temporal axis)
        if random.random() < self.p and tensor.size(1) > 1:  
            kernel_size = 3
            kernel = torch.ones(1, 1, kernel_size) / kernel_size
            tensor = torch.nn.functional.conv1d(
                tensor.unsqueeze(0), kernel, padding=kernel_size // 2
            ).squeeze(0)

        return tensor, eda, bvp


class AugmentedDataset(Dataset):
    """
    Wraps an existing dataset and applies augmentation using the given transform.
    """
    def __init__(self, original_dataset, transform):
        self.dataset = original_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tensor, eda, bvp = self.dataset[idx]

        # Apply augmentation only to the tensor (image input)
        augmented_tensor, _, _ = self.transform(tensor, eda, bvp)

        # Return the augmented tensor and the original labels (eda and bvp)
        return augmented_tensor, eda, bvp


class DeepPhysDataPreprocessorWithAugment:
    """
    Creates DataLoaders for training, validation, and testing datasets.
    Applies augmentations only to the training dataset.
    """
    def create_dataloaders_with_augment(self, train_dir, val_dir, test_dir, batch_size):
        
        train_files = [
            os.path.join(train_dir, folder, file)
            for folder in sorted(os.listdir(train_dir))
            for file in os.listdir(os.path.join(train_dir, folder))
            if file.endswith('.pt')
        ]
        val_files = [
            os.path.join(val_dir, folder, file)
            for folder in sorted(os.listdir(val_dir))
            for file in os.listdir(os.path.join(val_dir, folder))
            if file.endswith('.pt')
        ]
        test_files = [
            os.path.join(test_dir, folder, file)
            for folder in sorted(os.listdir(test_dir))
            for file in os.listdir(os.path.join(test_dir, folder))
            if file.endswith('.pt')
        ]

        print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")

        # Create single-frame datasets
        train_dataset = SingleFrameDataset3D(train_files)
        val_dataset = SingleFrameDataset3D(val_files)
        test_dataset = SingleFrameDataset3D(test_files)

        # Initialize augmentation for the training dataset
        augmentation = PhysioAugmentation(p=0.7)
        train_dataset = AugmentedDataset(train_dataset, augmentation)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader