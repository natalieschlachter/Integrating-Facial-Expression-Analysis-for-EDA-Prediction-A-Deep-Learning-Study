import os
import torch
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
        data = torch.load(file_path)   # => { 'tensor': (3, H, W), 'bvp': float, 'eda': float }

        image_2d = data['tensor']     # shape (3, H, W)
        # Insert a dummy time dimension => (3, 1, H, W)
        image_3d = image_2d.unsqueeze(1)

        eda = torch.tensor(data['eda'], dtype=torch.float32)
        bvp = torch.tensor(data['bvp'], dtype=torch.float32)

        return image_3d, eda, bvp


class DeepPhysDataPreprocessor:
    def create_dataloaders(self, train_dir, val_dir, test_dir, batch_size):
        # Collect single-frame file paths
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

        # Create single-frame datasets (T=1)
        train_dataset = SingleFrameDataset3D(train_files)
        val_dataset   = SingleFrameDataset3D(val_files)
        test_dataset  = SingleFrameDataset3D(test_files)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
