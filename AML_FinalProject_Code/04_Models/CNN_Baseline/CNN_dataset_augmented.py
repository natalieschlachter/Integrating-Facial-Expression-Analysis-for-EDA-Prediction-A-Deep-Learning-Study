import os
import glob
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF  
import matplotlib.pyplot as plt

class WindowedUBFCPhysDataset(Dataset):
    def __init__(self, root_dir, window_size=16, stride=16, transform=None, p=0.5, isTrain=False):
       
        super().__init__()
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride = stride
        self.isTrain = isTrain
        self.transform = transform if isTrain else None
        self.p = p  
        
        pattern = os.path.join(root_dir, '**', '*.pt')
        all_files = sorted(glob.glob(pattern, recursive=True))
        
        if not all_files:
            print(f"[Warning] No .pt files found in {root_dir}!")
        
        all_files = sorted(all_files, key=self.parse_frame_number)
        
        self.filepaths = all_files
        
        self.indices = []
        
        N = len(self.filepaths)
        i = 0
        while i + window_size <= N:  
            self.indices.append(i)
            i += stride  

    @staticmethod
    def parse_frame_number(filename):
        basename = os.path.basename(filename)          
        parts = basename.split('_')                    
        frame_str = parts[-1]                          
        frame_num = int(frame_str.replace('.pt', ''))  
        return frame_num

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_size
        
        window_frames = []
        window_bvp = []
        window_eda = []
        
        for fpath in self.filepaths[start_idx:end_idx]:
            data = torch.load(fpath) 
            
            frame = data['tensor']
            bvp = data['bvp']
            eda = data['eda']
            
            if not isinstance(bvp, torch.Tensor):
                bvp = torch.tensor(bvp, dtype=torch.float32)
            if not isinstance(eda, torch.Tensor):
                eda = torch.tensor(eda, dtype=torch.float32)
            
            if self.transform:
                frame = self.transform(frame)
            
            window_frames.append(frame)
            window_bvp.append(bvp)
            window_eda.append(eda)
        
        window_frames = torch.stack(window_frames, dim=0)  
        window_bvp = torch.stack(window_bvp, dim=0).squeeze()  
        window_eda = torch.stack(window_eda, dim=0).squeeze()  
        
        if self.isTrain and random.random() < self.p:
            window_frames, window_bvp, window_eda = self.apply_augmentations(window_frames, window_bvp, window_eda)
        
        return window_frames, window_bvp, window_eda

    def apply_augmentations(self, frames, bvp, eda):

        if random.random() < self.p:
            frames = torch.flip(frames, dims=[-1])

        if random.random() < self.p:
            angle = random.uniform(-10, 10)
            frames = self.rotate_frames(frames, angle)
        
        if random.random() < self.p:
            brightness_factor = random.uniform(0.8, 1.2)
            frames = frames * brightness_factor
            frames = torch.clamp(frames, -1, 1)
        
        if random.random() < self.p:
            noise = torch.randn_like(frames) * 0.02
            frames = frames + noise
            frames = torch.clamp(frames, -1, 1)

        return frames, bvp, eda

    @staticmethod
    def rotate_frames(frames, angle):
     
        rotated_frames = []
        for frame in frames:
            rotated_frame = TF.rotate(frame, angle) 
            rotated_frames.append(rotated_frame)
        return torch.stack(rotated_frames, dim=0)

    @staticmethod
    def temporal_smoothing(signal, kernel_size, padding):
        signal = signal.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones(1, 1, kernel_size) / kernel_size
        smoothed = F.conv1d(signal, kernel, padding=padding)
        smoothed = smoothed.squeeze(0).squeeze(0)
        return smoothed
