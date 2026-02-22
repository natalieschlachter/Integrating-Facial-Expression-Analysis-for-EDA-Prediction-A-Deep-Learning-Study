import os
import glob
import torch
from torch.utils.data import Dataset

def parse_frame_number(filename):
    """
    Extract the frame index from something like 's1_T1_frame_0001.pt'.
    We'll assume the last underscore part is '0001.pt' or similar.
    """
    basename = os.path.basename(filename)          # 's1_T1_frame_0001.pt'
    parts = basename.split('_')                    # ['s1','T1','frame','0001.pt']
    frame_str = parts[-1]                          # '0001.pt'
    frame_num = int(frame_str.replace('.pt',''))   # 1
    return frame_num

class WindowedUBFCPhysDataset(Dataset):
    def __init__(self, root_dir, window_size=16, stride=16, transform=None):
        """
        Args:
            root_dir (str): e.g. 'DataSet/Train'.
            window_size (int): Number of frames per window.
            stride (int): Step size between consecutive windows.
            transform (callable, optional): Applied to each frame (image) tensor.
        """
        super().__init__()
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        
        # 1) Gather all .pt paths recursively
        pattern = os.path.join(root_dir, '**', '*.pt')
        all_files = sorted(glob.glob(pattern, recursive=True))
        
        if not all_files:
            print(f"[Warning] No .pt files found in {root_dir}!")
        
        # 2) Group by subject/trial *if* you want to avoid crossing boundaries
        #    If you want to keep it simple and treat them all as one big list, skip this step.
        #    We'll assume each subfolder is 'sX', inside that T1 / T2 / T3, etc.
        
        # Example: just keep them all in one big list, but we’ll make sure to sort by frame index:
        # (In practice, you might handle each subject/trial separately to avoid mixing them.)
        
        # Sort by frame number extracted from the filename
        all_files = sorted(all_files, key=parse_frame_number)
        
        self.filepaths = all_files
        
        # 3) Build the "window starts"
        #    We'll create a list of indices (start_idx) that define each sequence window
        self.indices = []
        
        N = len(self.filepaths)
        i = 0
        while i + window_size <= N:  # can fit a full window
            self.indices.append(i)
            i += stride  # move by stride

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Return a window of length 'window_size':
          frames: (window_size, 3, H, W)
          bvp:    (window_size,)
          eda:    (window_size,)
        """
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_size
        
        # Collect the frames/BVP/EDA
        window_frames = []
        window_bvp = []
        window_eda = []
        
        # For each time step in this window
        for fpath in self.filepaths[start_idx:end_idx]:
            data = torch.load(fpath)  # {'frame':..., 'bvp':..., 'eda':...}
            
            frame = data['tensor']  # shape: (3, H, W)
            bvp   = data['bvp']    # shape: (1,) or scalar
            eda   = data['eda']    # shape: (1,) or scalar
            if not isinstance(bvp, torch.Tensor):
                bvp = torch.tensor(bvp, dtype=torch.float32)

            if not isinstance(eda, torch.Tensor):
                eda = torch.tensor(eda, dtype=torch.float32)
            
            # (Optional) transform the frame
            if self.transform:
                frame = self.transform(frame)
            
            window_frames.append(frame)
            window_bvp.append(bvp)
            window_eda.append(eda)
        
        # Stack them along time dimension
        # frames shape: (window_size, 3, H, W)
        window_frames = torch.stack(window_frames, dim=0)
        
        # bvp, eda shape: (window_size,)
        # (If bvp/eda are scalars, we'll just make them shape (W,) 
        #  or if they are shape (1,), .squeeze() them.)
        window_bvp = torch.stack(window_bvp, dim=0).squeeze()
        window_eda = torch.stack(window_eda, dim=0).squeeze()
        
        return window_frames, window_bvp, window_eda
