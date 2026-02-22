import os
import glob
import torch
from torch.utils.data import Dataset
import cv2  # Ensure OpenCV is installed: pip install opencv-python

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

def scale_landmarks(landmarks, image_size=(128, 128)):
    """
    Scale landmarks to fit within the image size.

    Args:
        landmarks (torch.Tensor): Tensor of shape (136,) representing 68 (x, y) coordinates.
        image_size (tuple): Desired image size as (H, W).

    Returns:
        torch.Tensor: Scaled landmarks of shape (136,)
    """
    H, W = image_size
    landmarks = landmarks.clone()
    
    # Find the max coordinates to determine scaling factors
    max_x = landmarks[::2].max()
    max_y = landmarks[1::2].max()
    
    scale_x = W / max_x
    scale_y = H / max_y
    
    # Scale x and y coordinates
    landmarks[::2] = landmarks[::2] * scale_x
    landmarks[1::2] = landmarks[1::2] * scale_y
    
    return landmarks


def generate_landmark_heatmap(landmarks, image_size, sigma=5):
    """
    Generate a heatmap with Gaussian blobs at landmark positions.

    Args:
        landmarks (torch.Tensor or numpy.ndarray): Tensor of shape (136,)
            representing 68 (x, y) coordinates.
        image_size (tuple): (H, W) dimensions of the heatmap.
        sigma (float): Standard deviation for Gaussian blobs.

    Returns:
        torch.Tensor: Heatmap tensor of shape (1, H, W) with values in [0, 1].
    """
    H, W = image_size
    heatmap = np.zeros((H, W), dtype=np.float32)

    # Ensure landmarks are in numpy format
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()
    elif isinstance(landmarks, list):
        landmarks = np.array(landmarks)

    # Iterate over each (x, y) pair
    for i in range(0, len(landmarks), 2):
        x, y = landmarks[i], landmarks[i + 1]

        # Handle normalized coordinates (if applicable)
        # If landmarks are normalized between 0 and 1, scale them
        if x <= 1 and y <= 1:
            x = int(x * (W - 1))
            y = int(y * (H - 1))
        else:
            x = int(x)
            y = int(y)

        # Check for valid coordinates
        if x < 0 or x >= W or y < 0 or y >= H:
            continue

        # Draw a Gaussian blob
        # Create a small Gaussian kernel
        size = int(6 * sigma + 1)
        gaussian = cv2.getGaussianKernel(size, sigma)
        gaussian = gaussian * gaussian.T  # 2D Gaussian

        # Define the region of interest on the heatmap
        x1 = x - size // 2
        y1 = y - size // 2
        x2 = x1 + size
        y2 = y1 + size

        # Compute the intersection of the Gaussian with the heatmap boundaries
        gx1 = 0
        gy1 = 0
        gx2 = size
        gy2 = size

        if x1 < 0:
            gx1 = -x1
            x1 = 0
        if y1 < 0:
            gy1 = -y1
            y1 = 0
        if x2 > W:
            gx2 = size - (x2 - W)
            x2 = W
        if y2 > H:
            gy2 = size - (y2 - H)
            y2 = H

        # Add the Gaussian to the heatmap
        heatmap[y1:y2, x1:x2] += gaussian[gy1:gy2, gx1:gx2]

    # Clip values to [0,1]
    heatmap = np.clip(heatmap, 0, 1)

    # Convert to torch.Tensor and add channel dimension
    heatmap = torch.from_numpy(heatmap).unsqueeze(0)  # Shape: (1, H, W)

    return heatmap


def concatenate_heatmap_with_image(image_tensor, landmarks, image_size=(128, 128), sigma=5):
    """
    Concatenate a landmark heatmap to the RGB image tensor.

    Args:
        image_tensor (torch.Tensor): Tensor of shape (3, H, W)
        landmarks (torch.Tensor or numpy.ndarray): Tensor of shape (136,)
        image_size (tuple): (H, W) dimensions
        sigma (float): Standard deviation for Gaussian blobs

    Returns:
        torch.Tensor: Combined tensor of shape (4, H, W)
    """
    heatmap = generate_landmark_heatmap(landmarks, image_size, sigma=sigma)  # Shape: (1, H, W)
    combined = torch.cat((image_tensor, heatmap), dim=0)  # Shape: (4, H, W)
    return combined


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
        
        # 2) Sort by frame number to ensure temporal order
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
          frames: (window_size, 4, H, W)  # 3 RGB channels + 1 Heatmap
          landmarks: (window_size, 136)
          eda:    (window_size,)
        """
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_size
        
        # Collect the frames/landmarks/EDA
        window_frames = []
        window_landmarks = []
        window_eda = []
        
        # For each time step in this window
        for fpath in self.filepaths[start_idx:end_idx]:
            try:
                data = torch.load(fpath)  # {'tensor':..., 'eda':..., 'landmarks':...}
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                # Optionally, handle missing or corrupted files
                # Here, we'll skip this window by returning the next one
                raise e  # Or continue, depending on your preference
            
            # Extract and process the image tensor
            image = data.get('tensor')  # Expected shape: (3, H, W)
            if image is None:
                raise ValueError(f"'tensor' key not found in {fpath}")
            
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image, dtype=torch.float32)  # Ensure tensor type
            
            # Extract landmarks
            landmarks = data.get('landmarks')  # Expected shape: (68, 2)
            if landmarks is None:
                raise ValueError(f"'landmarks' key not found in {fpath}")
            
            if isinstance(landmarks, list):
                landmarks = torch.tensor(landmarks, dtype=torch.float32)
            elif isinstance(landmarks, np.ndarray):
                landmarks = torch.from_numpy(landmarks).float()
            elif not isinstance(landmarks, torch.Tensor):
                landmarks = torch.tensor(landmarks, dtype=torch.float32)
            
            if landmarks.dim() == 2 and landmarks.shape[1] == 2:
                landmarks = landmarks.view(-1)  # Flatten to (136,)
            else:
                raise ValueError(f"Unexpected landmarks shape in {fpath}: {landmarks.shape}")
            
            # Extract EDA
            eda = data.get('eda')  # Expected to be a scalar or tensor
            if eda is None:
                raise ValueError(f"'eda' key not found in {fpath}")
            
            if not isinstance(eda, torch.Tensor):
                eda = torch.tensor(eda, dtype=torch.float32)
            
            if eda.dim() > 0:
                eda = eda.squeeze()

            scaled_landmarks = scale_landmarks(landmarks, image_size=(image.shape[1], image.shape[2]))
            print(f"Scaled Landmarks: {scaled_landmarks}")

            
            # Generate Heatmap and Concatenate
            heatmap = generate_landmark_heatmap(scaled_landmarks, image_size=(image.shape[1], image.shape[2]), sigma=3)  # Shape: (1, H, W)
            combined_image = torch.cat((image, heatmap), dim=0)  # Shape: (4, H, W)

            # Debug: Print image size and landmarks
            print(f"Image Size: {image.shape[1]}x{image.shape[2]}")
            print(f"Landmarks: {landmarks}")

            print("~~~~~~~~~hello")
            if idx == 0:
                rgb = combined_image[:3].permute(1, 2, 0).numpy()
                heatmap_np = combined_image[3].numpy()
                
                # Normalize heatmap for better visibility
                heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
                
                plt.figure(figsize=(6, 6))
                plt.imshow(rgb.astype(np.uint8))
                plt.imshow(heatmap_np, cmap='jet', alpha=0.5)
                plt.title("RGB Image with Landmark Heatmap")
                plt.axis('off')
                plt.show()
            
            # (Optional) Apply transformations
            if self.transform:
                combined_image = self.transform(combined_image)
            
            window_frames.append(combined_image)      # Shape: (4, H, W)
            window_landmarks.append(landmarks)        # Shape: (136,)
            window_eda.append(eda)                    # Shape: ()
        
        # Stack them along time dimension
        # frames shape: (window_size, 4, H, W)
        window_frames = torch.stack(window_frames, dim=0)
        
        # landmarks shape: (window_size, 136)
        window_landmarks = torch.stack(window_landmarks, dim=0)
        
        # eda shape: (window_size,)
        window_eda = torch.stack(window_eda, dim=0)
        
        return window_frames, window_landmarks, window_eda
