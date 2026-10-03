import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from backend.app.imaging.preprocessing import preprocess_pipeline

class MedicalDataset(Dataset):
    """
    Standard PyTorch Dataset for medical images.
    """
    def __init__(self, metadata: pd.DataFrame, image_dir: str, path_col: str, label_col: str, target_size: int = 256):
        self.metadata = metadata.reset_index(drop=True)
        self.image_dir = image_dir
        self.path_col = path_col
        self.label_col = label_col
        self.target_size = target_size

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.image_dir, str(row[self.path_col]))
        
        # Load and preprocess
        try:
            raw_img = np.array(Image.open(img_path).convert("L"))
            processed_img = preprocess_pipeline(raw_img, target_size=self.target_size)
            # Add channel dimension for PyTorch: (1, H, W)
            tensor_img = torch.from_numpy(processed_img).unsqueeze(0).float()
            
            # Label parsing (1.0 for normal, 0.0 for abnormal depending on logic)
            label = float(row[self.label_col]) if self.label_col in row else 1.0
            
            return tensor_img, label
            
        except Exception as e:
            # Fallback for broken images to avoid crashing the DataLoader loop
            # In production, use a custom collate_fn to filter these out
            dummy = torch.zeros((1, self.target_size, self.target_size), dtype=torch.float32)
            return dummy, -1.0
