import os
import pandas as pd
from abc import ABC, abstractmethod

class DatasetAdapter(ABC):
    """
    Base class for specific medical dataset adapters (CheXpert, MIMIC, etc).
    """
    @abstractmethod
    def load_metadata(self, data_dir: str) -> pd.DataFrame:
        pass
        
    @abstractmethod
    def get_patient_id_column(self) -> str:
        pass
        
    @abstractmethod
    def get_image_path_column(self) -> str:
        pass

    def get_normal_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule 5: Explicitly construct a normal training subset."""
        if "is_normal" not in df.columns:
            raise ValueError("Dataset must have 'is_normal' column to filter normal subset.")
        return df[df["is_normal"] == 1.0].copy()

    def create_patient_split(self, df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15) -> tuple:
        """
        Rule 6: Patient-level splitting to prevent data leakage.
        Ensures the same patient never appears in multiple splits.
        """
        patient_col = self.get_patient_id_column()
        unique_patients = df[patient_col].unique()
        
        # Shuffle safely
        import numpy as np
        np.random.seed(42)
        np.random.shuffle(unique_patients)
        
        n_total = len(unique_patients)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_patients = unique_patients[:n_train]
        val_patients = unique_patients[n_train:n_train + n_val]
        test_patients = unique_patients[n_train + n_val:]
        
        train_df = df[df[patient_col].isin(train_patients)].copy()
        val_df = df[df[patient_col].isin(val_patients)].copy()
        test_df = df[df[patient_col].isin(test_patients)].copy()
        
        # Leakage validation
        train_p = set(train_df[patient_col])
        val_p = set(val_df[patient_col])
        test_p = set(test_df[patient_col])
        
        if train_p.intersection(val_p) or train_p.intersection(test_p) or val_p.intersection(test_p):
            raise RuntimeError("CRITICAL ERROR: Patient leakage detected between splits!")
            
        return train_df, val_df, test_df

class CheXpertAdapter(DatasetAdapter):
    def load_metadata(self, data_dir: str) -> pd.DataFrame:
        csv_path = os.path.join(data_dir, "train.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CheXpert metadata not found at {csv_path}")
        df = pd.read_csv(csv_path)
        # Add normality flag: 1 if "No Finding" == 1 else 0
        df["is_normal"] = (df["No Finding"] == 1.0).astype(float)
        
        # Rule 6: Fix CheXpert adapter to use a real patient identifier
        # Path format: CheXpert-v1.0-small/train/patient00001/study1/...
        def extract_patient(path_str):
            parts = path_str.split('/')
            for p in parts:
                if p.startswith('patient'):
                    return p
            return path_str # fallback
            
        df["patient_id"] = df["Path"].apply(extract_patient)
        return df
        
    def get_patient_id_column(self) -> str:
        return "patient_id"
        
    def get_image_path_column(self) -> str:
        return "Path"

class MIMICCXRAdapter(DatasetAdapter):
    def load_metadata(self, data_dir: str) -> pd.DataFrame:
        csv_path = os.path.join(data_dir, "mimic-cxr-2.0.0-metadata.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"MIMIC-CXR metadata not found at {csv_path}")
        return pd.read_csv(csv_path)
        
    def get_patient_id_column(self) -> str:
        return "subject_id"
        
    def get_image_path_column(self) -> str:
        return "dicom_id"

class NIHAdapter(DatasetAdapter):
    def load_metadata(self, data_dir: str) -> pd.DataFrame:
        csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"NIH metadata not found at {csv_path}")
        df = pd.read_csv(csv_path)
        df["is_normal"] = (df["Finding Labels"] == "No Finding").astype(float)
        return df

    def get_patient_id_column(self) -> str:
        return "Patient ID"

    def get_image_path_column(self) -> str:
        return "Image Index"

class RSNAPneumoniaAdapter(DatasetAdapter):
    def load_metadata(self, data_dir: str) -> pd.DataFrame:
        csv_path = os.path.join(data_dir, "stage_2_train_labels.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"RSNA metadata not found at {csv_path}")
        df = pd.read_csv(csv_path)
        df["is_normal"] = (df["Target"] == 0).astype(float)
        return df

    def get_patient_id_column(self) -> str:
        return "patientId"

    def get_image_path_column(self) -> str:
        return "patientId" # Needs .dcm appended during loading
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class MedicalImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_dir: str, adapter: DatasetAdapter, transform=None):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.adapter = adapter
        self.transform = transform
        self.path_col = self.adapter.get_image_path_column()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_dir, row[self.path_col])
        
        # Load grayscale
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing medical image: {img_path}")
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Corrupt or unreadable medical image: {img_path}")
            
        img = cv2.resize(img, (256, 256))
        
        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0
        
        # Add channel dim
        tensor = torch.from_numpy(img).unsqueeze(0)
        
        if self.transform:
            tensor = self.transform(tensor)
            
        # Return image and is_normal label (for calibration/val logic)
        label = float(row.get("is_normal", -1.0))
        return tensor, label
