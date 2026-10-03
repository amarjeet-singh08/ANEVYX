import pandas as pd
import numpy as np
from typing import Tuple, List

def create_patient_level_splits(
    metadata_df: pd.DataFrame, 
    patient_col: str, 
    train_ratio: float = 0.6, 
    val_ratio: float = 0.15,
    cal_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset ensuring no patient appears in multiple splits.
    Outputs: Train, Validation, Calibration, Test
    """
    # 1. Pre-split duplicate check
    _verify_no_duplicates(metadata_df)
    
    unique_patients = metadata_df[patient_col].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_patients)
    
    n_total = len(unique_patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_cal = int(n_total * cal_ratio)
    
    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train + n_val])
    cal_patients = set(unique_patients[n_train + n_val:n_train + n_val + n_cal])
    test_patients = set(unique_patients[n_train + n_val + n_cal:])
    
    train_df = metadata_df[metadata_df[patient_col].isin(train_patients)]
    val_df = metadata_df[metadata_df[patient_col].isin(val_patients)]
    cal_df = metadata_df[metadata_df[patient_col].isin(cal_patients)]
    test_df = metadata_df[metadata_df[patient_col].isin(test_patients)]
    
    # 2. Leakage Check
    _verify_no_leakage(train_patients, val_patients, cal_patients, test_patients)
    
    return train_df, val_df, cal_df, test_df

def _verify_no_duplicates(df: pd.DataFrame):
    """Detect duplicate rows and fail if present."""
    if df.duplicated().any():
        raise ValueError("CRITICAL: Duplicate rows detected in dataset metadata! Cannot proceed safely.")

def _verify_no_leakage(train_p: set, val_p: set, cal_p: set, test_p: set):
    """Fails loudly if patient leakage is detected across any sets."""
    sets = [("train", train_p), ("val", val_p), ("cal", cal_p), ("test", test_p)]
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            n1, s1 = sets[i]
            n2, s2 = sets[j]
            if s1.intersection(s2):
                raise ValueError(f"CRITICAL: Patient leakage between {n1} and {n2} sets!")
