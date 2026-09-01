import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict

class HomeCreditDataset(Dataset):
    def __init__(self, x_list: List[np.ndarray], y: np.ndarray, mask: np.ndarray):
        # Chuyển đổi sang tensor ngay tại đây để tăng tốc khi train
        self.x_list = [torch.tensor(x, dtype=torch.float32) for x in x_list]
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [x[idx] for x in self.x_list], self.y[idx], self.mask[idx]

def collate_fn(batch):
    num_modalities = len(batch[0][0])
    x_list_batch = []
    for m in range(num_modalities):
        x_list_batch.append(torch.stack([item[0][m] for item in batch]))
    
    y_batch = torch.stack([item[1] for item in batch])
    mask_batch = torch.stack([item[2] for item in batch])
    return x_list_batch, y_batch, mask_batch

def prepare_data_splits(
    raw_data: Dict,  # Đã sửa từ 'raw_ Dict' thành 'raw_data: Dict'
    modality_names: List[str], 
    test_size: float = 0.2, 
    random_state: int = 42
):
    """
    Chuẩn bị dữ liệu: Chia Train/Val và Test.
    """
    y_full = raw_data['y']
    mask_full = raw_data['level2_modality_mask']
    
    # Lấy raw features chưa scale
    X_raw_list = []
    for name in modality_names:
        key = f"X_{name}"
        if key in raw_data:
            x = raw_data[key]
            # Xử lý NaN trước khi tách
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            X_raw_list.append(x)
        else:
            print(f"Cảnh báo: Không tìm thấy {key}")

    n_samples = len(y_full)
    indices = np.arange(n_samples)

    # 1. Chia tập Hold-out Test (Stratified)
    train_val_idx, test_idx = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_full
    )

    # Tách dữ liệu Train/Val và Test (Dữ liệu THÔ)
    meta = {
        "modality_names": modality_names,
        "modality_dims": [x.shape[1] for x in X_raw_list],
        "n_train_val": len(train_val_idx), # THÊM DÒNG NÀY
        "n_test": len(test_idx),           # THÊM DÒNG NÀY
        "test_indices": test_idx
    }

    return {
        "train_val": {
            "X": [x[train_val_idx] for x in X_raw_list], 
            "y": y_full[train_val_idx], 
            "mask": mask_full[train_val_idx]
        },
        "test": {
            "X": [x[test_idx] for x in X_raw_list], 
            "y": y_full[test_idx], 
            "mask": mask_full[test_idx]
        },
        "meta": meta
    }
