import os
import sys
import argparse
import numpy as np
import torch
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.moe import HomeCreditMoE
from src.data.dataset import collate_fn

def run_inference(model_path, npz_path, sample_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model & Meta
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    meta = checkpoint["meta"]
    modality_dims = meta["modality_dims"]
    modality_names = meta["modality_names"]
    scalers = checkpoint["scalers"]
    
    model = HomeCreditMoE(modality_dims=modality_dims, rep_dim=64).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 2. Load Data gốc
    data = np.load(npz_path, allow_pickle=True)
    y_full = data['y']
    mask_full = data['level2_modality_mask']
    
    # Lấy sample cụ thể (dựa trên index gốc trong file npz)
    # Lưu ý: Sample index này phải thuộc tập test hoặc tập bất kỳ, nhưng ta sẽ tái sử dụng logic scale
    # Để đơn giản demo, ta giả sử sample_index là index trong file gốc.
    # Ta cần xác định xem sample này thuộc phần nào để dùng scaler đúng? 
    # -> Cách an toàn nhất: Lưu toàn bộ scaler đã fit trên toàn bộ tập train (trong thực tế deploy sẽ làm vậy).
    # Ở đây, vì script train lưu scaler của fold tốt nhất, ta dùng scaler đó để transform sample này.
    # Nếu sample này nằm ngoài tập train ban đầu, nó vẫn được transform đúng cách bởi scaler đó.
    
    X_raw_sample = []
    for i, name in enumerate(modality_names):
        x_col = data[f"X_{name}"][sample_index:sample_index+1]
        x_col = np.nan_to_num(x_col, nan=0.0, posinf=0.0, neginf=0.0)
        X_raw_sample.append(x_col)
    
    # Transform bằng scalers đã lưu
    X_scaled_sample = []
    for i, scaler in enumerate(scalers):
        X_scaled_sample.append(scaler.transform(X_raw_sample[i]))
    
    mask_sample = mask_full[sample_index:sample_index+1]
    true_label = y_full[sample_index]
    
    # 3. Inference
    x_tensor_list = [torch.tensor(x, dtype=torch.float32).to(device) for x in X_scaled_sample]
    mask_tensor = torch.tensor(mask_sample, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits, gates, _ = model(x_tensor_list, mask_tensor)
        prob = torch.sigmoid(logits).item()
        gate_weights = gates.cpu().numpy()[0]
    
    # 4. Hiển thị kết quả
    print("\n" + "="*40)
    print(f"👤 CUSTOMER INDEX: {sample_index}")
    print(f"🎯 TRUE LABEL: {true_label}")
    print(f"🔮 PREDICTED PROB: {prob:.4f} -> {'DEFAULT ⚠️' if prob > 0.5 else 'SAFE ✅'}")
    print("-" * 40)
    print("⚖️ MODALITY IMPORTANCE (HEATMAP DATA):")
    
    for name, weight in zip(modality_names, gate_weights):
        bar_len = int(weight * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"   {name:<15}: {weight:.3f} |{bar}|")
    
    print("-" * 40)
    print("📂 DATA AVAILABILITY:")
    avail = mask_sample[0]
    for name, a in zip(modality_names, avail):
        status = "✅ Present" if a > 0 else "❌ Missing"
        print(f"   {name:<15}: {status}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model_final.pt")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to original features.npz")
    parser.add_argument("--sample_index", type=int, default=0, help="Index of customer to predict")
    args = parser.parse_args()
    
    run_inference(args.model_path, args.npz_path, args.sample_index)