import os
import sys
import argparse
import numpy as np
import torch
import json
from sklearn.preprocessing import StandardScaler

# Thêm đường dẫn để import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.moe import HomeCreditMoE
from src.utils.feature_groups import group_application_features

def run_inference(model_dir, npz_path, sample_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Tải Metadata từ training
    model_path = os.path.join(model_dir, "best_home_credit_moe.pt")
    meta_path = os.path.join(model_dir, "model_meta.json")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Không tìm thấy file metadata tại {meta_path}. Hãy chạy train.py trước.")

    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    modality_dims = meta["modality_dims"]
    modality_names = meta["modality_names"]
    app_group_names = meta.get("application_groups", [])

    # 2. Tái tạo Application Group Indices (để init model)
    feat_meta_path = npz_path.replace('.npz', '_metadata.json')
    if not os.path.exists(feat_meta_path):
        feat_meta_path = npz_path.replace('features.npz', 'feature_metadata.json')
        
    try:
        with open(feat_meta_path, 'r') as f:
            f_meta = json.load(f)
            app_feature_names = f_meta['feature_names']['application']
    except:
        app_feature_names = [f"feat_{i}" for i in range(modality_dims[0])]
    
    app_group_indices = group_application_features(app_feature_names)
    
    # Cập nhật lại app_group_names nếu meta cũ không có
    if not app_group_names:
        app_group_names = list(app_group_indices.keys())

    # 3. Khởi tạo Model đúng cấu trúc bạn cung cấp
    print(f"🛠️  Initializing HomeCreditMoE...")
    model = HomeCreditMoE(
        modality_dims=modality_dims, 
        application_group_indices=app_group_indices,
        rep_dim=64
    ).to(device)
    
    # Load trọng số
    print(f"📥 Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 4. Load và Scale dữ liệu mẫu
    print(f"🔍 Processing customer index: {sample_index}...")
    data = np.load(npz_path, allow_pickle=True)
    y_full = data['y']
    mask_full = data['level2_modality_mask']
    
    X_sample_tensors = []
    for i, name in enumerate(modality_names):
        # Lấy data thô
        x_raw = data[f"X_{name}"]
        x_raw = np.nan_to_num(x_raw, nan=0.0)
        
        # Fit scaler tại chỗ (vì model train trên dữ liệu đã scale)
        scaler = StandardScaler()
        scaler.fit(x_raw)
        
        # Transform 1 sample
        x_single = x_raw[sample_index : sample_index+1]
        x_scaled = scaler.transform(x_single)
        X_sample_tensors.append(torch.tensor(x_scaled, dtype=torch.float32).to(device))
    
    mask_tensor = torch.tensor(mask_full[sample_index : sample_index+1], dtype=torch.float32).to(device)
    
    # 5. Inference
    with torch.no_grad():
        # Theo code của bạn: trả về (logits, gate_weights, unimodal_logits, app_sub_gate)
        logits, gate_weights, _, app_sub_gate = model(X_sample_tensors, mask_tensor)
        
        prob = torch.sigmoid(logits).item()
        l2_weights = gate_weights.cpu().numpy()[0]
        l1_weights = app_sub_gate.cpu().numpy()[0]
    
    # 6. In kết quả chi tiết
    print("\n" + "█" * 65)
    print(f"📊 PREDICTION REPORT FOR CUSTOMER #{sample_index}")
    print(f"🎯 ACTUAL STATUS   : {int(y_full[sample_index])} ({'DEFAULT' if y_full[sample_index]==1 else 'SAFE'})")
    print(f"🔮 MODEL PREDICTION : {prob:.4f} -> {'❌ REJECT' if prob > 0.5 else '✅ APPROVE'}")
    print("█" * 65)

    # Hiển thị Tầng 2 (Giữa các bảng dữ liệu)
    print("\n🌐 LEVEL 2: MODALITY IMPORTANCE (Global Gating)")
    for name, weight in zip(modality_names, l2_weights):
        # Kiểm tra xem dữ liệu này có bị thiếu không dựa trên mask
        is_missing = mask_full[sample_index][modality_names.index(name)] == 0
        status = "(Missing)" if is_missing else ""
        
        bar = "█" * int(weight * 30)
        print(f"   {name:<15}: {weight:.3f} |{bar.ljust(30)}| {status}")

    # Hiển thị Tầng 1 (Bên trong bảng Application)
    print("\n📂 LEVEL 1: APPLICATION INTERNAL GROUPS (Feature Gating)")
    for name, weight in zip(app_group_names, l1_weights):
        bar = "▓" * int(weight * 30)
        print(f"   {name:<15}: {weight:.3f} |{bar.ljust(30)}|")

    print("\n" + "=" * 65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Thư mục chứa model .pt và meta .json")
    parser.add_argument("--npz_path", type=str, required=True, help="Đường dẫn file .npz")
    parser.add_argument("--sample_index", type=int, default=100, help="Index khách hàng cần kiểm tra")
    args = parser.parse_args()
    
    run_inference(args.model_dir, args.npz_path, args.sample_index)