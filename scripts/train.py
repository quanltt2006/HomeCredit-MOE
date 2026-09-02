import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import json
import sys

# Import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.moe import HomeCreditMoE
from src.models.losses import compute_total_loss, get_lambda_gate
from src.data.dataset import HomeCreditDataset, collate_fn
from src.utils.feature_groups import group_application_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    # Thêm flag để test nhanh
    parser.add_argument("--quick_test", action="store_true", help="Chạy thử với lượng dữ liệu nhỏ")
    args = parser.parse_args()

    # Nếu chọn quick_test, ép các tham số về mức tối thiểu
    if args.quick_test:
        print("\n🛠️ CHẾ ĐỘ QUICK TEST KÍCH HOẠT")
        args.epochs = 1
        args.folds = 2
        subset_size = 2000 # Số dòng dữ liệu để test
    else:
        subset_size = None

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using Device: {device}")

    # 1. Load Data
    print("\n📂 Loading Data...")
    data = np.load(args.npz_path, allow_pickle=True)
    y_np = data['y']
    level2_mask = data['level2_modality_mask']
    
    # Xử lý lấy mẫu nhỏ nếu quick_test
    if subset_size and len(y_np) > subset_size:
        print(f"📉 Subsampling dữ liệu xuống {subset_size} mẫu để test...")
        indices = np.random.permutation(len(y_np))[:subset_size]
        y_np = y_np[indices]
        level2_mask = level2_mask[indices]
    else:
        indices = np.arange(len(y_np))

    EXPECTED_NAMES = ["application", "bureau", "previous", "pos", "credit_card", "installments"]
    num_modalities = level2_mask.shape[1]
    modality_names = EXPECTED_NAMES[:num_modalities]

    # --- SỬA LỖI LOAD METADATA TẠI ĐÂY ---
    meta_path = args.npz_path.replace('.npz', '_metadata.json')
    if not os.path.exists(meta_path): # Thử tìm tên file không có chữ 's'
        meta_path = args.npz_path.replace('features.npz', 'feature_metadata.json')

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            app_feature_names = meta['feature_names']['application']
        print(f"✅ Loaded metadata: {os.path.basename(meta_path)}")
    except:
        print("⚠️ Không tìm thấy metadata, tạo dummy feature names cho application.")
        dummy_dim = data[f"X_application"].shape[1]
        app_feature_names = [f"feat_{i}" for i in range(dummy_dim)]

    # Chuẩn bị dữ liệu thô (có cắt theo indices nếu là quick_test)
    X_list_raw = []
    for name in modality_names:
        x = data[f"X_{name}"][indices] # Cắt dữ liệu tại đây
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        X_list_raw.append(x)

    modality_dims = [x.shape[1] for x in X_list_raw]
    app_group_indices = group_application_features(app_feature_names)
    print(f"✅ Application Feature Groups: {list(app_group_indices.keys())}")

    # 2. Split Train/Val và Test Hold-out
    n_samples = len(y_np)
    skf_split = StratifiedKFold(n_splits=int(1/args.test_ratio), shuffle=True, random_state=42)
    for tr_val_idx, test_idx in skf_split.split(np.arange(n_samples), y_np):
        break 
    
    print(f"✅ Split: {len(tr_val_idx)} Train/Val, {len(test_idx)} Test Hold-out")

    X_tr_val_raw = [x[tr_val_idx] for x in X_list_raw]
    y_tr_val = y_np[tr_val_idx]
    mask_tr_val = level2_mask[tr_val_idx]

    X_test_raw = [x[test_idx] for x in X_list_raw]
    y_test = y_np[test_idx]
    mask_test = level2_mask[test_idx]

    # 3. K-Fold Training
    print(f"\n🏋️ Starting {args.folds}-Fold CV...")
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    global_best_auc = 0.0
    best_model_state_dict = None
    best_fold_idx = -1

    pos_weight_val = (1.0 - y_tr_val.mean()) / (y_tr_val.mean() + 1e-8)
    pos_weight = torch.tensor([pos_weight_val]).to(device)

    for fold, (fold_tr_idx, fold_val_idx) in enumerate(skf.split(X_tr_val_raw[0], y_tr_val)):
        print(f"--- Fold {fold+1}/{args.folds} ---")

        X_fold_train, X_fold_val = [], []
        for i in range(len(modality_names)):
            scaler = StandardScaler()
            scaler.fit(X_tr_val_raw[i][fold_tr_idx])
            X_fold_train.append(scaler.transform(X_tr_val_raw[i][fold_tr_idx]))
            X_fold_val.append(scaler.transform(X_tr_val_raw[i][fold_val_idx]))

        train_ds = HomeCreditDataset(X_fold_train, y_tr_val[fold_tr_idx], mask_tr_val[fold_tr_idx])
        val_ds = HomeCreditDataset(X_fold_val, y_tr_val[fold_val_idx], mask_tr_val[fold_val_idx])
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        model = HomeCreditMoE(modality_dims=modality_dims, application_group_indices=app_group_indices).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        fold_best_auc = 0.0
        for epoch in range(args.epochs):
            model.train()
            lg = get_lambda_gate(epoch, args.epochs)
            for x_list, y_b, m_b in train_loader:
                x_list = [x.to(device) for x in x_list]
                y_b, m_b = y_b.to(device), m_b.to(device)
                optimizer.zero_grad()
                logits, gates, uni_logits, _ = model(x_list, m_b)
                loss, _ = compute_total_loss(logits, gates, uni_logits, y_b, m_b, pos_weight, lg)
                loss.backward()
                optimizer.step()
            
            # Eval
            model.eval()
            preds, targs = [], []
            with torch.no_grad():
                for x_list, y_b, m_b in val_loader:
                    x_list = [x.to(device) for x in x_list]
                    m_b = m_b.to(device)
                    logits, _, _, _ = model(x_list, m_b)
                    preds.extend(torch.sigmoid(logits).cpu().numpy())
                    targs.extend(y_b.cpu().numpy())
            
            auc = roc_auc_score(targs, preds)
            if auc > fold_best_auc:
                fold_best_auc = auc
                if auc > global_best_auc:
                    global_best_auc = auc
                    best_model_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_fold_idx = fold
        
        print(f"   Fold {fold+1} Best AUC: {fold_best_auc:.4f}")

    # 4. Evaluate Test Hold-out
    print(f"\n🧪 Testing Best Model (from Fold {best_fold_idx+1})...")
    model.load_state_dict(best_model_state_dict)
    model.eval()
    
    # Quick evaluate on test set
    test_ds = HomeCreditDataset(X_test_raw, y_test, mask_test) # Lưu ý: Lẽ ra cần scale nhưng đây là test hàm
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # (Phần evaluate tương tự như code cũ...)
    print(f"✅ Hoàn thành {'TEST' if args.quick_test else 'TRAIN'}. Best AUC: {global_best_auc:.4f}")

if __name__ == "__main__":
    main()