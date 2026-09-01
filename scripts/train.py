import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import HomeCreditDataset, collate_fn, prepare_data_splits
from src.models.moe import HomeCreditMoE
from src.models.losses import compute_total_loss, get_lambda_gate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using Device: {device}")

    # 1. Load & Split Data (Chưa Scale)
    print("📂 Loading and splitting data...")
    raw_data = np.load(args.npz_path, allow_pickle=True)
    EXPECTED_NAMES = ["application", "bureau", "previous", "pos", "credit_card", "installments"]
    
    splits = prepare_data_splits(
        raw_data, 
        modality_names=EXPECTED_NAMES, 
        test_size=args.test_size
    )
    
    meta = splits["meta"]
    modality_dims = meta["modality_dims"]
    test_indices = meta["test_indices"]
    
    print(f"✅ Split Done: {meta['n_train_val']} Train/Val, {meta['n_test']} Test (Hold-out)")
    print(f"📊 Modality Dims: {modality_dims}")

    # Dữ liệu thô để fit scaler riêng trong từng fold
    X_raw_train = splits["train_val"]["X"]
    y_train = splits["train_val"]["y"]
    mask_train = splits["train_val"]["mask"]

    X_raw_test = splits["test"]["X"]
    y_test = splits["test"]["y"]
    mask_test = splits["test"]["mask"]

    # 2. K-Fold Training Loop
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_scores = []
    oof_preds = np.zeros(len(y_train))
    
    # Tính pos_weight từ tập train
    pos_ratio = y_train.mean()
    pos_weight_val = (1.0 - pos_ratio) / (pos_ratio + 1e-8)
    pos_weight = torch.tensor([pos_weight_val]).to(device)

    print(f"\n🏋️ Starting {args.folds}-Fold Training on Train/Val set...")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_raw_train[0], y_train)):
        print(f"\n--- FOLD {fold + 1}/{args.folds} ---")

        # ⚠️ FIX LEAKAGE: Fit Scaler CHỈ trên tập Train của Fold này
        scalers = []
        X_tr_scaled = []
        X_val_scaled = []

        for m in range(len(modality_dims)):
            scaler = StandardScaler()
            # Fit chỉ trên train
            scaler.fit(X_raw_train[m][tr_idx])
            scalers.append(scaler)
            
            # Transform cả train và val
            X_tr_scaled.append(scaler.transform(X_raw_train[m][tr_idx]))
            X_val_scaled.append(scaler.transform(X_raw_train[m][val_idx]))

        # Tạo Datasets
        tr_ds = HomeCreditDataset(X_tr_scaled, y_train[tr_idx], mask_train[tr_idx])
        val_ds = HomeCreditDataset(X_val_scaled, y_train[val_idx], mask_train[val_idx])

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, collate_fn=collate_fn)

        # Init Model
        model = HomeCreditMoE(modality_dims=modality_dims, rep_dim=64).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        best_val_auc = 0.0
        best_model_path = os.path.join(args.output_dir, "models", f"model_fold_{fold+1}.pt")

        for epoch in range(args.epochs):
            lambda_g = get_lambda_gate(epoch, args.epochs)
            model.train()
            
            loss_sum = 0.0
            for x_list, y_b, m_b in tr_loader:
                x_list = [x.to(device) for x in x_list]
                y_b, m_b = y_b.to(device), m_b.to(device)

                optimizer.zero_grad()
                logits, gates, uni_logits = model(x_list, m_b)
                loss, logs = compute_total_loss(logits, gates, uni_logits, y_b, m_b, pos_weight, lambda_g)
                
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            # Validation
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for x_list, y_b, m_b in val_loader:
                    x_list = [x.to(device) for x in x_list]
                    m_b = m_b.to(device)
                    logits, _, _ = model(x_list, m_b)
                    val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    val_targets.extend(y_b.cpu().numpy())

            auc = roc_auc_score(val_targets, val_preds)
            scheduler.step(auc)

            # ⚠️ FIX SAVE MODEL: Lưu ngay ra disk khi tốt nhất
            if auc > best_val_auc:
                best_val_auc = auc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "scalers": scalers, # Lưu kèm scalers của fold này
                    "epoch": epoch,
                    "val_auc": auc
                }, best_model_path)
                # Cập nhật OOF preds
                oof_preds[val_idx] = val_preds

            print(f"Epoch {epoch+1}: Loss={loss_sum/len(tr_loader):.4f}, Val AUC={auc:.4f} (Best={best_val_auc:.4f})")

        fold_scores.append(best_val_auc)
        print(f">>> Fold {fold+1} Best AUC: {best_val_auc:.4f} | Saved to {best_model_path}")

    # 3. Báo cáo OOF
    print("\n" + "="*40)
    print(f"📈 OOF Mean AUC: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
    print(f"📈 OOF Global AUC: {roc_auc_score(y_train, oof_preds):.4f}")
    print("="*40)

    # 4. Evaluation trên tập HOLD-OUT TEST (20%)
    print("\n🧪 Evaluating on Hold-out Test Set (20%)...")
    
    # Chọn mô hình tốt nhất trong các fold để test (hoặc ensemble, ở đây chọn fold best nhất)
    best_fold_idx = np.argmax(fold_scores)
    best_fold_path = os.path.join(args.output_dir, "models", f"model_fold_{best_fold_idx+1}.pt")
    
    checkpoint = torch.load(best_fold_path, map_location=device)
    final_model = HomeCreditMoE(modality_dims=modality_dims, rep_dim=64).to(device)
    final_model.load_state_dict(checkpoint["model_state_dict"])
    final_scalers = checkpoint["scalers"]
    
    # Scale tập Test bằng scalers của fold tốt nhất (đúng quy trình)
    X_test_scaled = []
    for m, scaler in enumerate(final_scalers):
        X_test_scaled.append(scaler.transform(X_raw_test[m]))
    
    test_ds = HomeCreditDataset(X_test_scaled, y_test, mask_test)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, collate_fn=collate_fn)
    
    final_model.eval()
    test_preds, test_targets, test_gates = [], [], []
    
    with torch.no_grad():
        for x_list, y_b, m_b in test_loader:
            x_list = [x.to(device) for x in x_list]
            m_b = m_b.to(device)
            logits, gates, _ = final_model(x_list, m_b)
            test_preds.extend(torch.sigmoid(logits).cpu().numpy())
            test_targets.extend(y_b.cpu().numpy())
            test_gates.extend(gates.cpu().numpy())

    test_auc = roc_auc_score(test_targets, test_preds)
    test_ap = average_precision_score(test_targets, test_preds)
    
    print(f"🏆 HELD-OUT TEST AUC: {test_auc:.4f}")
    print(f"🏆 HELD-OUT TEST AP : {test_ap:.4f}")

    # Lưu kết quả cuối cùng
    final_model_path = os.path.join(args.output_dir, "models", "best_model_final.pt")
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "scalers": final_scalers,
        "meta": meta,
        "test_auc": test_auc,
        "test_ap": test_ap
    }, final_model_path)
    
    result_log = {
        "oof_auc": float(np.mean(fold_scores)),
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
        "best_fold": best_fold_idx + 1
    }
    
    with open(os.path.join(args.output_dir, "logs", "results.json"), "w") as f:
        json.dump(result_log, f, indent=2)
        
    print(f"💾 Final Model & Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()