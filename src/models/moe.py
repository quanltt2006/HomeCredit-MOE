import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTableExpert(nn.Module):
    """Expert MLP cơ bản cho các bảng nhỏ (bureau, previous, ...)."""
    def __init__(self, input_dim, output_dim=64, hidden_dim=256, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.net(x)

class TwoLevelGroupedExpert(nn.Module):
    """
    Expert 2 tầng cho modality 'application'.
    Tầng 1: Nhóm feature ngữ nghĩa -> Expert riêng.
    Tầng 2: Sub-gate kết hợp các nhóm thành 1 vector representation duy nhất.
    """
    def __init__(self, group_indices, rep_dim=64, group_hidden_dim=128, dropout=0.3):
        super().__init__()
        self.group_names = list(group_indices.keys())
        self.num_groups = len(self.group_names)

        # Register indices làm buffer để tự động di chuyển theo device
        for i, idxs in enumerate(group_indices.values()):
            self.register_buffer(f"_group_idx_{i}", torch.tensor(idxs, dtype=torch.long))

        # Tầng 1: Expert cho từng nhóm
        self.group_experts = nn.ModuleList([
            SimpleTableExpert(
                input_dim=len(idxs),
                output_dim=rep_dim,
                hidden_dim=min(group_hidden_dim, max(32, len(idxs) * 4)),
                dropout=dropout,
            )
            for idxs in group_indices.values()
        ])

        # Tầng 2: Sub-gate kết hợp các nhóm
        self.sub_gate = nn.Sequential(
            nn.Linear(rep_dim * self.num_groups, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.num_groups)
        )

    def forward(self, x):
        group_outputs = []
        for i, expert in enumerate(self.group_experts):
            idxs = getattr(self, f"_group_idx_{i}")
            # Lấy feature theo index nhóm
            group_outputs.append(expert(x[:, idxs]))

        group_stack = torch.stack(group_outputs, dim=1)              # [B, num_groups, rep_dim]
        sub_gate_logits = self.sub_gate(group_stack.reshape(x.shape[0], -1))
        sub_gate_weights = torch.softmax(sub_gate_logits, dim=1)      # [B, num_groups]

        # Fusion nội bộ
        z = torch.sum(group_stack * sub_gate_weights.unsqueeze(-1), dim=1)  # [B, rep_dim]
        return z, sub_gate_weights

class HomeCreditMoE(nn.Module):
    """
    Mô hình Hierarchical Multimodal MoE hoàn chỉnh.
    - Application: dùng TwoLevelGroupedExpert.
    - Các bảng khác: dùng SimpleTableExpert.
    """
    def __init__(self, modality_dims, application_group_indices, rep_dim=64, num_classes=1):
        super().__init__()
        self.num_modalities = len(modality_dims)

        # Expert đặc biệt cho application (modality đầu tiên)
        self.application_expert = TwoLevelGroupedExpert(application_group_indices, rep_dim=rep_dim)
        
        # Experts cho các modality còn lại (bureau, previous, ...)
        self.other_experts = nn.ModuleList([
            SimpleTableExpert(dim, output_dim=rep_dim) for dim in modality_dims[1:]
        ])

        # Gate cấp cao (kết hợp các modality)
        # Input vẫn lấy từ application vì đây là core modality luôn có dữ liệu
        self.gate_net = nn.Sequential(
            nn.Linear(modality_dims[0], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.num_modalities)
        )

        self.head = nn.Sequential(
            nn.Linear(rep_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

        self.unimodal_heads = nn.ModuleList([
            nn.Linear(rep_dim, num_classes) for _ in range(self.num_modalities)
        ])

    def forward(self, x_list, mask):
        expert_outputs = []
        app_sub_gate = None

        # 1. Xử lý Application (MoE 2 tầng)
        z_app, app_sub_gate = self.application_expert(x_list[0])
        expert_outputs.append(z_app * mask[:, 0:1])

        # 2. Xử lý các modality còn lại
        for i, expert in enumerate(self.other_experts, start=1):
            z = expert(x_list[i])
            expert_outputs.append(z * mask[:, i:i+1])

        # 3. Tính gate weights cấp cao
        gate_logits = self.gate_net(x_list[0])
        mask_inf = (1.0 - mask) * -1e9
        gate_weights = torch.softmax(gate_logits + mask_inf, dim=1)

        # 4. Fusion
        expert_stack = torch.stack(expert_outputs, dim=1)
        z_fused = torch.sum(expert_stack * gate_weights.unsqueeze(-1), dim=1)

        logits = self.head(z_fused).squeeze(-1)
        
        # Unimodal logits cho auxiliary loss
        unimodal_logits = [
            head(z_i).squeeze(-1) for head, z_i in zip(self.unimodal_heads, expert_outputs)
        ]

        # Trả về thêm app_sub_gate để phục vụ interpretability chi tiết
        return logits, gate_weights, unimodal_logits, app_sub_gate