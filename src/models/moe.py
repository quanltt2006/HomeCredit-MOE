import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTableExpert(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        return self.net(x)

class HomeCreditMoE(nn.Module):
    def __init__(self, modality_dims, rep_dim=64, num_classes=1):
        super().__init__()
        self.num_modalities = len(modality_dims)
        self.experts = nn.ModuleList([
            SimpleTableExpert(dim, output_dim=rep_dim) for dim in modality_dims
        ])

        # Gate dùng feature của modality đầu tiên (Application - Core)
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
        for i, expert in enumerate(self.experts):
            z = expert(x_list[i])
            expert_outputs.append(z * mask[:, i:i+1])

        gate_logits = self.gate_net(x_list[0])
        mask_inf = (1.0 - mask) * -1e9
        gate_weights = torch.softmax(gate_logits + mask_inf, dim=1)

        expert_stack = torch.stack(expert_outputs, dim=1)
        z_fused = torch.sum(expert_stack * gate_weights.unsqueeze(-1), dim=1)

        logits = self.head(z_fused).squeeze(-1)
        unimodal_logits = [
            head(z_i).squeeze(-1) for head, z_i in zip(self.unimodal_heads, expert_outputs)
        ]

        return logits, gate_weights, unimodal_logits