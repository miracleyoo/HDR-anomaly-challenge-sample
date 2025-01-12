import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义 Transformer Encoder 分类模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=1536, num_classes=2, num_heads=8, num_layers=2, hidden_dim=512):
        super(TransformerClassifier, self).__init__()
        # 位置编码（可选）
        self.position_embedding = nn.Parameter(torch.randn(1, 1, input_dim))  # 可学的位置编码
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation='relu',
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        # 将输入 reshape 为 (B, L=1, D)
        x = x.unsqueeze(1)  # 添加伪序列维度
        # 添加位置编码
        x = x + self.position_embedding
        # print("Inside TransformerClassifier, x.shape (1):", x.shape)
        # Transformer Encoder
        x = self.transformer_encoder(x)
        # print("Inside TransformerClassifier, x.shape (2):", x.shape)
        # 分类头（取特征的第一个时间步）
        logits = self.classifier(x[:, 0, :])  # 假设 CLIP 特征是 batch x 1 x dim 的形状
        # print("Inside TransformerClassifier, logits.shape:", logits.shape)
        return logits
    
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1536, num_classes=2, hidden_dim=512):
        super(MLPClassifier, self).__init__()
        # 3层 MLP (Linear -> BN -> ReLU)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
            
    def forward(self, x):
        x = x.unsqueeze(1)  # 添加伪序列维度
        logits = self.classifier(x).squeeze(1)
        return logits