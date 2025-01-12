import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义 Transformer Encoder 分类模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=1536, num_classes=2, num_heads=8, num_layers=2, hidden_dim=32, seq_len=4):
        super(TransformerClassifier, self).__init__()
        # Positional encoding
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, input_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation='relu',
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Classifier Head
        self.classifier = nn.Sequential(
            # Linear -> ReLU -> Linear
            nn.Linear(seq_len*input_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2,hidden_dim),# num_classes),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Add positional encoding
        x = x + self.position_embedding

        # Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Classifier Head
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)  
        return logits
    
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1536, num_classes=2, hidden_dim=512):
        super(MLPClassifier, self).__init__()
        # MLP (Linear -> BN -> ReLU)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
            
    def forward(self, x):
        x = x.unsqueeze(1)  # 添加伪序列维度
        logits = self.classifier(x).squeeze(1)
        # print(logits)
        return logits