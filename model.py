import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import transforms

# 定义 Transformer Encoder 分类模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=1536, num_classes=2, num_heads=8, num_layers=2, hidden_dim=512, seq_len=4):
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

class Model:
    def __init__(self):
        # model will be called from the load() method
        self.clf = None

    def load(self):
        self.device='cuda' if torch.cuda.is_available() else 'cpu'

        model = AutoModel.from_pretrained('facebook/dinov2-base')
        model.eval()
        self.model = model.to(self.device)

        self.clf = TransformerClassifier()
        self.clf.eval()

        weight_path = "./best_classifier.pth"
        state_dict = torch.load(weight_path)
        self.clf.load_state_dict(state_dict)      
        
        self.preprocess_img = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)[0]
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py#L707
        cls_token = feats[:, 0]
        patch_tokens = feats[:, 1:]
        feats = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        # Recover the original batch size
        feats = feats.reshape(feats.shape[0]//4, 4, -1)
        return feats   
    

    def predict(self, datapoint):
        with torch.no_grad():
            image = self.preprocess_img(datapoint).to(self.device).unsqueeze(0)
            # Split the img tensor into 2x2 patches and send the patches dim to batch dim
            B, C, H, W = image.shape
            # Ensure H and W are divisible by 2
            assert H % 2 == 0 and W % 2 == 0, "H and W must be divisible by 2"
            # Reshape into 2x2 patches
            x_patches = image.view(B, C, H // 2, 2, W // 2, 2)
            x_patches = x_patches.permute(0, 3, 5, 1, 2, 4).reshape(B*4, C, H // 2, W // 2)
            
            features = self._get_features(image)
            outputs = self.clf(features).detach().cpu().numpy()
            score = outputs[:, 1][0]       
        return score
    
    