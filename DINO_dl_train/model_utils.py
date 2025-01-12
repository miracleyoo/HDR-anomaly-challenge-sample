import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel

def get_dino_model(dino_name='facebook/dinov2-base'):
    model = AutoModel.from_pretrained(dino_name)
    model.eval()  
    return model

    
def get_feats_and_meta(dloader, model, device, ignore_feats=False):
    all_feats = None
    labels = []

    for img, lbl in tqdm(dloader, desc="Extracting features"):
        with torch.no_grad():
            # Split the img tensor into 2x2 patches and send the patches dim to batch dim
            B, C, H, W = img.shape
            # Ensure H and W are divisible by 2
            assert H % 2 == 0 and W % 2 == 0, "H and W must be divisible by 2"
            
            # Reshape into 2x2 patches
            x_patches = img.view(B, C, H // 2, 2, W // 2, 2)
            x_patches = x_patches.permute(0, 3, 5, 1, 2, 4).reshape(B*4, C, H // 2, W // 2)
            
            feats = None
            if not ignore_feats:
                feats = model(x_patches.to(device))[0]
                # https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py#L707
                cls_token = feats[:, 0]
                patch_tokens = feats[:, 1:]
                feats = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1).cpu().numpy()
                # Recover the original batch size
                feats = feats.reshape(B, 4, -1)
                # feats = feats.mean(dim=1)
            if all_feats is None:
                all_feats = feats
            else:
                all_feats = np.concatenate((all_feats, feats), axis=0) if feats is not None else all_feats
        # print("all_feats.shape", all_feats.shape)
        labels.extend(lbl.cpu().numpy().tolist())
        
    labels = np.array(labels)
    return all_feats, labels
