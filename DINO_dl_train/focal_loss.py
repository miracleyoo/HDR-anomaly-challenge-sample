import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        初始化 Focal Loss
        :param alpha: 平衡因子，用于调整正负样本权重（默认为 0.25）
        :param gamma: 调节因子，用于调整易分样本的权重降低程度（默认为 2）
        :param reduction: 损失的聚合方式，可选 'none'、'mean' 或 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        计算 Focal Loss
        :param inputs: 模型的预测值 (logits), shape (N, 1) 或 (N, 2)。
        :param targets: 真实标签 (0 或 1), shape (N,)。
        :return: 计算的 Focal Loss
        """
        if inputs.dim() > 1 and inputs.size(1) == 2:
            # 二分类的多输出形式 (N, 2)，需要取出正样本的概率
            inputs = inputs[:, 1]

        # 将 logits 转换为概率
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # 计算交叉熵项
        ce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        
        # 计算 Focal Loss 修正项
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma

        loss = self.alpha * focal_term * ce_loss

        # 根据 reduction 聚合损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
