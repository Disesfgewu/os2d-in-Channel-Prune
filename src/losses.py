import torch

import torch.nn as nn
import torch.nn.functional as F


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss for bounding box regression
    """
    def __init__(self, eps=1e-6):
        super(GIoULoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        Calculate GIoU loss between prediction and target boxes
        
        Args:
            pred (Tensor): Predicted bounding boxes in format [x1, y1, x2, y2], shape (N, 4)
            target (Tensor): Target bounding boxes in format [x1, y1, x2, y2], shape (N, 4)
            
        Returns:
            Tensor: GIoU loss
        """
        # Ensure proper shape for processing
        if len(pred.shape) > 2:
            # Check if the last dimension is 4 (bbox coordinates)
            if pred.shape[-1] == 4:
                pred = pred.reshape(-1, 4)
            else:
                # If not, we need to transpose and then reshape
                pred = pred.transpose(1, 2).reshape(-1, 4)
        if len(target.shape) > 2:
            if target.shape[-1] == 4:
                target = target.reshape(-1, 4)
            else:
                target = target.transpose(1, 2).reshape(-1, 4)
            
        # Extract coordinates
        pred_x1, pred_y1, pred_x2, pred_y2 = pred.unbind(dim=-1)
        target_x1, target_y1, target_x2, target_y2 = target.unbind(dim=-1)
        
        # Calculate area of prediction and target boxes
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        
        # Intersection
        x1 = torch.max(pred_x1, target_x1)
        y1 = torch.max(pred_y1, target_y1)
        x2 = torch.min(pred_x2, target_x2)
        y2 = torch.min(pred_y2, target_y2)
        
        # Width and height of intersection
        w = torch.clamp(x2 - x1, min=0)
        h = torch.clamp(y2 - y1, min=0)
        
        # Intersection area
        inter = w * h
        
        # Union area
        union = pred_area + target_area - inter
        
        # IoU
        iou = inter / (union + self.eps)
        
        # Find the smallest enclosing box
        encl_x1 = torch.min(pred_x1, target_x1)
        encl_y1 = torch.min(pred_y1, target_y1)
        encl_x2 = torch.max(pred_x2, target_x2)
        encl_y2 = torch.max(pred_y2, target_y2)
        
        # Width and height of enclosing box
        encl_w = torch.clamp(encl_x2 - encl_x1, min=0)
        encl_h = torch.clamp(encl_y2 - encl_y1, min=0)
        
        # Area of enclosing box
        encl_area = encl_w * encl_h
        
        # GIoU
        giou = iou - (encl_area - union) / (encl_area + self.eps)
        
        # Return mean loss (1 - GIoU)
        return torch.mean(1.0 - giou)


class LCPLoss(nn.Module):
    """
    Localization-aware Channel Pruning Loss
    結合重建誤差和輔助網絡的損失
    """
    def __init__(self, alpha=0.5, beta=0.5):
        """
        初始化 LCP 損失
        
        Args:
            alpha (float): 重建誤差的權重
            beta (float): 輔助網絡損失的權重
        """
        super(LCPLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = GIoULoss()
        
    def forward(self, cls_preds, reg_preds, cls_targets, reg_targets):
        """
        計算 LCP 損失
        
        Args:
            cls_preds (Tensor): 分類預測
            reg_preds (Tensor): 回歸預測
            cls_targets (Tensor): 分類目標
            reg_targets (Tensor): 回歸目標
        
        Returns:
            Tensor: 計算得到的損失
        """
        # 分類損失
        cls_loss = self.cls_loss(cls_preds, cls_targets)
        
        # 回歸損失 (使用 GIoU 損失)
        reg_loss = self.reg_loss(reg_preds, reg_targets)
        
        # 結合損失
        total_loss = self.alpha * cls_loss + self.beta * reg_loss
        
        return total_loss