# src/contextual_roi_align.py
import torch
import torchvision.ops as ops
import torch.nn as nn

class ContextualRoIAlign(nn.Module):
    """
    定位感知的 RoI Align 層
    支援上下文增強的特徵提取，基於 LCP 論文的設計
    """
    def __init__(self, output_size=7):
        super().__init__()
        self.output_size = output_size
    
    def forward(self, feature_map, boxes, gt_boxes=None):
        """
        前向傳播
        
        Args:
            feature_map: 特徵圖 [B, C, H, W]
            boxes: 預測框列表，每個元素形狀為 [N, 4]
            gt_boxes: 真實框列表，每個元素形狀為 [M, 4]，用於上下文增強
        
        Returns:
            RoI 對齊後的特徵 [N_total, C, output_size, output_size]
        """
        # 確保框是浮點型
        boxes = [b.float() if b.dtype != torch.float32 else b for b in boxes]
        
        batch_size = feature_map.size(0)
        device = feature_map.device
        
        # 輔助函數：創建 RoI 格式
        def make_rois(box_list):
            rois = []
            for i, b in enumerate(box_list):
                if b.numel() > 0:
                    # 處理 1D 張量的情況
                    if b.dim() == 1:
                        b = b.unsqueeze(0)  # [x1, y1, x2, y2] -> [1, 4]
                    
                    # 添加批次索引
                    idx = torch.full((b.size(0), 1), i, dtype=b.dtype, device=b.device)
                    rois.append(torch.cat([idx, b], dim=1))  # [N, 5]
            
            if rois:
                return torch.cat(rois, dim=0)
            else:
                return torch.zeros((0, 5), dtype=feature_map.dtype, device=device)
        
        # 普通 RoI Align (無上下文)
        if gt_boxes is None:
            rois = make_rois(boxes)
            
            if rois.size(0) == 0:
                return torch.empty((0, feature_map.size(1), self.output_size, self.output_size), device=device)
            
            return ops.roi_align(feature_map, rois, (self.output_size, self.output_size), 1.0, -1)
        
        # 上下文增強 RoI Align
        context_rois = []
        
        for i in range(batch_size):
            if boxes[i].numel() == 0 or gt_boxes[i].numel() == 0:
                continue
            
            box_batch = boxes[i]
            gt_box_batch = gt_boxes[i]
            
            # 計算每個預測框與每個真實框的上下文框
            N, M = box_batch.size(0), gt_box_batch.size(0)
            
            # 擴展維度以便計算
            box_exp = box_batch.unsqueeze(1).expand(N, M, 4)
            gt_exp = gt_box_batch.unsqueeze(0).expand(N, M, 4)
            
            # 計算上下文框 (最小外接矩形)
            x1 = torch.min(box_exp[..., 0], gt_exp[..., 0])
            y1 = torch.min(box_exp[..., 1], gt_exp[..., 1])
            x2 = torch.max(box_exp[..., 2], gt_exp[..., 2])
            y2 = torch.max(box_exp[..., 3], gt_exp[..., 3])
            
            ctx_boxes = torch.stack([x1, y1, x2, y2], dim=2).reshape(-1, 4)
            
            # 添加批次索引
            idx = torch.full((ctx_boxes.size(0), 1), i, dtype=ctx_boxes.dtype, device=ctx_boxes.device)
            context_rois.append(torch.cat([idx, ctx_boxes], dim=1))
        
        if context_rois:
            context_rois = torch.cat(context_rois, dim=0)
        else:
            context_rois = torch.zeros((0, 5), dtype=feature_map.dtype, device=device)
        
        # 提取上下文特徵
        # Ensure both tensors are on the same device
        context_rois = context_rois.to(feature_map.device)
        context_features = ops.roi_align(feature_map, context_rois, (self.output_size, self.output_size), 1.0, -1)
        
        # 提取原始框特徵
        box_rois = make_rois(boxes)
        
        if box_rois.size(0) == 0:
            return context_features
        
        # Ensure both tensors are on the same device
        box_rois = box_rois.to(feature_map.device)
        box_features = ops.roi_align(feature_map, box_rois, (self.output_size, self.output_size), 1.0, -1)
        
        # 論文公式6: 返回組合特徵
        if context_features.size(0) > 0:
            # 計算上下文特徵的均值並擴展
            context_mean = torch.mean(context_features, dim=0, keepdim=True).expand_as(box_features)
            
            # 增強特徵 = 原始特徵 + 上下文特徵均值
            enhanced_features = box_features + context_mean
            return enhanced_features
        else:
            return box_features
