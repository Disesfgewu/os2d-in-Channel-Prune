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
            RoI 對齊後的特徵，保持批次維度
        """
        if isinstance(boxes, list):
            batch_size = len(boxes)
        else:
            batch_size = feature_map.size(0)
            # 如果 boxes 不是列表，將其轉換為列表形式
            if boxes.dim() == 3:  # [B, N, 4]
                boxes = [boxes[i] for i in range(boxes.size(0))]
            else:
                boxes = [boxes]  # 單個批次
        
        if gt_boxes is not None and not isinstance(gt_boxes, list):
            if gt_boxes.dim() == 3:  # [B, M, 4]
                gt_boxes = [gt_boxes[i] for i in range(gt_boxes.size(0))]
            else:
                gt_boxes = [gt_boxes]  # 單個批次
        
        device = feature_map.device
        channels = feature_map.size(1)
        
        # 處理每個批次
        batch_features = []
        for i in range(batch_size):
            # 確保框是浮點型
            if i < len(boxes) and boxes[i].numel() > 0:
                box_batch = boxes[i].float() if boxes[i].dtype != torch.float32 else boxes[i]
                # 確保 box_batch 是 2D 張量 [N, 4]
                if box_batch.dim() > 2:
                    # 檢查 tensor 的最後一個維度是否為 4
                    if box_batch.size(-1) == 4:
                        box_batch = box_batch.reshape(-1, 4)
                    else:
                        # 如果最後一個維度不是 4，需要先處理或轉換為有效的邊界框格式
                        # 假設這是一個格式錯誤的邊界框，我們創建一個空的邊界框代替
                        box_batch = torch.zeros((0, 4), dtype=torch.float32, device=box_batch.device)
                
                # 處理上下文增強
                if gt_boxes is not None and i < len(gt_boxes) and gt_boxes[i].numel() > 0:
                    gt_box_batch = gt_boxes[i].float() if gt_boxes[i].dtype != torch.float32 else gt_boxes[i]
                    
                    # 計算上下文框
                    N, M = box_batch.size(0), gt_box_batch.size(0)
                    box_exp = box_batch.unsqueeze(1).expand(N, M, 4)
                    gt_exp = gt_box_batch.unsqueeze(0).expand(N, M, 4)
                    
                    x1 = torch.min(box_exp[..., 0], gt_exp[..., 0])
                    y1 = torch.min(box_exp[..., 1], gt_exp[..., 1])
                    x2 = torch.max(box_exp[..., 2], gt_exp[..., 2])
                    y2 = torch.max(box_exp[..., 3], gt_exp[..., 3])
                    ctx_boxes = torch.stack([x1, y1, x2, y2], dim=2).view(-1, 4)
                    
                    # 添加批次索引
                    box_idx = torch.full((box_batch.size(0), 1), i, dtype=box_batch.dtype, device=box_batch.device)
                    ctx_idx = torch.full((ctx_boxes.size(0), 1), i, dtype=ctx_boxes.dtype, device=ctx_boxes.device)
                    
                    box_rois = torch.cat([box_idx, box_batch], dim=1).to(feature_map.device)
                    ctx_rois = torch.cat([ctx_idx, ctx_boxes], dim=1).to(feature_map.device)
                    
                    # 提取特徵
                    box_features = ops.roi_align(feature_map, box_rois, (self.output_size, self.output_size), 1.0, -1)
                    ctx_features = ops.roi_align(feature_map, ctx_rois, (self.output_size, self.output_size), 1.0, -1)
                    
                    # 計算上下文特徵的均值
                    ctx_mean = torch.mean(ctx_features.view(N, M, channels, self.output_size, self.output_size), dim=1)
                    
                    # 增強特徵 = 原始特徵 + 上下文特徵均值
                    enhanced_features = box_features + ctx_mean
                    
                    batch_features.append(enhanced_features)
                else:
                    # 只使用原始框
                    box_idx = torch.full((box_batch.size(0), 1), i, dtype=box_batch.dtype, device=box_batch.device)
                    box_rois = torch.cat([box_idx, box_batch], dim=1).to(feature_map.device)
                    box_features = ops.roi_align(feature_map, box_rois, (self.output_size, self.output_size), 1.0, -1)
                    batch_features.append(box_features)
            else:
                # 空框情況
                batch_features.append(torch.zeros((0, channels, self.output_size, self.output_size), device=device))
        
        # 返回結果，保持批次維度
        return batch_features
