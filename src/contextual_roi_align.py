import torch
import torch.nn as nn
import torchvision.ops as ops

class ContextualRoIAlign(nn.Module):
    """執行 RoI 對齊並保持 batch 維度"""
    
    def __init__(self, output_size=7, spatial_scale=1.0):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.spatial_scale = spatial_scale

    def _to_bbox_tensor(self, box_obj):
        # 支援 BoxList、tensor 或帶批次索引的 ROI 格式
        if hasattr(box_obj, "bbox_xyxy"):
            # BoxList 格式
            return box_obj.bbox_xyxy.float()
        elif isinstance(box_obj, torch.Tensor):
            # 張量格式
            if box_obj.dim() >= 2 and box_obj.size(1) == 5:
                # 如果是 ROI 格式 [batch_idx, x1, y1, x2, y2]，去除批次索引
                print(f"[LOG] Detected ROI format box with shape {box_obj.shape}, extracting bbox")
                return box_obj[:, 1:5].float()
            else:
                # 普通框格式 [x1, y1, x2, y2]
                return box_obj.float()
        else:
            print(f"[ERROR] Unknown box type: {type(box_obj)}")
            raise TypeError(f"Unknown box type: {type(box_obj)}")

    def _make_rois(self, box_list, device, dtype):
        rois = []
        for i, b in enumerate(box_list):
            b = self._to_bbox_tensor(b)  # 這已經會處理 [N, 5] -> [N, 4] 的轉換
            print(f"[LOG] make_rois: batch_idx={i}, box.shape={b.shape}, numel={b.numel()}")
            if b.numel() == 0:
                print(f"[LOG] make_rois: batch_idx={i} is empty, skip")
                continue
            if b.dim() == 1:
                b = b.unsqueeze(0)
            # Ensure the box tensor has 4 columns
            if b.size(1) != 4:
                print(f"[WARNING] Box shape incorrect: {b.shape}, expected 4 columns")
                if b.size(1) > 4:
                    b = b[:, :4]  # Take only the first 4 columns if there are more
                else:
                    continue  # Skip this batch if the shape is invalid
            
            idx = torch.full((b.size(0), 1), i, dtype=dtype, device=device)
            rois.append(torch.cat([idx, b], dim=1))
        if rois:
            all_rois = torch.cat(rois, dim=0)
            print(f"[LOG] make_rois: final rois.shape={all_rois.shape}")
            return all_rois
        else:
            print(f"[LOG] make_rois: all empty, return zeros")
            return torch.zeros((0, 5), dtype=dtype, device=device)

    def forward(self, features, boxes, gt_boxes=None):
        """
        Args:
            features (Tensor): 特徵圖，形狀為 [B, C, H, W]
            boxes (Tensor): RoI 坐標，形狀為 [K, 5]，每行是 (batch_idx, x1, y1, x2, y2)
            gt_boxes (Tensor, optional): 用於上下文增強的GT框
            
        Returns:
            Tensor: 對齊後的特徵，形狀為 [K, C, output_size[0], output_size[1]]
        """
        device = features.device
        dtype = features.dtype
        batch_size = features.size(0)

        print(f"[LOG] ContextualRoIAlign.forward: batch_size={batch_size}")
        
        # 檢查 boxes 是否已經是 ROI 格式 [N, 5]
        if isinstance(boxes, torch.Tensor) and boxes.dim() == 2 and boxes.size(1) == 5:
            print(f"[LOG] Received boxes in ROI format: {boxes.shape}")
            rois = boxes  # 已經是 ROI 格式，直接使用
            
            # 檢查並修復無效的框
            if rois.size(0) > 0:
                # 確保 batch_idx 在有效範圍內
                rois[:, 0] = torch.clamp(rois[:, 0], min=0, max=features.size(0) - 1)
                
                # 確保坐標在有效範圍內
                h, w = features.size(2), features.size(3)
                rois[:, 1] = torch.clamp(rois[:, 1], min=0, max=w - 1)
                rois[:, 2] = torch.clamp(rois[:, 2], min=0, max=h - 1)
                rois[:, 3] = torch.clamp(rois[:, 3], min=1, max=w)
                rois[:, 4] = torch.clamp(rois[:, 4], min=1, max=h)
                
                # 確保 x1 < x2, y1 < y2
                x1, y1, x2, y2 = rois[:, 1], rois[:, 2], rois[:, 3], rois[:, 4]
                min_size = 1.0
                rois[:, 3] = torch.max(x1 + min_size, x2)
                rois[:, 4] = torch.max(y1 + min_size, y2)
                
                # 確保框在有效範圍內
                inds_inside = (
                    (rois[:, 3] > rois[:, 1]) &
                    (rois[:, 4] > rois[:, 2]) &
                    (rois[:, 1] < w) &
                    (rois[:, 2] < h) &
                    (rois[:, 3] > 0) &
                    (rois[:, 4] > 0)
                )
                
                if not torch.all(inds_inside):
                    rois = rois[inds_inside]
                    print(f"[LOG] Filtered {(~inds_inside).sum().item()} invalid boxes, {rois.shape[0]} remaining")
        else:
            # 轉換框格式
            print(f"[LOG] boxes types: {[type(b) for b in boxes]}")
            boxes = [self._to_bbox_tensor(b) for b in boxes]
            rois = self._make_rois(boxes, device, dtype)
        
        if gt_boxes is not None:
            if isinstance(gt_boxes, torch.Tensor) and gt_boxes.dim() == 2 and gt_boxes.size(1) == 5:
                print(f"[LOG] Received gt_boxes in ROI format: {gt_boxes.shape}")
                # 處理已經是 ROI 格式的 gt_boxes
                gt_rois = gt_boxes
            else:
                print(f"[LOG] gt_boxes types: {[type(b) for b in gt_boxes]}")
                gt_boxes = [self._to_bbox_tensor(b) for b in gt_boxes]

        # 普通 RoI Align
        if gt_boxes is None:
            print(f"[LOG] RoIAlign: rois.shape={rois.shape}")
            if rois.size(0) == 0:
                print(f"[LOG] RoIAlign: empty rois, return empty tensor")
                return torch.empty((0, features.size(1), self.output_size[0], self.output_size[1]), device=device, dtype=dtype)
                
            # 使用 ROI Align
            out = torch.ops.torchvision.roi_align(
                features,
                rois,
                self.spatial_scale,
                self.output_size[0] if isinstance(self.output_size, tuple) else self.output_size,
                self.output_size[1] if isinstance(self.output_size, tuple) else self.output_size,
                0,  # sampling_ratio (0表示自適應)
                False,  # aligned
            )
            print(f"[LOG] RoIAlign: output.shape={out.shape}")
            return out

        # 上下文增強 RoI Align
        context_rois = []
        max_boxes = 5  # <--- 這裡限制最多只取 5 個 box
        for i in range(batch_size):
            box_batch = boxes[i]
            gt_box_batch = gt_boxes[i]
            # 限制每個 batch 最多只取前 5 個 box
            box_batch = box_batch[:max_boxes]
            gt_box_batch = gt_box_batch[:max_boxes]
            if box_batch.numel() == 0 or gt_box_batch.numel() == 0:
                continue
            N, M = box_batch.size(0), gt_box_batch.size(0)
            box_exp = box_batch.unsqueeze(1).expand(N, M, 4)
            gt_exp = gt_box_batch.unsqueeze(0).expand(N, M, 4)
            x1 = torch.min(box_exp[..., 0], gt_exp[..., 0])
            y1 = torch.min(box_exp[..., 1], gt_exp[..., 1])
            x2 = torch.max(box_exp[..., 2], gt_exp[..., 2])
            y2 = torch.max(box_exp[..., 3], gt_exp[..., 3])
            ctx_boxes = torch.stack([x1, y1, x2, y2], dim=2).reshape(-1, 4)
            idx = torch.full((ctx_boxes.size(0), 1), i, dtype=dtype, device=device)
            context_rois.append(torch.cat([idx, ctx_boxes], dim=1))

        if context_rois:
            context_rois = torch.cat(context_rois, dim=0)
        else:
            context_rois = torch.zeros((0, 5), dtype=dtype, device=device)
        context_features = torch.ops.torchvision.roi_align(
            features,
            context_rois,
            self.spatial_scale,
            self.output_size[0] if isinstance(self.output_size, tuple) else self.output_size,
            self.output_size[1] if isinstance(self.output_size, tuple) else self.output_size,
            0,  # sampling_ratio
            False,  # aligned
        )

        box_rois = self._make_rois(boxes, device, dtype)
        box_rois = self._make_rois(boxes, device, dtype)
        if box_rois.size(0) == 0:
            return torch.empty((0, features.size(1), self.output_size[0], self.output_size[1]), device=device, dtype=dtype)
        box_features = torch.ops.torchvision.roi_align(
            box_rois,
            self.spatial_scale,
            self.output_size[0] if isinstance(self.output_size, tuple) else self.output_size,
            self.output_size[1] if isinstance(self.output_size, tuple) else self.output_size,
            0,  # sampling_ratio
            False,  # aligned
        )

        if context_features.size(0) > 0:
            context_mean = torch.mean(context_features, dim=0, keepdim=True).expand_as(box_features)
            enhanced_features = box_features + context_mean
            return enhanced_features
        else:
            return box_features
