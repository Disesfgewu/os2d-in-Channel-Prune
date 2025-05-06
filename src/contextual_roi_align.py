import torch
import torch.nn as nn
import torchvision.ops as ops

class ContextualRoIAlign(nn.Module):
    def __init__(self, output_size=7):
        super().__init__()
        self.output_size = output_size

    def _to_bbox_tensor(self, box_obj):
        # 支援 BoxList 或 tensor
        if hasattr(box_obj, "bbox_xyxy"):
            print(f"[LOG] BoxList detected, bbox_xyxy.shape={box_obj.bbox_xyxy.shape}")
            return box_obj.bbox_xyxy.float()
        elif isinstance(box_obj, torch.Tensor):
            print(f"[LOG] Tensor box detected, shape={box_obj.shape}")
            return box_obj.float()
        else:
            print(f"[ERROR] Unknown box type: {type(box_obj)}")
            raise TypeError(f"Unknown box type: {type(box_obj)}")

    def _make_rois(self, box_list, device, dtype):
        rois = []
        for i, b in enumerate(box_list):
            b = self._to_bbox_tensor(b)
            print(f"[LOG] make_rois: batch_idx={i}, box.shape={b.shape}, numel={b.numel()}")
            if b.numel() == 0:
                print(f"[LOG] make_rois: batch_idx={i} is empty, skip")
                continue
            if b.dim() == 1:
                b = b.unsqueeze(0)
            idx = torch.full((b.size(0), 1), i, dtype=dtype, device=device)
            rois.append(torch.cat([idx, b], dim=1))
        if rois:
            all_rois = torch.cat(rois, dim=0)
            print(f"[LOG] make_rois: final rois.shape={all_rois.shape}")
            return all_rois
        else:
            print(f"[LOG] make_rois: all empty, return zeros")
            return torch.zeros((0, 5), dtype=dtype, device=device)

    def forward(self, feature_map, boxes, gt_boxes=None):
        device = feature_map.device
        dtype = feature_map.dtype
        batch_size = feature_map.size(0)

        print(f"[LOG] ContextualRoIAlign.forward: batch_size={batch_size}")
        print(f"[LOG] boxes types: {[type(b) for b in boxes]}")
        if gt_boxes is not None:
            print(f"[LOG] gt_boxes types: {[type(b) for b in gt_boxes]}")

        boxes = [self._to_bbox_tensor(b) for b in boxes]
        if gt_boxes is not None:
            gt_boxes = [self._to_bbox_tensor(b) for b in gt_boxes]

        # 普通 RoI Align
        if gt_boxes is None:
            rois = self._make_rois(boxes, device, dtype)
            print(f"[LOG] RoIAlign: rois.shape={rois.shape}")
            if rois.size(0) == 0:
                print(f"[LOG] RoIAlign: empty rois, return empty tensor")
                return torch.empty((0, feature_map.size(1), self.output_size, self.output_size), device=device, dtype=dtype)
            out = ops.roi_align(feature_map, rois, (self.output_size, self.output_size), 1.0, -1)
            print(f"[LOG] RoIAlign: output.shape={out.shape}")
            return out

        # 上下文增強 RoI Align
        context_rois = []
        max_boxes = 5  # <--- 這裡限制最多只取 10 個 box
        for i in range(batch_size):
            box_batch = boxes[i]
            gt_box_batch = gt_boxes[i]
            # 限制每個 batch 最多只取前 10 個 box
            box_batch = box_batch[:max_boxes]
            gt_box_batch = gt_box_batch[:max_boxes]
            print(f"[LOG] context batch {i}: box.shape={box_batch.shape}, gt_box.shape={gt_box_batch.shape}")
            if box_batch.numel() == 0 or gt_box_batch.numel() == 0:
                print(f"[LOG] context batch {i}: empty box or gt_box, skip")
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
            print(f"[LOG] context batch {i}: ctx_boxes.shape={ctx_boxes.shape}")

        if context_rois:
            context_rois = torch.cat(context_rois, dim=0)
            print(f"[LOG] context_rois.shape={context_rois.shape}")
        else:
            print(f"[LOG] context_rois: all empty")
            context_rois = torch.zeros((0, 5), dtype=dtype, device=device)
        context_features = ops.roi_align(feature_map, context_rois, (self.output_size, self.output_size), 1.0, -1)
        print(f"[LOG] context_features.shape={context_features.shape}")

        box_rois = self._make_rois(boxes, device, dtype)
        print(f"[LOG] box_rois.shape={box_rois.shape}")
        if box_rois.size(0) == 0:
            print(f"[LOG] box_rois: empty, return context_features")
            return context_features
        box_features = ops.roi_align(feature_map, box_rois, (self.output_size, self.output_size), 1.0, -1)
        print(f"[LOG] box_features.shape={box_features.shape}")

        if context_features.size(0) > 0:
            context_mean = torch.mean(context_features, dim=0, keepdim=True).expand_as(box_features)
            enhanced_features = box_features + context_mean
            print(f"[LOG] enhanced_features.shape={enhanced_features.shape}")
            return enhanced_features
        else:
            print(f"[LOG] context_features empty, return box_features")
            return box_features
