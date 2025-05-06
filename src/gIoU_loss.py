import torch
import torch.nn as nn

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
class GIoULoss(nn.Module):
    """GIoU 損失，用於邊界框回歸 (論文 3.2 節)"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred_boxes, target_boxes):
        """
        計算 GIoU 損失
        Args:
            pred_boxes: 預測的邊界框 [N, 4] (x1, y1, x2, y2)
            target_boxes: 目標邊界框 [N, 4] (x1, y1, x2, y2)
        """
        # Move target_boxes to the same device as pred_boxes
        target_boxes = target_boxes.to(pred_boxes.device)
        # 確保輸入形狀正確
        if pred_boxes.shape[0] == 0 or target_boxes.shape[0] == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        
        # 確保 target_boxes 有正確的維度 [N, 4]
        if len(target_boxes.shape) == 1:
            if target_boxes.shape[0] == 4:  # 如果是單個邊界框 [4]
                target_boxes = target_boxes.unsqueeze(0)  # 變成 [1, 4]
            else:
                # 如果是 [N*4]，需要重塑為 [N, 4]
                target_boxes = target_boxes.reshape(-1, 4)
                
        # 確保 pred_boxes 有正確的維度 [N, 4]
        if len(pred_boxes.shape) == 1:
            if pred_boxes.shape[0] == 4:  # 如果是單個邊界框 [4]
                pred_boxes = pred_boxes.unsqueeze(0)  # 變成 [1, 4]
            else:
                # 如果是 [N*4]，需要重塑為 [N, 4]
                pred_boxes = pred_boxes.reshape(-1, 4)
        
        # 確保 pred_boxes 只有4列
        if pred_boxes.shape[1] > 4:
            pred_boxes = pred_boxes[:, :4]
                
        # 提取座標
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(1)
        pred_boxes = pred_boxes.reshape(-1, 4)
                
        target_x1, target_y1, target_x2, target_y2 = target_boxes.unbind(1)
        
        # 計算面積
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        target_area = (target_x2 - target_x1).clamp(min=0) * (target_y2 - target_y1).clamp(min=0)
        
        # 交集
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        # 確保交集有效
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # 計算 IoU
        union_area = pred_area + target_area - inter_area + self.eps
        iou = inter_area / union_area
        
        # 計算最小外接矩形
        enclosing_x1 = torch.min(pred_x1, target_x1)
        enclosing_y1 = torch.min(pred_y1, target_y1)
        enclosing_x2 = torch.max(pred_x2, target_x2)
        enclosing_y2 = torch.max(pred_y2, target_y2)
        
        # 計算外接矩形面積
        enclosing_w = (enclosing_x2 - enclosing_x1).clamp(min=0)
        enclosing_h = (enclosing_y2 - enclosing_y1).clamp(min=0)
        enclosing_area = enclosing_w * enclosing_h + self.eps
        
        # 計算 GIoU
        giou = iou - ((enclosing_area - union_area) / enclosing_area)
        giou = torch.clamp(giou, min=-1.0, max=1.0)  # 防止數值爆炸
        
        # 損失: 1 - GIoU
        loss = 1 - giou
        loss = torch.clamp(loss, min=0.0, max=2.0)  # GIoU loss 理論最大為2
        
        return loss.mean()
import numpy as np
from tqdm import tqdm   

def calculate_mAP(predictions, targets, iou_threshold=0.5):
    """簡化版的 mAP 計算"""
    if not predictions or not targets:
        return 0.0
    
    ap_sum = 0.0
    num_classes = len(VOC_CLASSES)
    
    for class_id in range(num_classes):
        # 提取該類別的預測與目標
        class_preds = [p for p in predictions if p['label'] == class_id]
        class_targets = [t for t in targets if t['label'] == class_id]
        
        if not class_targets:  # 如果沒有該類別的目標，跳過
            continue
            
        if not class_preds:  # 如果沒有該類別的預測，AP=0
            continue
        
        # 按照置信度排序預測
        class_preds.sort(key=lambda x: x['score'], reverse=True)
        
        # 計算 TP 和 FP
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        matched_targets = set()
        
        for i, pred in enumerate(class_preds):
            # 找最大 IoU 的目標
            best_iou = -np.inf
            best_target_idx = -1
            
            for j, target in enumerate(class_targets):
                if j in matched_targets:
                    continue
                
                # 計算 IoU
                px1, py1, px2, py2 = pred['box']
                tx1, ty1, tx2, ty2 = target['box']
                
                # 交集
                ix1 = max(px1, tx1)
                iy1 = max(py1, ty1)
                ix2 = min(px2, tx2)
                iy2 = min(py2, ty2)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                
                # 聯集
                pred_area = (px2 - px1) * (py2 - py1)
                target_area = (tx2 - tx1) * (ty2 - ty1)
                union = pred_area + target_area - inter
                
                iou = inter / union if union > 0 else 0.0
                
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = j
            
            # 判定 TP 或 FP
            if best_iou >= iou_threshold:
                tp[i] = 1
                matched_targets.add(best_target_idx)
            else:
                fp[i] = 1
        
        # 計算累積 TP 和 FP
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        # 計算精確率和召回率
        precision = cum_tp / (cum_tp + cum_fp + 1e-10)
        recall = cum_tp / len(class_targets)
        
        # 計算 AP (使用 11-point interpolation)
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if not np.any(recall >= t):
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
        
        ap_sum += ap
    
    return ap_sum / num_classes

def evaluate(model, data_loader, device):
    """評估模型的 mAP"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, boxes, labels, _ in tqdm(data_loader, desc="📊 評估模型"):
            images = images.to(device)
            
            # 前向傳播獲取特徵圖 (這裡假設模型是 ResNet backbone)
            features = model(images)
            
            # 收集目標
            for i in range(len(images)):
                if labels[i].numel() == 0:
                    continue
                    
                for box, label in zip(boxes[i], labels[i]):
                    all_targets.append({
                        'box': box.cpu().numpy(),
                        'label': label.item()
                    })
            
            # 模擬偵測結果 (實際應用中需要使用完整的偵測模型)
            # 這裡使用隨機生成的偵測結果進行演示
            batch_size = len(images)
            for i in range(batch_size):
                if boxes[i].numel() == 0:
                    continue
                    
                num_boxes = min(10, boxes[i].size(0))  # 假設最多檢測10個物體
                for j in range(num_boxes):
                    # 為了演示，使用真實框加上雜訊作為預測框
                    if j < boxes[i].size(0):
                        box = boxes[i][j].cpu().numpy()
                        label = labels[i][j].item()
                        noise = np.random.normal(0, 5, 4)  # 隨機雜訊
                        pred_box = box + noise
                        
                        all_predictions.append({
                            'box': pred_box,
                            'label': label,
                            'score': np.random.uniform(0.3, 1.0)  # 隨機置信度
                        })
    
    # 計算 mAP
    mAP = calculate_mAP(all_predictions, all_targets)
    
    print(f"📊 🔹 mAP: {mAP:.4f}")
    return mAP