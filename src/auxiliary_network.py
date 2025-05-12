import torch.nn as nn
import torch
import torch.nn.functional as F
from src.contextual_roi_align import ContextualRoIAlign

class AuxiliaryNetwork(nn.Module):
    """
    用於監督通道選擇的輔助網絡，具有動態記憶體管理功能
    """
    def __init__(self, in_channels=64, hidden_channels=128, num_classes=20, batch_size=None, memory_efficient=True):
        """
        初始化輔助網絡
        
        Args:
            in_channels (int): 輸入通道數
            hidden_channels (int): 隱藏層通道數
            num_classes (int): 類別數量
            batch_size (int, optional): 預設批次大小，用於分批處理
            memory_efficient (bool): 是否啟用記憶體效率模式
        """
        super(AuxiliaryNetwork, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 分類頭
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        )
        
        # 邊界框回歸頭
        self.reg_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, kernel_size=1)
        )
        
        self.contextual_roi_align = ContextualRoIAlign(output_size=7)
        
        # 記憶體管理相關參數
        self.memory_efficient = memory_efficient
        self.default_batch_size = batch_size or 4  # 預設分批大小
        self.fallback_device = "cpu"  # 記憶體不足時的後備設備
    
    def update_input_channels(self, new_channels):
        """動態更新輸入通道 (保持梯度連續性)"""
        if not isinstance(new_channels, int) or new_channels <= 0:
            raise ValueError(f"無效的通道數: {new_channels}，必須是正整數")
        
        if self.conv.in_channels == new_channels:
            return  # 無變化時快速返回
        
        # 保留原始設備信息
        device = self.conv.weight.device
        old_weight = self.conv.weight.data
        old_bias = self.conv.bias.data if self.conv.bias is not None else None
        
        # 創建新卷積層
        new_conv = nn.Conv2d(
            new_channels,
            self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            bias=self.conv.bias is not None
        ).to(device)
        
        # 智能權重移植策略
        with torch.no_grad():
            if new_channels <= self.conv.in_channels:
                # 通道減少：直接裁剪
                new_conv.weight.copy_(old_weight[:, :new_channels])
            else:
                # 通道增加：混合策略
                # 1. 移植現有通道
                new_conv.weight[:, :self.conv.in_channels].copy_(old_weight)
                # 2. 新通道使用現有通道的均值初始化
                channel_mean = old_weight.mean(dim=1, keepdim=True)
                new_channels_to_add = new_channels - self.conv.in_channels
                new_conv.weight[:, self.conv.in_channels:].copy_(
                    channel_mean.expand(-1, new_channels_to_add, -1, -1)
                )
            
            # 偏置項處理
            if old_bias is not None:
                new_conv.bias.copy_(old_bias)
        
        # 無縫替換卷積層
        self.conv = new_conv
        print(f"✓ 輔助網路輸入通道更新: {self.conv.in_channels} → {new_channels}")
    
    def _process_batch_with_memory_management(self, func, *args, **kwargs):
        """
        使用記憶體管理執行函數，處理 CUDA 記憶體不足的情況
        
        Args:
            func: 要執行的函數
            *args, **kwargs: 函數的參數
            
        Returns:
            函數的返回值
        """
        try:
            # 嘗試直接執行函數
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA 記憶體不足，嘗試優化策略...")
                
                # 策略1: 清理快取
                torch.cuda.empty_cache()
                
                try:
                    # 再次嘗試執行
                    return func(*args, **kwargs)
                except RuntimeError as e2:
                    if "CUDA out of memory" in str(e2):
                        # 策略2: 移至 CPU
                        print(f"仍然記憶體不足，將操作移至 CPU...")
                        
                        # 保存當前設備
                        original_device = next(self.parameters()).device
                        
                        # 將模型移至 CPU
                        self.to(self.fallback_device)
                        
                        # 將輸入數據移至 CPU
                        cpu_args = []
                        for arg in args:
                            if isinstance(arg, torch.Tensor):
                                cpu_args.append(arg.to(self.fallback_device))
                            elif isinstance(arg, list) and all(isinstance(item, torch.Tensor) for item in arg if item is not None):
                                cpu_args.append([item.to(self.fallback_device) if item is not None else None for item in arg])
                            else:
                                cpu_args.append(arg)
                        
                        cpu_kwargs = {}
                        for key, value in kwargs.items():
                            if isinstance(value, torch.Tensor):
                                cpu_kwargs[key] = value.to(self.fallback_device)
                            elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value if item is not None):
                                cpu_kwargs[key] = [item.to(self.fallback_device) if item is not None else None for item in value]
                            else:
                                cpu_kwargs[key] = value
                        
                        # 在 CPU 上執行
                        result = func(*cpu_args, **cpu_kwargs)
                        
                        # 將模型移回原始設備
                        self.to(original_device)
                        
                        # 將結果移回原始設備
                        if isinstance(result, torch.Tensor):
                            result = result.to(original_device)
                        elif isinstance(result, tuple) and all(isinstance(item, torch.Tensor) for item in result if item is not None):
                            result = tuple(item.to(original_device) if item is not None else None for item in result)
                        elif isinstance(result, list) and all(isinstance(item, torch.Tensor) for item in result if item is not None):
                            result = [item.to(original_device) if item is not None else None for item in result]
                        
                        return result
                    else:
                        raise
            else:
                raise
            
    def _process_on_cpu(self, features, boxes=None, gt_boxes=None):
        """
        將所有計算移至 CPU 執行
        
        Args:
            features (Tensor): 特徵圖，形狀為 [batch_size, channels, height, width]
            boxes (Tensor or list, optional): 預設框
            gt_boxes (Tensor or list, optional): 真實框
            
        Returns:
            tuple: 在 CPU 上計算的結果
        """
        print(f"執行完全 CPU 處理...")
        
        # 保存當前設備
        original_device = next(self.parameters()).device
        
        # 將模型移至 CPU
        self.to('cpu')
        
        # 將特徵移至 CPU
        cpu_features = features.cpu() if features is not None else None
        
        # 處理框數據
        cpu_boxes = None
        if boxes is not None:
            if isinstance(boxes, torch.Tensor):
                cpu_boxes = boxes.cpu()
            elif isinstance(boxes, list):
                cpu_boxes = []
                for box in boxes:
                    if box is not None:
                        if isinstance(box, torch.Tensor):
                            cpu_boxes.append(box.cpu())
                        else:
                            cpu_boxes.append(box)
                    else:
                        cpu_boxes.append(None)
        
        # 處理真實框數據
        cpu_gt_boxes = None
        if gt_boxes is not None:
            if isinstance(gt_boxes, torch.Tensor):
                cpu_gt_boxes = gt_boxes.cpu()
            elif isinstance(gt_boxes, list):
                cpu_gt_boxes = []
                for gt_box in gt_boxes:
                    if gt_box is not None:
                        if isinstance(gt_box, torch.Tensor):
                            cpu_gt_boxes.append(gt_box.cpu())
                        else:
                            cpu_gt_boxes.append(gt_box)
                    else:
                        cpu_gt_boxes.append(None)
        
        # 使用無梯度上下文在 CPU 上執行
        with torch.no_grad():
            # 前向傳播
            x = self.conv(cpu_features)
            x = self.bn(x)
            x = self.relu(x)
            
            if cpu_boxes is not None:
                # 使用 ContextualRoIAlign 提取框特徵
                roi_features_list = self.contextual_roi_align(x, cpu_boxes, cpu_gt_boxes)
                
                # 處理每個批次的特徵
                batch_size = len(roi_features_list)
                cls_scores_list = []
                bbox_preds_list = []
                
                for i in range(batch_size):
                    if roi_features_list[i].size(0) > 0:
                        # 應用分類和回歸頭
                        batch_cls_scores = self.cls_head(roi_features_list[i])
                        batch_bbox_preds = self.reg_head(roi_features_list[i])
                        cls_scores_list.append(batch_cls_scores)
                        bbox_preds_list.append(batch_bbox_preds)
                    else:
                        # 處理空框情況
                        cls_scores_list.append(torch.zeros((0, self.cls_head[-1].out_channels), device='cpu'))
                        bbox_preds_list.append(torch.zeros((0, 4), device='cpu'))
                
                result = (cls_scores_list, bbox_preds_list)
            else:
                # 如果沒有框，直接返回特徵圖的分類和回歸結果
                cls_scores = self.cls_head(x)
                bbox_preds = self.reg_head(x)
                result = (cls_scores, bbox_preds)
        
        # 將模型移回原始設備
        self.to(original_device)
        
        # 將結果移回原始設備
        if isinstance(result, tuple) and len(result) == 2:
            cls_scores, bbox_preds = result
            
            # 處理分類分數
            if isinstance(cls_scores, list):
                cls_scores = [score.to(original_device) for score in cls_scores]
            else:
                cls_scores = cls_scores.to(original_device)
                
            # 處理邊界框預測
            if isinstance(bbox_preds, list):
                bbox_preds = [pred.to(original_device) for pred in bbox_preds]
            else:
                bbox_preds = bbox_preds.to(original_device)
                
            result = (cls_scores, bbox_preds)
        elif isinstance(result, torch.Tensor):
            result = result.to(original_device)
        
        print(f"CPU 處理完成，結果已移回 {original_device}")
        return result

    def _process_in_batches(self, features, boxes=None, gt_boxes=None, batch_size=None):
        """
        分批處理特徵和框
        
        Args:
            features (Tensor): 特徵圖
            boxes (list, optional): 預設框列表
            gt_boxes (list, optional): 真實框列表
            batch_size (int, optional): 批次大小
            
        Returns:
            tuple: (cls_scores_list, bbox_preds_list)
        """
        if batch_size is None:
            batch_size = self.default_batch_size
        
        # 初始化結果列表
        cls_scores_list = []
        bbox_preds_list = []
        
        # 獲取總批次數
        total_batches = features.size(0)
        
        # 分批處理
        for i in range(0, total_batches, batch_size):
            # 獲取當前批次
            try:
                batch_end = min(i + batch_size, total_batches)
                batch_features = features[i:batch_end]
                
                # 處理框
                batch_boxes = None
                if boxes is not None:
                    batch_boxes = boxes[i:batch_end] if isinstance(boxes, list) else None
                
                batch_gt_boxes = None
                if gt_boxes is not None:
                    batch_gt_boxes = gt_boxes[i:batch_end] if isinstance(gt_boxes, list) else None
                
                # 執行前向傳播
                with torch.no_grad():
                    x = self.conv(batch_features)
                    x = self.bn(x)
                    x = self.relu(x)
                    
                    if batch_boxes is not None:
                        # 使用 ContextualRoIAlign 提取框特徵
                        roi_features_list = self.contextual_roi_align(x, batch_boxes, batch_gt_boxes)
                        
                        # 處理每個批次的特徵
                        for j in range(len(roi_features_list)):
                            if roi_features_list[j].size(0) > 0:
                                # 應用分類和回歸頭
                                batch_cls_scores = self.cls_head(roi_features_list[j])
                                batch_bbox_preds = self.reg_head(roi_features_list[j])
                                cls_scores_list.append(batch_cls_scores)
                                bbox_preds_list.append(batch_bbox_preds)
                            else:
                                # 處理空框情況
                                cls_scores_list.append(torch.zeros((0, self.cls_head[-1].out_channels), device=x.device))
                                bbox_preds_list.append(torch.zeros((0, 4), device=x.device))
                    else:
                        # 如果沒有框，直接返回特徵圖的分類和回歸結果
                        batch_cls_scores = self.cls_head(x)
                        batch_bbox_preds = self.reg_head(x)
                        cls_scores_list.append(batch_cls_scores)
                        bbox_preds_list.append(batch_bbox_preds)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and batch_size > 1:
                    # 減小批次大小並重試
                    print(f"批次大小 {batch_size} 導致記憶體不足，嘗試更小的批次大小...")
                    new_batch_size = max(1, batch_size // 2)
                    return self._process_in_batches(features, boxes, gt_boxes, new_batch_size)
                elif "CUDA out of memory" in str(e) and batch_size == 1:
                    # 批次大小已經是1，移至CPU
                    print(f"即使批次大小為1也記憶體不足，移至CPU...")
                    return self._process_on_cpu(features, boxes, gt_boxes)
                else:
                    raise
        
        return cls_scores_list, bbox_preds_list
    
    def forward(self, features, boxes=None, gt_boxes=None):
        """
        前向傳播函數，具有記憶體管理功能
        
        Args:
            features (Tensor): 特徵圖，形狀為 [batch_size, channels, height, width]
            boxes (Tensor or list, optional): 預設框，形狀為 [batch_size, num_boxes, 4] 或者是這種張量的列表
            gt_boxes (Tensor or list, optional): 真實框，形狀為 [batch_size, num_boxes, 4] 或者是這種張量的列表
            
        Returns:
            tuple: (cls_scores, bbox_preds) - 分類分數和邊界框預測
        """
        # 檢查是否為空列表
        if boxes is not None and isinstance(boxes, list) and (len(boxes) == 0 or all(b is None for b in boxes)):
            boxes = None
            
        # 處理張量格式的boxes (非列表)
        if boxes is not None and isinstance(boxes, torch.Tensor) and boxes.dim() == 3:
            # 將張量轉換為列表格式 [batch_size, num_boxes, 4] -> list of [num_boxes, 4]
            boxes = [boxes[i] for i in range(boxes.size(0))]
        
        if self.memory_efficient:
            # 使用記憶體管理執行前向傳播
            def _forward():
                x = self.conv(features)
                x = self.bn(x)
                x = self.relu(x)
                
                if boxes is not None:
                    # 使用 ContextualRoIAlign 提取框特徵
                    roi_features_list = self.contextual_roi_align(x, boxes, gt_boxes)
                    
                    # 處理每個批次的特徵
                    batch_size = len(roi_features_list)
                    cls_scores_list = []
                    bbox_preds_list = []
                    
                    for i in range(batch_size):
                        if roi_features_list[i].size(0) > 0:
                            # 應用分類和回歸頭
                            batch_cls_scores = self.cls_head(roi_features_list[i])
                            batch_bbox_preds = self.reg_head(roi_features_list[i])
                            cls_scores_list.append(batch_cls_scores)
                            bbox_preds_list.append(batch_bbox_preds)
                        else:
                            # 處理空框情況
                            cls_scores_list.append(torch.zeros((0, self.cls_head[-1].out_channels), device=x.device))
                            bbox_preds_list.append(torch.zeros((0, 4), device=x.device))
                    
                    return cls_scores_list, bbox_preds_list
                
                # 如果沒有框，直接返回特徵圖的分類和回歸結果
                cls_scores = self.cls_head(x)
                bbox_preds = self.reg_head(x)
                
                return cls_scores, bbox_preds
            
            try:
                return self._process_batch_with_memory_management(_forward)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"嘗試分批處理...")
                    return self._process_in_batches(features, boxes, gt_boxes)
                else:
                    raise
        else:
            # 不使用記憶體管理，直接執行前向傳播
            x = self.conv(features)
            x = self.bn(x)
            x = self.relu(x)
            
            if boxes is not None:
                # 使用 ContextualRoIAlign 提取框特徵
                roi_features_list = self.contextual_roi_align(x, boxes, gt_boxes)
                
                # 處理每個批次的特徵
                batch_size = len(roi_features_list)
                cls_scores_list = []
                bbox_preds_list = []
                
                for i in range(batch_size):
                    if roi_features_list[i].size(0) > 0:
                        # 應用分類和回歸頭
                        batch_cls_scores = self.cls_head(roi_features_list[i])
                        batch_bbox_preds = self.reg_head(roi_features_list[i])
                        cls_scores_list.append(batch_cls_scores)
                        bbox_preds_list.append(batch_bbox_preds)
                    else:
                        # 處理空框情況
                        cls_scores_list.append(torch.zeros((0, self.cls_head[-1].out_channels), device=x.device))
                        bbox_preds_list.append(torch.zeros((0, 4), device=x.device))
                
                return cls_scores_list, bbox_preds_list
            
            # 如果沒有框，直接返回特徵圖的分類和回歸結果
            cls_scores = self.cls_head(x)
            bbox_preds = self.reg_head(x)
            
            return cls_scores, bbox_preds
