import torch

import torch.nn.functional as F

class OS2DChannelSelector:
    """
    用於選擇重要通道的選擇器
    """
    def __init__(self, model, auxiliarynet, alpha=0.6, beta=0.4):
        """
        初始化通道選擇器
        
        Args:
            model (nn.Module): 原始模型
            auxiliarynet (nn.Module): 輔助網絡
            alpha (float): 重建誤差的權重
            beta (float): 輔助網絡損失的權重
        """
        self.net_feature_maps = model
        self.auxiliarynet = auxiliarynet
        self.alpha = alpha
        self.beta = beta
        self.logger = model.logger
        
    def get_features(self, layer_name, images, class_images=None):
        """
        獲取指定層的特徵
        
        Args:
            layer_name (str): 層名稱
            images (Tensor): 輸入圖像
        
        Returns:
            Tensor: 特徵圖
        """
        # 獲取目標層
        target_layer = self.net_feature_maps.net_feature_maps
        for part in layer_name.split('.'):
            if part.isdigit():
                target_layer = target_layer[int(part)]
            else:
                target_layer = getattr(target_layer, part)
                
        # 註冊鉤子函數
        features = []
        def hook_fn(module, input, output):
            features.append(output.detach())
        
        handle = target_layer.register_forward_hook(hook_fn)
        
        # 前向傳播
        with torch.no_grad():
            # 如果有提供class_images
            if class_images is not None:
                # 確保 class_images 格式正確
                if not isinstance(class_images, list):
                    # 如果 class_images 不是列表，將其轉換為列表
                    class_images = list(class_images)
                self.logger.info(f"1 Class images: {len(class_images)}")
                
                # 確保類別圖像具有正確的維度 [C, H, W]
                processed_class_images = []
                for img in class_images:
                    # 確保圖像是 3D [C, H, W]
                    if img.dim() == 4:  # 如果是 [B, C, H, W]
                        img = img.squeeze(0)  # 移除批次維度
                    elif img.dim() > 4 or img.dim() < 3:
                        # 處理其他不合適的形狀
                        continue
                    
                    # 現在 img 應該是 [C, H, W]
                    processed_class_images.append(img)
                
                self.logger.info(f"2 Class images: {len(processed_class_images)}")
                
                # 確保處理後的列表非空
                if not processed_class_images:
                    # 如果所有類別圖像都被跳過，使用空白圖像
                    dummy_img = torch.zeros(3, 224, 224, device=images.device)
                    processed_class_images = [dummy_img]
                
                self.logger.info(f"3 Class images: {len(processed_class_images)} , images : {images.shape}")
                self.net_feature_maps(images, class_images=processed_class_images)
            else:
                # 沒有提供 class_images 時使用假的類別圖像
                dummy_class_images = [images[0]]  # 使用第一張圖像作為假的類別圖像 (已移除批次維度)
                self.net_feature_maps(images, class_images=dummy_class_images)
            
        # 移除鉤子
        handle.remove()
        
        return features[0]
    
    def compute_importance(self, layer_name, images, boxes, class_images=None):
        """
        計算通道重要性
        
        Args:
            layer_name (str): 層名稱
            images (Tensor): 輸入圖像
            boxes (Tensor): 邊界框
            class_images (Tensor, optional): 類別圖像
        
        Returns:
            Tensor: 通道重要性分數
        """
        # 獲取特徵
        features = self.get_features(layer_name, images, class_images)
        
        # 計算分類損失
        cls_loss = self.compute_cls_loss(features, boxes)
        
        # 獲取目標層
        target_layer = self.net_feature_maps.net_feature_maps
        for part in layer_name.split('.'):
            if part.isdigit():
                target_layer = target_layer[int(part)]
            else:
                target_layer = getattr(target_layer, part)
                
        # 獲取通道數
        num_channels = target_layer.out_channels
        
        # 初始化重要性分數
        importance = torch.zeros(num_channels, device=images.device)
        
        # 對每個通道計算重要性
        for i in range(num_channels):
            # 創建掩碼
            mask = torch.ones(num_channels, device=images.device)
            mask[i] = 0
            
            # 應用掩碼
            masked_features = features * mask.view(1, -1, 1, 1)
            
            # 計算重建誤差
            reconstruction_error = F.mse_loss(masked_features, features)
            
            # 使用輔助網絡計算損失
            cls_scores, reg_preds = self.auxiliarynet(masked_features, boxes)
            aux_loss = self.compute_aux_loss(cls_scores, reg_preds, boxes)
            
            # 計算總損失
            total_loss = self.alpha * reconstruction_error + self.beta * aux_loss
            
            # 更新重要性分數
            importance[i] = total_loss
            
        return importance
    
    def compute_cls_loss(self, features, boxes):
        """
        計算分類損失
        
        Args:
            features (Tensor): 特徵圖
            boxes (Tensor): 邊界框
        
        Returns:
            Tensor: 分類損失
        """
        # 使用輔助網絡計算分類分數
        self.auxiliarynet.update_input_channels(features.size(1))
        cls_scores, _ = self.auxiliarynet(features, boxes)
        
        # 檢查 cls_scores 是列表還是張量
        if isinstance(cls_scores, list):
            # 如果是列表，處理第一個元素
            cls_scores = cls_scores[0]
        
        # 確保 cls_scores 是合適的形狀 [batch_size, num_classes, ...]
        if cls_scores.dim() > 2:
            # 如果 cls_scores 是多維的，將其重塑為 [batch_size, num_classes]
            cls_scores = cls_scores.view(cls_scores.size(0), -1)
        
        cls_targets = torch.zeros(cls_scores.size(0), dtype=torch.long, device=cls_scores.device)
        
        cls_loss = F.cross_entropy(cls_scores, cls_targets)
        
        return cls_loss
    
    def compute_aux_loss(self, cls_scores, reg_preds, boxes):
        """
        計算輔助網絡損失
        
        Args:
            cls_scores (Tensor): 分類分數
            reg_preds (Tensor): 回歸預測
            boxes (Tensor): 邊界框
        
        Returns:
            Tensor: 輔助網絡損失
        """
        # 檢查 cls_scores 是列表還是張量
        if isinstance(cls_scores, list):
            # 如果是列表，處理第一個元素
            cls_scores = cls_scores[0]
        
        # 確保 cls_scores 是合適的形狀 [batch_size, num_classes]
        if cls_scores.dim() > 2:
            # 如果 cls_scores 是多維的，將其重塑為 [batch_size, num_classes]
            cls_scores = cls_scores.view(cls_scores.size(0), -1)
        
        # 創建分類目標
        cls_targets = torch.zeros(cls_scores.size(0), dtype=torch.long, device=cls_scores.device)
        
        # 處理回歸預測和目標
        if isinstance(reg_preds, list):
            reg_preds = reg_preds[0]
        
        # 檢查 reg_preds 的維度
        if reg_preds.dim() > 2:
            # 如果是 ROI 特徵 [N, 4, H, W]，將其平均池化為 [N, 4]
            reg_preds = F.adaptive_avg_pool2d(reg_preds, 1).squeeze(-1).squeeze(-1)
        
        # 處理 boxes
        if isinstance(boxes, list):
            # 如果是列表，取第一個元素
            boxes = boxes[0]
        
        # 確保 boxes 的形狀與 reg_preds 匹配
        if boxes.dim() == 3:  # [batch_size, num_boxes, 4]
            # 使用第一個批次的所有框
            reg_targets = boxes[0]
        else:
            reg_targets = boxes
        
        # 計算損失
        cls_loss = F.cross_entropy(cls_scores, cls_targets)
        
        # 處理空張量情況
        if reg_preds.size(0) == 0 or reg_targets.size(0) == 0:
            # 如果任一張量為空，返回零張量作為回歸損失
            reg_loss = torch.tensor(0.0, device=cls_loss.device)
        else:
            # 確保 reg_targets 與 reg_preds 形狀匹配
            if reg_targets.shape[0] != reg_preds.shape[0]:
                # 如果數量不匹配，截斷或擴展
                if reg_targets.shape[0] > reg_preds.shape[0]:
                    reg_targets = reg_targets[:reg_preds.shape[0]]
                else:
                    # 擴展 reg_targets 或截斷 reg_preds
                    reg_preds = reg_preds[:reg_targets.shape[0]]
            
            # 確保列數匹配
            if reg_targets.shape[1] != reg_preds.shape[1]:
                # 如果列數不匹配，調整 reg_targets
                if reg_targets.shape[1] > reg_preds.shape[1]:
                    reg_targets = reg_targets[:, :reg_preds.shape[1]]
                else:
                    # 創建與 reg_preds 形狀匹配的零張量
                    new_reg_targets = torch.zeros_like(reg_preds)
                    new_reg_targets[:, :reg_targets.shape[1]] = reg_targets
                    reg_targets = new_reg_targets
            
            reg_loss = F.smooth_l1_loss(reg_preds, reg_targets)
        
        # 結合損失
        aux_loss = cls_loss + reg_loss
        
        return aux_loss

    def select_channels(self, layer_name, images, boxes, percentage=0.5):
        """
        選擇重要通道
        
        Args:
            layer_name (str): 層名稱
            images (Tensor): 輸入圖像
            boxes (Tensor): 邊界框
            percentage (float): 要保留的通道比例
        
        Returns:
            list: 選擇的通道索引
        """
        # 計算通道重要性
        importance = self.compute_importance(layer_name, images, boxes)
        
        # 根據重要性排序
        _, indices = torch.sort(importance, descending=True)
        
        # 選擇前N個通道
        target_layer = self.net_feature_maps
        for part in layer_name.split('.'):
            if part.isdigit():
                target_layer = target_layer[int(part)]
            else:
                target_layer = getattr(target_layer, part)
                
        num_channels = target_layer.out_channels
        num_to_keep = int(num_channels * percentage)
        selected_indices = indices[:num_to_keep].tolist()
        
        return selected_indices