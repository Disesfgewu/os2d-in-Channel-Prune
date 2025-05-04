# src/lcp_channel_selector.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from src.gIoU_loss import GIoULoss

class OS2DChannelSelector:
    """
    LCP (Localization-aware Channel Pruning) 通道選擇器
    結合了DCP的判別能力和LCP的定位能力
    """
    def __init__(self, model, auxiliary_net, device='cuda', alpha=0.6, beta=0.3, gamma=0.1):
        if model is None:
            raise ValueError("Model cannot be None")
        self.model = model
        self.auxiliary_net = auxiliary_net
        self.device = device
        self.original_model = copy.deepcopy(model)
        if self.original_model is None:
            raise ValueError("Failed to create a copy of the model")
        self.original_model.eval()
        
        # 確保模型和權重都在正確的設備上
        self.model = self.model.to(device)
        self.original_model = self.original_model.to(device)
        self.auxiliary_net = self.auxiliary_net.to(device)
        
        # 確保模型權重類型一致
        if device == 'cuda':
            self.model = self.model.cuda()
            self.original_model = self.original_model.cuda()
            self.auxiliary_net = self.auxiliary_net.cuda()
        
        self.alpha = alpha  # 分類重要性權重
        self.beta = beta    # 定位重要性權重
        self.gamma = gamma  # 重建重要性權重
    
    def compute_importance(self, layer_name, images, boxes, gt_boxes, labels):
        """
        計算通道重要性，結合三個組件：
        1. 分類重要性 (DCP)
        2. 定位重要性 (LCP)
        3. 重建重要性 (共同)
        """
        return self.compute_channel_importance(layer_name, images, boxes, gt_boxes, labels,
                                              self.alpha, self.beta, self.gamma)
    
    def compute_cls_loss(self, features, boxes, labels):
        """計算分類損失"""
        features = features.to(self.device)
        boxes = [b.to(self.device) for b in boxes]
        cls_scores, _ = self.auxiliary_net(features, boxes)
        
        # 檢查 labels 是否為 None 或為空列表
        if labels is None or len(labels) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 過濾出有效的張量標籤
        valid_labels = [l for l in labels if isinstance(l, torch.Tensor) and l.numel() > 0 and l.dim() > 0]
        
        # 檢查是否有有效標籤
        if not valid_labels:
            return torch.tensor(0.0, device=self.device)
        
        # 嘗試連接有效標籤
        try:
            flat_labels = torch.cat(valid_labels, dim=0).to(self.device)
            if flat_labels.numel() > 0 and cls_scores.size(0) > 0:
                return F.cross_entropy(cls_scores, flat_labels)
            else:
                return torch.tensor(0.0, device=self.device)
        except RuntimeError:
            # 如果連接失敗，返回零損失
            return torch.tensor(0.0, device=self.device)
    
    def compute_reg_loss(self, features, boxes, gt_boxes):
        """計算回歸損失"""
        _, bbox_preds = self.auxiliary_net(features, boxes, gt_boxes)
        flat_gt_boxes = torch.cat([b for b in gt_boxes if b.numel() > 0], dim=0) if len(gt_boxes) > 0 else torch.tensor([], device=self.device)
        if flat_gt_boxes.numel() > 0 and bbox_preds.size(0) > 0:
            giou_loss_fn = GIoULoss()
            return giou_loss_fn(bbox_preds, flat_gt_boxes)
        else:
            return torch.tensor(0.0, device=self.device)
    
    def compute_channel_importance(self, layer_name, images, boxes, gt_boxes, labels,
                                  alpha=0.6, beta=0.3, gamma=0.1):
        """
        計算通道重要性，結合三個組件：
        1. 分類重要性 (DCP)
        2. 定位重要性 (LCP)
        3. 重建重要性 (共同)
        """
        self.model.eval()
        self.original_model.eval()
        
        # 獲取目標層
        target_layer = self._get_target_layer(layer_name)
        if target_layer is None:
            return None
        
        # 更新輔助網路的輸入通道數
        self.auxiliary_net.update_input_channels(target_layer.out_channels)
        
        # 獲取特徵圖
        feature_map, orig_feature_map = self._get_feature_maps(layer_name, images)
        if feature_map is None or orig_feature_map is None:
            return None
        
        # 計算各種梯度
        cls_importance = self._compute_classification_importance(layer_name, feature_map, boxes, gt_boxes, labels)
        reg_importance = self._compute_regression_importance(layer_name, feature_map, boxes, gt_boxes)
        recon_importance = self._compute_reconstruction_importance(layer_name, feature_map, orig_feature_map)
        
        # 標準化重要性分數
        cls_importance = self._normalize(cls_importance)
        reg_importance = self._normalize(reg_importance)
        recon_importance = self._normalize(recon_importance)
        
        # 組合重要性分數
        combined_importance = alpha * cls_importance + beta * reg_importance + gamma * recon_importance
        
        return torch.tensor(combined_importance, device=self.device)
    
    def _get_target_layer(self, layer_name):
        """獲取目標層"""
        parts = layer_name.split('.')
        current_module = self.model.backbone
        try:
            for part in parts:
                if part.isdigit():
                    current_module = current_module[int(part)]
                else:
                    if hasattr(current_module, part):
                        current_module = getattr(current_module, part)
                    else:
                      raise AttributeError(f"模組 {type(current_module)} 沒有屬性 {part}")

        except Exception as e:
            print(f"❌ 訪問層時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        if isinstance(current_module, nn.Conv2d):
            return current_module
        else:
            return None
    
    def _get_feature_maps(self, layer_name, images):
        """獲取原始模型和當前模型的特徵圖，並確保特徵圖有梯度"""
        feature_maps = {}
        orig_feature_maps = {}
        
        def hook_fn(maps_dict):
            def hook(name):
                def inner_hook(module, input, output):
                    maps_dict[name] = output
                return inner_hook
            return hook
        
        # 修改層名稱以匹配實際路徑
        backbone_layer_name = f"net_feature_maps.{layer_name}" 
        
        # 註冊 hook
        hook = None
        orig_hook = None
        
        # 嘗試註冊 hook
        for name, module in self.model.named_modules():
            if name == backbone_layer_name:
                print(f"✓ 找到目標層: {name}")
                hook = module.register_forward_hook(hook_fn(feature_maps)(backbone_layer_name))
                break
        
        for name, module in self.original_model.named_modules():
            if name == backbone_layer_name:
                orig_hook = module.register_forward_hook(hook_fn(orig_feature_maps)(backbone_layer_name))
                break
        
        # 確保輸入數據維度正確
        if images.dim() == 3:  # [C, H, W]
            images = images.unsqueeze(0)  # [1, C, H, W]
        
        # 確保輸入數據類型與模型一致
        images_with_grad = images.detach().clone().requires_grad_(True).to(self.device)
        
        # 正確處理 class_images - 這是關鍵修正
        class_images = [images[0].clone()]  # 列表中的元素應為 [C, H, W]，不需再套一層 unsqueeze
        class_images = [img.to(self.device) for img in class_images]
        
        print(f"Forward pass shapes - images: {images_with_grad.shape}, class_image: {class_images[0].shape}")
        
        try:
            with torch.amp.autocast('cuda', enabled=False):  # 修正警告
                with torch.no_grad():  # 禁用梯度計算以避免意外更新
                    try:
                        _ = self.model(images_with_grad, class_images=class_images)
                        _ = self.original_model(images.to(self.device), class_images=class_images)
                    except TypeError as te:
                        print(f"⚠️ 模型不接受 class_images 參數: {te}")
                        try:
                            _ = self.model(images_with_grad)
                            _ = self.original_model(images.to(self.device))
                        except Exception as e:
                            print(f"❌ 基本模式也失敗: {e}")
                            return None, None
        except Exception as e:
            print(f"❌ 前向傳播錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 嘗試使用備選方法
            print("⚠️ 嘗試獲取特徵圖的備選方法")
            try:
                return self._get_feature_maps_backup(layer_name, images)
            except Exception as backup_err:
                print(f"❌ 備選方法也失敗: {backup_err}")
                return None, None
        finally:
            if hook:
                hook.remove()
            if orig_hook:
                orig_hook.remove()
        
        # 檢查特徵圖是否成功獲取
        if backbone_layer_name not in feature_maps:
            print(f"❌ 無法獲取層 {backbone_layer_name} 的特徵圖")
            print(f"獲取到的特徵圖層: {list(feature_maps.keys())}")
            
            # 嘗試使用備選方法
            print("⚠️ 嘗試使用備選方法獲取特徵圖")
            return self._get_feature_maps_backup(layer_name, images)
        
        if backbone_layer_name not in orig_feature_maps:
            print(f"❌ 無法獲取原始模型層 {backbone_layer_name} 的特徵圖")
            print(f"獲取到的特徵圖層: {list(orig_feature_maps.keys())}")
            return None, None
        
        # 確保特徵圖在正確的設備上
        feature_maps[backbone_layer_name] = feature_maps[backbone_layer_name].to(self.device)
        orig_feature_maps[backbone_layer_name] = orig_feature_maps[backbone_layer_name].to(self.device)
        
        return feature_maps[backbone_layer_name], orig_feature_maps[backbone_layer_name]

    def _get_feature_maps_backup(self, layer_name, images):
        """獲取特徵圖的備選方法 - 使用直接 hook"""
        feature_maps = {}
        orig_feature_maps = {}
        
        def hook_fn(maps_dict):
            def inner_hook(module, input, output):
                # 保留梯度，不使用 detach()
                maps_dict[layer_name] = output
            return inner_hook
        
        # 查找匹配層
        target_layer = None
        orig_target_layer = None
        
        # 查找具有特定模式的層
        search_patterns = [
            f"net_feature_maps.{layer_name}",
            f"backbone.{layer_name}",
            layer_name
        ]
        
        for pattern in search_patterns:
            for name, module in self.model.named_modules():
                if pattern in name and isinstance(module, nn.Conv2d):
                    target_layer = module
                    print(f"✓ 找到匹配層: {name}")
                    break
            if target_layer:
                break
        
        if not target_layer:
            print(f"❌ 備選方法無法找到匹配層: {layer_name}")
            # 直接使用 get_feature_map 方法
            if hasattr(self.model, 'get_feature_map'):
                print("⚠️ 嘗試使用 model.get_feature_map() 方法")
                try:
                    # 啟用梯度計算
                    images_with_grad = images.detach().clone().requires_grad_(True).to(self.device)
                    feature_map = self.model.get_feature_map(images_with_grad)
                    
                    with torch.no_grad():
                        orig_feature_map = self.original_model.get_feature_map(images.to(self.device))
                    return feature_map, orig_feature_map
                except Exception as e:
                    print(f"❌ get_feature_map() 也失敗: {e}")
                    return None, None
            return None, None
        
        # 在原始模型中尋找相同層
        for name, module in self.original_model.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == target_layer.in_channels and module.out_channels == target_layer.out_channels:
                orig_target_layer = module
                break
        
        if not orig_target_layer:
            print("❌ 無法在原始模型中找到匹配層")
            return None, None
        
        # 註冊 hooks
        hook = target_layer.register_forward_hook(hook_fn(feature_maps))
        orig_hook = orig_target_layer.register_forward_hook(hook_fn(orig_feature_maps))
        
        try:
            # 確保輸入數據維度正確
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            # 啟用梯度計算
            images_with_grad = images.detach().clone().requires_grad_(True).to(self.device)
            
            # 前向傳播 - 當前模型啟用梯度
            with torch.enable_grad():
                _ = self.model(images_with_grad)
            
            # 原始模型不需要梯度
            with torch.no_grad():
                _ = self.original_model(images.to(self.device))
        finally:
            # 移除 hooks
            hook.remove()
            orig_hook.remove()
        
        # 檢查特徵圖是否獲取成功
        if layer_name not in feature_maps or layer_name not in orig_feature_maps:
            print("❌ 備選方法無法獲取特徵圖")
            return None, None
        
        # 確保特徵圖需要梯度
        feature_map = feature_maps[layer_name]
        if not feature_map.requires_grad:
            print("⚠️ 特徵圖沒有設置 requires_grad，正在修正...")
            feature_map = feature_map.detach().clone().requires_grad_(True)
        
        orig_feature_map = orig_feature_maps[layer_name]
        
        print(f"✓ 特徵圖獲取成功: shape={feature_map.shape}, requires_grad={feature_map.requires_grad}")
        
        return feature_map, orig_feature_map
    
    def _compute_classification_importance(self, layer_name, feature_map, boxes, gt_boxes, labels):
        """計算分類重要性"""
        target_layer = self._get_target_layer(layer_name)
        cls_gradients = []
        
        def save_gradient(grad):
            cls_gradients.append(grad.clone())
        
        target_layer.weight.requires_grad_(True)
        handle = target_layer.weight.register_hook(save_gradient)
        
        # 確保特徵圖需要梯度
        if not feature_map.requires_grad:
            feature_map = feature_map.detach().clone().requires_grad_(True)
            
        # 分類損失
        cls_loss = self.compute_cls_loss(feature_map, boxes, labels)
        if cls_loss.requires_grad:
            cls_loss.backward(retain_graph=True)
        else:
            print("Warning: Classification loss doesn't require gradients")
        
        handle.remove()
        target_layer.weight.grad = None  # 清除梯度
        
        # 計算重要性
        if not cls_gradients:
            return np.zeros(target_layer.out_channels)
        
        cls_grad = cls_gradients[0]
        return torch.sum(torch.abs(cls_grad), dim=(1, 2, 3)).cpu().numpy()
    
    def _compute_regression_importance(self, layer_name, feature_map, boxes, gt_boxes):
        """計算定位重要性"""
        target_layer = self._get_target_layer(layer_name)
        reg_gradients = []
        
        def save_gradient(grad):
            reg_gradients.append(grad.clone())
            
        # 確保特徵圖需要梯度
        if not feature_map.requires_grad:
            feature_map = feature_map.detach().clone().requires_grad_(True)
        
        target_layer.weight.requires_grad_(True)
        handle = target_layer.weight.register_hook(save_gradient)
        
        # 定位損失
        reg_loss = self.compute_reg_loss(feature_map, boxes, gt_boxes)
        reg_loss.backward(retain_graph=True)
        
        handle.remove()
        target_layer.weight.grad = None  # 清除梯度
        
        # 計算重要性
        if not reg_gradients:
            return np.zeros(target_layer.out_channels)
        
        reg_grad = reg_gradients[0]
        return torch.sum(torch.abs(reg_grad), dim=(1, 2, 3)).cpu().numpy()
    
    def _compute_reconstruction_importance(self, layer_name, feature_map, orig_feature_map):
        """計算重建重要性"""
        target_layer = self._get_target_layer(layer_name)
        
        # 重建損失
        if not feature_map.requires_grad:
            feature_map.requires_grad_(True)
        reconstruction_loss = F.mse_loss(feature_map, orig_feature_map.detach())
        reconstruction_loss.backward(retain_graph=True)
        
        # 計算重要性
        if target_layer.weight.grad is None:
            return np.zeros(target_layer.out_channels)
        
        recon_grad = target_layer.weight.grad
        return torch.sum(torch.abs(recon_grad), dim=(1, 2, 3)).cpu().numpy()
    
    def _normalize(self, x):
        """標準化重要性分數"""
        if np.sum(x) == 0:
            return x
        return x / np.sum(x)
    
    def select_channels(self, layer_name, importance_scores, prune_ratio):
        """根據重要性分數選擇通道"""
        if importance_scores is None:
            return None
        
        target_layer = self._get_target_layer(layer_name)
        if target_layer is None:
            return None
        
        # 計算保留通道數
        num_channels = importance_scores.shape[0] if isinstance(importance_scores, torch.Tensor) else len(importance_scores)
        keep_num = int(num_channels * (1 - prune_ratio))
        keep_num = max(1, keep_num)  # 至少保留一個通道
        
        # 選擇重要通道
        keep_indices = torch.topk(importance_scores, keep_num).indices.cpu().numpy()
        keep_indices = np.sort(keep_indices)  # 按索引順序排列
        
        return keep_indices
