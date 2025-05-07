import torch
import torch.nn as nn
import os
import time
import datetime
import logging
import traceback
import copy
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
from os2d.modeling.model import Os2dModel
from os2d.modeling.feature_extractor import build_feature_extractor
from src.lcp_channel_selector import OS2DChannelSelector
from src.gIoU_loss import GIoULoss

class Os2dModelInPrune(Os2dModel):
    """
    擴展 OS2D 模型以支持通道剪枝功能
    """
    def __init__(self, logger=None, is_cuda=True, backbone_arch="resnet50", 
                 use_group_norm=False, img_normalization=None, 
                 pretrained_path=None, pruned_checkpoint=None, **kwargs):
        # 如果沒有提供 logger，創建一個
        if logger is None:
            logger = logging.getLogger("OS2D")
        
        # 調用父類初始化
        self.device = torch.device('cuda' if is_cuda else 'cpu')
        
        super(Os2dModelInPrune, self).__init__(
            logger=logger,
            is_cuda=is_cuda,
            backbone_arch=backbone_arch,
            use_group_norm=use_group_norm,
            img_normalization=img_normalization,
            **kwargs
        )
        
        self.backbone = self.net_feature_maps
        # 將模型移至指定設備
        if is_cuda:
            self.cuda()
            self.backbone = self.backbone.cuda()
            # 確保所有子模塊都在 CUDA 上
            for module in self.modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    module.cuda()
        # 載入預訓練權重
        if pretrained_path:
            self.init_model_from_file(pretrained_path)
        self.device = torch.device('cuda' if is_cuda else 'cpu')
        self.original_device = self.device  # 保存原始設備
        
        self.teacher_model = copy.deepcopy(self)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        if pruned_checkpoint:
           self.load_checkpoint(pruned_checkpoint)
        if is_cuda:
            try:
                self.cuda()
                self.backbone = self.backbone.cuda()
                # 確保所有子模塊都在 CUDA 上
                for module in self.modules():
                    if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                        try:
                            module.cuda()
                        except RuntimeError as e:
                            print(f"⚠️ 模塊 {type(module)} 無法移至 CUDA: {e}")
                            # 如果失敗，先用 CPU
                            module.cpu()
            except RuntimeError as e:
                print(f"⚠️ CUDA 初始化失敗，使用 CPU 作為備選: {e}")
                self.device = torch.device('cpu')
                self.cpu()
                
    
    def _safe_forward(self, x, device=None, class_images=None):
        """安全的前向傳播，如果 GPU 執行失敗會fallback到CPU"""
        if x.dim() == 3:  # 如果是 [C, H, W]
            x = x.unsqueeze(0)  # 轉換為 [1, C, H, W]
        try:
            if class_images is not None:
                return self(x, class_images=class_images)
            return self(x)
        except RuntimeError as e:
            if "Input type" in str(e) and device and device.type == 'cuda':
                print("⚠️ GPU 執行失敗，嘗試使用 CPU...")
                # 暫時將模型和輸入移到 CPU
                original_device = next(self.parameters()).device
                cpu_model = self.cpu()
                cpu_x = x.cpu()
                cpu_class_images = class_images.cpu() if class_images is not None else None
                
                try:
                    # 在 CPU 上執行
                    with torch.no_grad():
                        if cpu_class_images is not None:
                            output = cpu_model(cpu_x, class_images=cpu_class_images)
                        else:
                            output = cpu_model(cpu_x)
                    
                    # 執行成功後移回 GPU
                    self.to(original_device)
                    output = output.to(original_device)
                    print("✓ CPU 執行成功，已移回 GPU")
                    return output
                
                except Exception as cpu_e:
                    print(f"❌ CPU 執行也失敗: {cpu_e}")
                    # 確保模型移回原始設備
                    self.to(original_device)
                    raise cpu_e
            else:
                raise e  # 如果不是設備問題，則重新引發錯誤
    def set_layer_out_channels(self, layer_name, new_out_channels):
        """設置指定層的輸出通道數"""
        # 尋找目標層
        target_layer = None
        for name, module in self.backbone.named_modules():
            if name == layer_name:
                target_layer = module
                break
                
        if target_layer is None:
            print(f"❌ 找不到層: {layer_name}")
            return False
            
        if not isinstance(target_layer, nn.Conv2d):
            print(f"❌ 層 {layer_name} 不是卷積層")
            return False
            
        # 創建新的卷積層
        new_conv = nn.Conv2d(
            in_channels=target_layer.in_channels,
            out_channels=new_out_channels,
            kernel_size=target_layer.kernel_size,
            stride=target_layer.stride,
            padding=target_layer.padding,
            dilation=target_layer.dilation,
            groups=target_layer.groups,
            bias=target_layer.bias is not None
        ).to(target_layer.weight.device)
        
        # 將原始權重複製到新層
        with torch.no_grad():
            # 只複製需要的通道
            min_channels = min(new_out_channels, target_layer.weight.size(0))
            new_conv.weight.data[:min_channels] = target_layer.weight.data[:min_channels]
            if target_layer.bias is not None:
                new_conv.bias.data[:min_channels] = target_layer.bias.data[:min_channels]
        
        # 設置新的層
        parts = layer_name.split('.')
        parent = self.backbone
        for i in range(len(parts)-1):
            if parts[i].isdigit():
                parent = parent[int(parts[i])]
            else:
                parent = getattr(parent, parts[i])
        setattr(parent, parts[-1], new_conv)
        
        print(f"✓ {layer_name} out_channels 強制設為 {new_out_channels}")
        
        # 同步更新 BatchNorm
        if parts[-1].startswith('conv'):
            bn_name = parts[-1].replace('conv', 'bn')
            if hasattr(parent, bn_name):
                old_bn = getattr(parent, bn_name)
                new_bn = nn.BatchNorm2d(new_out_channels).to(old_bn.weight.device)
                min_channels = min(new_out_channels, old_bn.num_features)
                new_bn.weight.data[:min_channels] = old_bn.weight.data[:min_channels]
                new_bn.bias.data[:min_channels] = old_bn.bias.data[:min_channels]
                new_bn.running_mean[:min_channels] = old_bn.running_mean[:min_channels]
                new_bn.running_var[:min_channels] = old_bn.running_var[:min_channels]
                setattr(parent, bn_name, new_bn)
                print(f"✓ {'.'.join(parts[:-1])}.{bn_name} 通道數同步設為 {new_out_channels}")
        
        # 更新下一層的輸入通道
        if parts[-1].startswith('conv'):
            conv_idx = int(parts[-1][-1])
            if conv_idx < 3:  # 只處理 conv1/conv2 到下一層
                next_conv_name = f"conv{conv_idx+1}"
                if hasattr(parent, next_conv_name):
                    old_next_conv = getattr(parent, next_conv_name)
                    new_next_conv = nn.Conv2d(
                        new_out_channels,
                        old_next_conv.out_channels,
                        old_next_conv.kernel_size,
                        old_next_conv.stride,
                        old_next_conv.padding,
                        old_next_conv.dilation,
                        old_next_conv.groups,
                        bias=old_next_conv.bias is not None
                    ).to(old_next_conv.weight.device)
                    min_channels = min(new_out_channels, old_next_conv.weight.size(1))
                    new_next_conv.weight.data[:, :min_channels] = old_next_conv.weight.data[:, :min_channels]
                    if old_next_conv.bias is not None:
                        new_next_conv.bias.data = old_next_conv.bias.data.clone()
                    setattr(parent, next_conv_name, new_next_conv)
                    print(f"✓ {'.'.join(parts[:-1])}.{next_conv_name} in_channels 同步設為 {new_out_channels}")
        
        return True
    
    def _should_skip_pruning(self, layer_name):
        """確定是否應跳過該層的剪枝以維持 OS2D 架構"""
        import re
        
        # 解析層名稱
        m = re.match(r'(layer\d+)\.(\d+)\.(conv\d+)', layer_name)
        if not m:
            print(layer_name + " 不符合預期格式，無法解析")
            return False
        
        layer_prefix, block_idx_str, conv_name = m.groups()
        block_idx = int(block_idx_str)
        
        # 檢查該層的 block
        layer = getattr(self.backbone, layer_prefix)
        block = layer[block_idx]
        
        # 如果是 layer4，更謹慎，因為它是 OS2D 的最終輸出
        if layer_prefix == 'layer4':
            return True  # 為了 OS2D 兼容性跳過剪枝 layer4
        
        # 強制跳過所有 conv3 層
        if conv_name == 'conv3':
            print(f"⚠️ 跳過剪枝 {layer_name} (原因: conv3層)")
            return True
        
        # 如果是帶有 downsample 的 block 中的 conv1，使用專用的處理方式
        if conv_name == 'conv1' and hasattr(block, 'downsample') and block.downsample is not None:
            print(f"ℹ️ 發現帶有 downsample 的 conv1: {layer_name}，將使用專用處理方式")
            return False # 
        
        # 如果是最後一個 block，跳過所有剪枝以保護跨層連接
        if block_idx == len(layer) - 1:
            return True
        
        return False
    
    def _handle_residual_connection(self, layer_name, keep_indices):
        """處理殘差連接"""
        print(f"\n🔍 開始處理殘差連接: {layer_name}")
        
        # 解析層名稱
        parts = layer_name.split('.')
        if len(parts) < 3:
            print(f"⚠️ 無效的層名稱格式: {layer_name}")
            return
        parts = layer_name.split('.')
        if parts[-1] == 'conv1':
            print(f"ℹ️ conv1 剪枝不影響 downsample 輸出通道")
            return True
        layer_str, block_idx, conv_type = parts[0], int(parts[1]), parts[2]
        
        # 獲取當前 block
        layer = getattr(self.backbone, layer_str)
        current_block = layer[block_idx]
        
        if conv_type == 'conv1':
            # 更新 conv2 的輸入通道
            next_conv_name = f"{layer_str}.{block_idx}.conv2"
            next_conv = None
            for name, module in self.backbone.named_modules():
                if name == next_conv_name and isinstance(module, nn.Conv2d):
                    next_conv = module
                    break
            
            if next_conv is not None:
                # 檢查索引範圍
                keep_indices = keep_indices[keep_indices < next_conv.weight.size(1)]
                if len(keep_indices) == 0:
                    print(f"⚠️ 所有索引都超出範圍，跳過更新 {next_conv_name}")
                    return
                # 更新 conv2 的輸入通道
                new_conv = nn.Conv2d(
                    len(keep_indices),
                    next_conv.out_channels,
                    next_conv.kernel_size,
                    next_conv.stride,
                    next_conv.padding,
                    next_conv.dilation,
                    next_conv.groups,
                    bias=next_conv.bias is not None
                ).to(next_conv.weight.device)
                
                # 更新權重，確保索引在有效範圍內
                try:
                    # 更新權重時確保維度匹配
                    if next_conv.weight.size(1) != len(keep_indices):
                        # 如果輸入通道數不匹配，需要調整權重
                        old_weight = next_conv.weight.data
                        new_weight = torch.zeros(
                            old_weight.size(0),
                            len(keep_indices),
                            old_weight.size(2),
                            old_weight.size(3),
                            device=old_weight.device
                        )
                        # 只複製有效的通道
                        valid_indices = keep_indices[keep_indices < old_weight.size(1)]
                        new_weight[:, :len(valid_indices)] = old_weight[:, valid_indices]
                        new_conv.weight.data = new_weight
                    else:
                        new_conv.weight.data = next_conv.weight.data[:, keep_indices].clone()
                    
                    if next_conv.bias is not None:
                        new_conv.bias.data = next_conv.bias.data.clone()
                except IndexError as e:
                    print(f"⚠️ 更新權重時發生索引錯誤: {e}")
                    print(f"next_conv.weight.shape: {next_conv.weight.shape}")
                    print(f"keep_indices max: {keep_indices.max()}")
                    print(f"keep_indices min: {keep_indices.min()}")
                    print(f"keep_indices shape: {keep_indices.shape}")
                    return
                
                # 替換 conv2
                parts2 = next_conv_name.split('.')
                parent = self.backbone
                for i in range(len(parts2) - 1):
                    if parts2[i].isdigit():
                        parent = parent[int(parts2[i])]
                    else:
                        parent = getattr(parent, parts2[i])
                
                setattr(parent, parts2[-1], new_conv)
                print(f"✓ 更新 conv2 輸入通道: {next_conv_name} (in_channels: {new_conv.in_channels})")
                
                # 同步更新 BatchNorm
                bn_name = next_conv_name.replace('conv', 'bn')
                if hasattr(parent, bn_name.split('.')[-1]):
                    old_bn = getattr(parent, bn_name.split('.')[-1])
                    new_bn = nn.BatchNorm2d(new_conv.out_channels).to(old_bn.weight.device)
                    new_bn.weight.data = old_bn.weight.data.clone()
                    new_bn.bias.data = old_bn.bias.data.clone()
                    new_bn.running_mean = old_bn.running_mean.clone()
                    new_bn.running_var = old_bn.running_var.clone()
                    setattr(parent, bn_name.split('.')[-1], new_bn)
                    print(f"✓ 更新 BatchNorm: {bn_name}")
        
        elif conv_type == 'conv2':
            # 更新 conv3 的輸入通道
            next_conv_name = f"{layer_str}.{block_idx}.conv3"
            next_conv = None
            for name, module in self.backbone.named_modules():
                if name == next_conv_name and isinstance(module, nn.Conv2d):
                    next_conv = module
                    break
            
            if next_conv is not None:
                # 檢查索引範圍
                keep_indices = keep_indices[keep_indices < next_conv.weight.size(1)]
                if len(keep_indices) == 0:
                    print(f"⚠️ 所有索引都超出範圍，跳過更新 {next_conv_name}")
                    return
                    
                new_conv = nn.Conv2d(
                    len(keep_indices),
                    next_conv.out_channels,
                    next_conv.kernel_size,
                    next_conv.stride,
                    next_conv.padding,
                    next_conv.dilation,
                    next_conv.groups,
                    bias=next_conv.bias is not None
                ).to(next_conv.weight.device)
                
                try:
                    new_conv.weight.data = next_conv.weight.data[:, keep_indices, :, :].clone()
                    if next_conv.bias is not None:
                        new_conv.bias.data = next_conv.bias.data.clone()
                except IndexError as e:
                    print(f"⚠️ 更新權重時發生索引錯誤: {e}")
                    print(f"next_conv.weight.shape: {next_conv.weight.shape}")
                    print(f"keep_indices max: {keep_indices.max()}")
                    print(f"keep_indices min: {keep_indices.min()}")
                    print(f"keep_indices shape: {keep_indices.shape}")
                    return
                
                parts2 = next_conv_name.split('.')
                parent = self.backbone
                for i in range(len(parts2) - 1):
                    if parts2[i].isdigit():
                        parent = parent[int(parts2[i])]
                    else:
                        parent = getattr(parent, parts2[i])
                
                setattr(parent, parts2[-1], new_conv)
                print(f"✓ 更新 conv3 輸入通道: {next_conv_name}")
    
    def _prune_conv_layer(self, layer_name, keep_indices):
        """
        剪枝指定的卷積層

        Args:
            layer_name: 要剪枝的層名稱
            keep_indices: 要保留的通道索引
        """
        # 獲取目標層
        target_layer = None
        for name, module in self.backbone.named_modules():
            if name == layer_name and isinstance(module, nn.Conv2d):
                target_layer = module
                break

        if target_layer is None:
            print(f"❌ 找不到層: {layer_name}")
            return False

        # 設置新的輸出通道數
        self.set_layer_out_channels(layer_name, len(keep_indices))

        # 處理殘差連接
        self._handle_residual_connection(layer_name, keep_indices)

        # 處理 downsample 連接（如果有）
        # parts = layer_name.split('.')
        # layer_str, block_idx, conv_type = parts[0], int(parts[1]), parts[2]
        # block = getattr(self.backbone, layer_str)[block_idx]
        # if conv_type != "conv1" and hasattr(block, "downsample") and block.downsample is not None:
        #     self._handle_downsample_connection(layer_name, keep_indices)

        return True
    
    def _reset_batchnorm_stats(self):
        """重置所有 BatchNorm 層的統計數據"""
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
        print("✓ 已重置所有 BatchNorm 層的統計數據")
    
    def prune_channel(self, layer_name, prune_ratio=0.3, images=None, boxes=None, labels=None, auxiliary_net=None):
        """
        對指定層進行通道剪枝
        
        Args:
            layer_name: 要剪枝的層名稱
            prune_ratio: 剪枝比例 (0.0-1.0)
            images, boxes, labels: 用於計算通道重要性的數據
            auxiliary_net: 輔助網路，用於評估通道重要性
        """
        # 檢查是否應跳過剪枝
        if self._should_skip_pruning(layer_name):
            print(f"⚠️ 跳過剪枝層 {layer_name}")
            return "SKIPPED"
        
        # 獲取目標層
        target_layer = None
        for name, module in self.backbone.named_modules():
            if name == layer_name and isinstance(module, nn.Conv2d):
                target_layer = module
                break
        
        if target_layer is None:
            print(f"❌ 找不到層: {layer_name}")
            return False
        
        # 計算通道重要性
        if images is not None and boxes is not None and auxiliary_net is not None:
            # 初始化通道選擇器
            channel_selector = OS2DChannelSelector(
                model=self, 
                auxiliary_net=auxiliary_net, 
                device=self.device
            )
            
            # 計算通道重要性
            importance_scores = channel_selector.compute_importance(layer_name, images, boxes, boxes, labels)
            
            # 選擇要保留的通道
            keep_indices = channel_selector.select_channels(layer_name, importance_scores, prune_ratio)
            
            if keep_indices is None:
                print(f"❌ 無法選擇 {layer_name} 的保留通道")
                return False
        else:
            # 如果沒有提供數據，則隨機選擇通道
            num_channels = target_layer.out_channels
            num_keep = int(num_channels * (1 - prune_ratio))
            keep_indices = torch.randperm(num_channels)[:num_keep]
        
        # 執行剪枝
        success = self._prune_conv_layer(layer_name, keep_indices)
        
        return success
    
    def prune_model(self, prune_ratio=0.3, images=None, boxes=None, labels=None, auxiliary_net=None, prunable_layers=None):
        """
        剪枝整個模型，根據 LCP 和 DCP 論文的方法
        
        Args:
            prune_ratio: 剪枝比例 (0.0-1.0)
            images, boxes, labels: 用於計算通道重要性的數據
            auxiliary_net: 輔助網路，用於評估通道重要性
            prunable_layers: 要剪枝的層列表，如果為 None 則自動選擇
        """
        if prunable_layers is None:
            # 主要針對 OS2D 使用的 layer2 和 layer3 進行剪枝
            prunable_layers = []
            for layer_name in ['layer2', 'layer3']:
                layer = getattr(self.backbone, layer_name)
                for block_idx in range(len(layer)):
                    # 對於每個殘差塊
                    for conv_idx in [1, 2]:  # 只剪枝 conv1 和 conv2 以維持架構
                        conv_name = f"{layer_name}.{block_idx}.conv{conv_idx}"
                        prunable_layers.append(conv_name)
        
        print(f"🔍 開始模型剪枝 (LCP + DCP)，剪枝比例: {prune_ratio}...")
        print(f"📋 可剪枝層: {prunable_layers}")
        
        # 按順序剪枝每一層
        pruned_layers = []
        for layer_name in prunable_layers:
            print(f"\n🔧 處理層: {layer_name}")
            
            # 檢查是否應跳過剪枝
            if self._should_skip_pruning(layer_name):
                print(f"⚠️ 跳過剪枝層 {layer_name}")
                continue
            
            # 剪枝層
            success = self.prune_channel(
                layer_name, 
                prune_ratio=prune_ratio, 
                images=images, 
                boxes=boxes, 
                labels=labels, 
                auxiliary_net=auxiliary_net
            )
            
            if success and success != "SKIPPED":
                pruned_layers.append(layer_name)
        
        # 重置 BatchNorm 統計數據
        self._reset_batchnorm_stats()
        
        print(f"✅ 模型剪枝完成! 共剪枝 {len(pruned_layers)}/{len(prunable_layers)} 層")
        print(f"成功剪枝的層: {pruned_layers}")
        
        return pruned_layers
    
    def visualize_model_architecture(self, output_path="model_architecture.png", input_shape=(1, 3, 224, 224)):
        """
        視覺化模型架構並保存為圖片
        
        Args:
            output_path: 輸出圖片路徑
            input_shape: 輸入張量形狀，默認為 (1, 3, 224, 224)
        """
        try:
            import torchviz
            from graphviz import Digraph
            from tqdm import tqdm
            import os
            import time
            from datetime import datetime
            import traceback
        except ImportError:
            print("請先安裝必要的套件: pip install torchviz graphviz")
            return False
        
        # 創建輸入張量
        x = torch.randn(input_shape).to(self.device)
        
        # 獲取輸出
        y = self(x)
        
        # 使用 torchviz 生成計算圖
        dot = torchviz.make_dot(y, params=dict(self.named_parameters()))
        
        # 設置圖的屬性
        dot.attr('node', fontsize='12')
        dot.attr('graph', rankdir='TB')  # 從上到下的佈局
        dot.attr('graph', size='12,12')  # 圖的大小
        
        # 保存圖片
        dot.render(output_path.replace('.png', ''), format='png', cleanup=True)
        
        print(f"✅ 模型架構已保存至: {output_path}")
        
        # 輸出模型摘要
        self._print_model_summary()
        
        return True

    def get_feature_map(self, x):
        """獲取特徵圖"""
        feature_maps = self.backbone(x)
        return feature_maps
    
    def forward(self, images=None, class_images=None, class_head=None, feature_maps=None, 
           max_boxes=100, nms_threshold=0.5, **kwargs):
        """
        支援 OS2D pipeline (class_head, feature_maps) 及標準 (images, class_images)
        加入 NMS 處理來限制框數量
        """
        import torchvision.ops

        # 先執行原始 forward
        outputs = super().forward(images, class_images, class_head, feature_maps, **kwargs)
        
        # 應用 NMS 減少框數量
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            class_scores, boxes = outputs[0], outputs[1]
            
            # 獲取用於 NMS 的分數 (使用每個框的最高類別分數)
            if class_scores.dim() == 2:  # [N, C] 標準格式
                scores, _ = class_scores.max(dim=1)
            elif class_scores.dim() == 4:  # [B, C, 4, N] OS2D 密集格式
                # 將密集格式轉換為標準格式，然後獲取分數
                scores_view = class_scores.view(class_scores.size(0), class_scores.size(1), -1)
                scores, _ = scores_view.max(dim=2)
                scores, _ = scores.max(dim=1)  # 獲取每個框在所有類別中的最大分數
            else:
                scores = torch.ones(boxes.size(0), device=boxes.device)  # 預設分數
                
            # 應用 NMS
            keep_indices = torchvision.ops.nms(
                boxes, 
                scores,
                iou_threshold=nms_threshold
            )[:max_boxes]  # 限制最大框數
            
            # 過濾輸出
            boxes = boxes[keep_indices]
            class_scores = class_scores[keep_indices] if class_scores.dim() <= 2 else class_scores[:, :, :, keep_indices]
            
            # 重建輸出元組
            if len(outputs) > 2:
                extra_outputs = []
                for extra in outputs[2:]:
                    if isinstance(extra, torch.Tensor) and extra.size(0) == outputs[0].size(0):
                        extra_outputs.append(extra[keep_indices])
                    else:
                        extra_outputs.append(extra)
                outputs = (class_scores, boxes) + tuple(extra_outputs)
            else:
                outputs = (class_scores, boxes)
        
        return outputs
    def forward(self, images=None, class_images=None, class_head=None, feature_maps=None, **kwargs):
        """
        支援 OS2D pipeline (class_head, feature_maps) 及標準 (images, class_images)
        """
        # OS2D pipeline: detection
        if class_head is not None and feature_maps is not None:
            # 調用父類的 detection forward
            return super().forward(class_head=class_head, feature_maps=feature_maps, **kwargs)
        # 標準訓練/推論
        if images is not None:
            if class_images is not None:
                return super().forward(images, class_images=class_images, **kwargs)
            else:
                return super().forward(images, **kwargs)
        raise ValueError("forward() 需要 (images) 或 (class_head, feature_maps)")

    
    def _print_model_summary(self):
        """打印模型摘要信息，包含每層的通道數"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 獲取每層的參數和通道信息
        layer_info = {}
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_info[name] = {
                    'type': 'Conv2d',
                    'params': sum(p.numel() for p in module.parameters()),
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels
                }
            elif isinstance(module, nn.BatchNorm2d):
                layer_info[name] = {
                    'type': 'BatchNorm2d',
                    'params': sum(p.numel() for p in module.parameters()),
                    'num_features': module.num_features
                }

        # 打印摘要
        print("\n====== 模型摘要 ======")
        print(f"總參數量: {total_params:,}")
        print(f"可訓練參數量: {trainable_params:,}")
        print("\n層級結構分析:")

        # 按層名排序打印
        for layer_name in sorted(layer_info.keys()):
            info = layer_info[layer_name]
            if info['type'] == 'Conv2d':
                # 檢查是否為 downsample 層
                is_downsample = 'downsample' in layer_name
                layer_type = '殘差分支' if is_downsample else '主分支'
                print(f"\n- {layer_name} ({layer_type}):")
                print(f"  類型: {info['type']}")
                print(f"  輸入通道: {info['in_channels']}")
                print(f"  輸出通道: {info['out_channels']}")
                print(f"  參數量: {info['params']:,}")
            else:  # BatchNorm2d
                print(f"\n- {layer_name}:")
                print(f"  類型: {info['type']}")
                print(f"  特徵數: {info['num_features']}")
                print(f"  參數量: {info['params']:,}")

        # 添加層級關係分析
        print("\n====== 層級連接分析 ======")
        for layer_idx in range(1, 4):  # layer1 到 layer4
            layer = getattr(self.backbone, f'layer{layer_idx}')
            print(f"\n[Layer {layer_idx}]")
            for block_idx in range(len(layer)):
                block = layer[block_idx]
                print(f"\nBlock {block_idx}:")
                # 打印主分支
                if hasattr(block, 'conv1'):
                    print(f"  Conv1: {block.conv1.in_channels} -> {block.conv1.out_channels}")
                if hasattr(block, 'conv2'):
                    print(f"  Conv2: {block.conv2.in_channels} -> {block.conv2.out_channels}")
                if hasattr(block, 'conv3'):
                    print(f"  Conv3: {block.conv3.in_channels} -> {block.conv3.out_channels}")
                # 打印殘差分支
                if hasattr(block, 'downsample') and block.downsample is not None:
                    downsample_conv = block.downsample[0]
                    print(f"  Downsample: {downsample_conv.in_channels} -> {downsample_conv.out_channels}")
                    print(f"  Downsample Type: {type(downsample_conv).__name__}")
                    
    def _normalize_batch_images(self, images, device=None, target_size=(224, 224)):
        """
        標準化處理圖像批次，確保所有圖像尺寸一致並轉換為批次張量
        
        Args:
            images: 圖像列表或單一張量
            device: 目標設備
            target_size: 目標尺寸 (H, W)
            
        Returns:
            torch.Tensor: 批次圖像張量 [B, C, H, W]
        """
        if device is None:
            device = self.device
            
        # 處理單一張量情況
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:  # [C, H, W]
                images = images.unsqueeze(0)  # [1, C, H, W]
            return images.to(device)
            
        # 處理圖像列表
        if isinstance(images, list):
            # 過濾有效圖像
            valid_images = []
            for img in images:
                if img is None or not isinstance(img, torch.Tensor):
                    continue
                
                # 確保圖像是 3D 張量 [C, H, W]
                if img.dim() == 3:
                    # 調整圖像尺寸為標準尺寸並移至設備
                    if img.shape[1] != target_size[0] or img.shape[2] != target_size[1]:
                        img = torch.nn.functional.interpolate(
                            img.unsqueeze(0),  # 添加批次維度
                            size=target_size,
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)  # 移除批次維度
                    
                    valid_images.append(img.to(device))
                
            if len(valid_images) == 0:
                return None
                
            # 堆疊為批次張量
            batch_tensor = torch.stack(valid_images)
            return batch_tensor
        
        return None
    
    def compute_classification_loss(self, outputs, class_ids):
        # 從輸出中獲取分類分數
        if isinstance(outputs, dict) and 'class_scores' in outputs:
            class_scores = outputs['class_scores']
        else:
            class_scores = outputs
            
        # 處理 OS2D 特殊的 4D 輸出格式
        if class_scores.dim() > 2:
            # 獲取批次大小和類別數
            batch_size = class_scores.size(0)
            num_classes = class_scores.size(1)
            
            # 將 [B, C, H, W] 轉換為 [B, C] 通過平均池化或最大池化
            class_scores = class_scores.view(batch_size, num_classes, -1)
            class_scores = class_scores.mean(dim=2)  # 或使用 max(dim=2)[0]
            
            print(f"✓ 將 class_scores 從 4D 轉換為 2D: {class_scores.shape}")
        
        # 只使用每個圖像的主要類別
        if isinstance(class_ids, list):
            main_class_ids = []
            for cls_id in class_ids:
                if isinstance(cls_id, torch.Tensor) and cls_id.numel() > 0:
                    main_class_ids.append(cls_id[0].unsqueeze(0))
            
            if main_class_ids:
                target = torch.cat(main_class_ids).long()
            else:
                target = torch.zeros(class_scores.size(0)).long()
        else:
            target = class_ids.long()
        
        # 確保目標在有效範圍內
        if target.max() >= class_scores.size(1):
            target = torch.clamp(target, max=class_scores.size(1)-1)
        
        # 使用交叉熵損失
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(class_scores, target)
    
    def compute_box_regression_loss(self, outputs, boxes):
        """
        計算邊界框回歸損失 - 優化處理不同數量框的情況
        
        Args:
            outputs: 模型輸出
            boxes: 目標邊界框 (可能是BoxList對象、張量或其列表)
                    
        Returns:
            torch.Tensor: 回歸損失
        """
        # 從輸出中獲取預測框
        if isinstance(outputs, dict) and 'boxes' in outputs:
            pred_boxes = outputs['boxes']
        elif isinstance(outputs, tuple):
            # 如果輸出是元組，假設第二個元素包含邊界框
            pred_boxes = outputs[1] if len(outputs) > 1 else torch.zeros(1, 4).to(self.device)
        else:
            # 如果沒有明確的邊界框輸出，使用模型的輸出
            pred_boxes = outputs
        
        # 確保 pred_boxes 是張量
        if not isinstance(pred_boxes, torch.Tensor):
            print(f"警告: pred_boxes 不是張量，而是 {type(pred_boxes)}，使用預設值")
            pred_boxes = torch.zeros(1, 4).to(self.device)
            
        # 使用 _cat_boxes_list 處理目標框
        if isinstance(boxes, list):
            target_boxes = self._cat_boxes_list(boxes, device=pred_boxes.device)
        else:
            # 處理單個 BoxList 對象或張量
            if hasattr(boxes, 'bbox_xyxy'):
                target_boxes = boxes.bbox_xyxy
            elif hasattr(boxes, 'bbox') and hasattr(boxes, 'size'):
                target_boxes = boxes.bbox
            else:
                target_boxes = boxes
        
        # 確保張量在同一設備上
        if isinstance(target_boxes, torch.Tensor):
            target_boxes = target_boxes.to(pred_boxes.device)
        else:
            print(f"警告: target_boxes 不是張量，而是 {type(target_boxes)}，使用預設值")
            target_boxes = torch.zeros(1, 4).to(pred_boxes.device)
        
        # 打印 debug 信息
        print(f"📊 Box 回歸內容檢查:")
        print(f"  pred_boxes: 形狀 {pred_boxes.shape}, 類型 {pred_boxes.dtype}, 裝置 {pred_boxes.device}")
        print(f"  target_boxes: 形狀 {target_boxes.shape}, 類型 {target_boxes.dtype}, 裝置 {target_boxes.device}")
        
        # 如果沒有有效的目標框，返回零損失
        if target_boxes.numel() == 0:
            print(f"⚠️ 目標框為空，返回零損失")
            return torch.tensor(0.0, device=pred_boxes.device)
            
        # 確保預測框和目標框的形狀匹配
        if pred_boxes.size(-1) != 4:
            pred_boxes = pred_boxes.view(-1, 4)
        if target_boxes.size(-1) != 4:
            target_boxes = target_boxes.view(-1, 4)
        
        # 檢查值範圍，避免異常值干擾回歸
        if torch.isnan(pred_boxes).any() or torch.isinf(pred_boxes).any():
            print(f"⚠️ 預測框包含 NaN 或 Inf，進行修正")
            pred_boxes = torch.nan_to_num(pred_boxes, nan=0.0, posinf=1.0, neginf=0.0)
            
        if torch.isnan(target_boxes).any() or torch.isinf(target_boxes).any():
            print(f"⚠️ 目標框包含 NaN 或 Inf，進行修正")
            target_boxes = torch.nan_to_num(target_boxes, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 處理 OS2D 密集格式 (特別處理)
        if pred_boxes.dim() > 2:
            # 如果是密集格式 [B, C, 4, N] 或 [B, C, N]
            if pred_boxes.dim() == 4:
                b, c, four, n = pred_boxes.shape
                if four == 4:
                    # 調整為標準形式 [B*C*N, 4]
                    pred_boxes = pred_boxes.permute(0, 1, 3, 2).reshape(-1, 4)
                else:
                    # 或者可能是 [B, C, N, 4]
                    pred_boxes = pred_boxes.reshape(-1, 4)
            elif pred_boxes.dim() == 3:
                # 可能是 [B, N, 4] 或 [B, C, N]
                if pred_boxes.size(2) == 4:
                    # 是 [B, N, 4]
                    pred_boxes = pred_boxes.reshape(-1, 4)
                else:
                    # 可能需要其他處理
                    print(f"⚠️ 未預期的 pred_boxes 形狀: {pred_boxes.shape}")
                    pred_boxes = pred_boxes.view(-1, 4)
        
        # 處理框數量不匹配的情況
        if pred_boxes.size(0) != target_boxes.size(0):
            print(f"⚠️ 預測框和目標框數量不匹配: {pred_boxes.size(0)} vs {target_boxes.size(0)}")
            
            # 優化：只選擇前K個預測框進行損失計算
            if pred_boxes.size(0) > target_boxes.size(0) * 5:
                # 如果預測框數遠大於目標框，隨機抽樣或選頂部
                k = min(max(100, target_boxes.size(0) * 5), pred_boxes.size(0))
                # 隨機抽樣以避免訓練偏差
                indices = torch.randperm(pred_boxes.size(0), device=pred_boxes.device)[:k]
                pred_boxes_sampled = pred_boxes[indices]
                
                # 計算所有預測框與所有目標框的 IoU
                try:
                    from torchvision.ops import box_iou
                    ious = box_iou(pred_boxes_sampled, target_boxes)
                    # 為每個預測框選擇最佳匹配的目標框
                    best_target_idx = ious.max(dim=1)[1]
                    matched_targets = target_boxes[best_target_idx]
                    
                    # 計算損失
                    loss = F.smooth_l1_loss(pred_boxes_sampled, matched_targets)
                    print(f"✓ 使用 IoU 匹配損失 (抽樣 {k}/{pred_boxes.size(0)} 框): {loss.item():.4f}")
                    return loss
                except Exception as e:
                    print(f"⚠️ IoU 計算失敗，使用備選方法: {e}")
                    # 使用最簡單的方法：只選擇與目標框數量相同的預測框
                    pred_boxes = pred_boxes[:target_boxes.size(0)]
            
            # 框數量較接近時，使用 IoU 匹配
            try:
                # 創建成本矩陣
                cost_matrix = torch.zeros(pred_boxes.size(0), target_boxes.size(0), device=pred_boxes.device)
                
                # 基於 L1 距離計算成本
                for i in range(pred_boxes.size(0)):
                    cost_matrix[i] = torch.sum(torch.abs(pred_boxes[i].unsqueeze(0) - target_boxes), dim=1)
                
                # 利用匈牙利算法匹配 (如果可用)
                try:
                    from scipy.optimize import linear_sum_assignment
                    cost_np = cost_matrix.detach().cpu().numpy()
                    pred_idx, target_idx = linear_sum_assignment(cost_np)
                    pred_idx = torch.tensor(pred_idx, device=pred_boxes.device)
                    target_idx = torch.tensor(target_idx, device=target_boxes.device)
                    
                    # 計算匹配的框之間的損失
                    matched_pred_boxes = pred_boxes[pred_idx]
                    matched_target_boxes = target_boxes[target_idx]
                    
                    loss = F.smooth_l1_loss(matched_pred_boxes, matched_target_boxes)
                    print(f"✓ 使用匈牙利算法匹配損失: {loss.item():.4f}")
                    return loss
                except (ImportError, ModuleNotFoundError):
                    print("⚠️ 匈牙利算法不可用，使用貪婪匹配")
                    
                    # 貪婪匹配
                    min_cost, min_idx = cost_matrix.min(dim=1)
                    matched_targets = target_boxes[min_idx]
                    
                    loss = F.smooth_l1_loss(pred_boxes, matched_targets)
                    print(f"✓ 使用貪婪匹配損失: {loss.item():.4f}")
                    return loss
                    
            except Exception as e:
                print(f"⚠️ 匹配計算失敗: {e}")
                # 如果所有嘗試都失敗，則使用最簡單的處理方法
                min_len = min(pred_boxes.size(0), target_boxes.size(0))
                loss = F.smooth_l1_loss(pred_boxes[:min_len], target_boxes[:min_len])
                print(f"✓ 使用簡單截斷匹配損失: {loss.item():.4f}")
                return loss
        
        # 標準情況：框數量匹配
        try:
            loss = F.smooth_l1_loss(pred_boxes, target_boxes)
            print(f"✓ 使用標準 Smooth L1 損失: {loss.item():.4f}")
            return loss
        except Exception as e:
            print(f"❌ Smooth L1 損失計算失敗: {e}")
            # 嘗試 L1 損失作為備選
            try:
                loss = F.l1_loss(pred_boxes, target_boxes)
                print(f"✓ 使用 L1 損失作為備選: {loss.item():.4f}")
                return loss
            except Exception as e2:
                print(f"❌ L1 損失也失敗: {e2}")
                return torch.tensor(1.0, device=pred_boxes.device, requires_grad=True)
    
    def _cat_boxes_list(self, boxes, device=None):
        """
        將 list of BoxList 或 tensor 轉為 [N,4] tensor，過濾空 box
        """
        valid_boxes = []
        for b in boxes:
            # BoxList 物件
            if hasattr(b, "bbox_xyxy"):
                t = b.bbox_xyxy
                if t.numel() > 0:
                    valid_boxes.append(t)
            # tensor
            elif isinstance(b, torch.Tensor) and b.numel() > 0:
                valid_boxes.append(b)
        if not valid_boxes:
            # 返回一個空 tensor [0,4]，防止 cat 報錯
            if device is None and len(boxes) > 0:
                if hasattr(boxes[0], "bbox_xyxy"):
                    device = boxes[0].bbox_xyxy.device
                elif isinstance(boxes[0], torch.Tensor):
                    device = boxes[0].device
            return torch.zeros((0, 4), device=device)
        return torch.cat(valid_boxes, dim=0)
    
    def analyze_os2d_outputs(self, outputs, targets=None):
        """
        解析 OS2D 模型輸出，提取有用的信息
        
        Args:
            outputs: 模型輸出，可能是元組或字典
            targets: 目標數據，可選
            
        Returns:
            dict: 包含解析結果的字典
        """
        results = {}
        
        # 解析輸出
        if isinstance(outputs, tuple):
            # 典型的 OS2D 輸出是一個 5 元素元組
            if len(outputs) >= 2:
                class_scores = outputs[0]
                boxes = outputs[1]
                
                results['output_type'] = 'tuple'
                results['num_elements'] = len(outputs)
                
                # 解析分類分數
                if isinstance(class_scores, torch.Tensor):
                    # 分析維度
                    if class_scores.dim() == 4:  # [B, C, 4, N]
                        results['class_scores_shape'] = list(class_scores.shape)
                        results['batch_size'] = class_scores.shape[0]
                        results['num_classes'] = class_scores.shape[1]
                        results['num_positions'] = class_scores.shape[3]
                        results['dense_format'] = True
                        
                        # 提取每個類別的最高分數
                        scores_view = class_scores.view(results['batch_size'], results['num_classes'], -1)
                        max_scores, _ = scores_view.max(dim=2)
                        results['class_confidence'] = max_scores.detach().cpu().tolist()
                        
                    elif class_scores.dim() == 2:  # [N, C]
                        results['class_scores_shape'] = list(class_scores.shape)
                        results['num_detections'] = class_scores.shape[0]
                        results['num_classes'] = class_scores.shape[1]
                        results['dense_format'] = False
                        
                        # 提取類別的最高分數
                        max_scores, pred_classes = class_scores.max(dim=1)
                        results['class_predictions'] = pred_classes.detach().cpu().tolist()
                        results['detection_scores'] = max_scores.detach().cpu().tolist()
                
                # 解析邊界框
                if isinstance(boxes, torch.Tensor):
                    results['boxes_shape'] = list(boxes.shape)
                    
                    if boxes.dim() == 2 and boxes.shape[1] == 4:  # [N, 4] 標準格式
                        results['num_boxes'] = boxes.shape[0]
                        # 提取一些框進行檢查
                        if boxes.shape[0] > 0:
                            sample_boxes = boxes[:min(5, boxes.shape[0])].detach().cpu().tolist()
                            results['sample_boxes'] = sample_boxes
                    
                    elif boxes.dim() == 4:  # [B, C, 4, N] OS2D 密集格式
                        results['dense_boxes'] = True
                        results['batch_size'] = boxes.shape[0]
                        results['num_classes'] = boxes.shape[1]
                        results['num_positions'] = boxes.shape[3]
                    
                    elif boxes.dim() == 3:  # [B, N, 4] 批次格式
                        results['dense_boxes'] = False
                        results['batch_size'] = boxes.shape[0]
                        results['num_boxes'] = boxes.shape[1]
        
        # 解析目標
        if targets is not None and isinstance(targets, dict):
            results['target_info'] = {}
            
            # 解析類別 ID
            if 'class_ids' in targets:
                class_ids = targets['class_ids']
                if isinstance(class_ids, list):
                    results['target_info']['num_classes'] = len(class_ids)
                    results['target_info']['class_ids'] = class_ids
                elif isinstance(class_ids, torch.Tensor):
                    results['target_info']['num_classes'] = class_ids.shape[0]
                    results['target_info']['class_ids'] = class_ids.detach().cpu().tolist()
            
            # 解析目標框
            if 'boxes' in targets:
                boxes = targets['boxes']
                if isinstance(boxes, list):
                    num_boxes = sum(1 for box in boxes if isinstance(box, torch.Tensor) and box.numel() > 0)
                    results['target_info']['num_boxes'] = num_boxes
                elif isinstance(boxes, torch.Tensor):
                    if boxes.dim() > 1:
                        results['target_info']['num_boxes'] = boxes.shape[0]
                        results['target_info']['boxes_shape'] = list(boxes.shape)
        
        return results
    
    def compute_losses(self, outputs, targets, class_num=0, auxiliary_net=None, use_lcp_loss=True, loss_weights=None):
        """
        計算訓練損失的組合（分類、框回歸、教師蒸餾和 LCP）
        
        Args:
            outputs: 模型輸出，字典或元組
            targets: 目標數據
            class_num: 類別數量
            auxiliary_net: 輔助網絡，用於 LCP 損失
            use_lcp_loss: 是否使用 LCP 損失
            loss_weights: 各損失的權重
            
        Returns:
            tuple: (總損失, 損失字典)
        """
        # 預設損失權重
        if loss_weights is None:
            loss_weights = {'cls': 1.0, 'box_reg': 1.0, 'teacher': 0.5, 'lcp': 0.1}
        
        # 初始化損失
        cls_loss = torch.tensor(0.0, device=self.device)
        box_loss = torch.tensor(0.0, device=self.device)
        teacher_loss = torch.tensor(0.0, device=self.device)
        lcp_loss = torch.tensor(0.0, device=self.device)
        
        # 1. 分類損失
        try:
            cls_loss = self.compute_classification_loss(outputs, targets['class_ids'])
            print(f"   ✓ 分類損失計算完成: {cls_loss.item():.4f}")
        except Exception as e:
            print(f"   ❌ 分類損失計算失敗: {e}")
        
        # 2. 框回歸損失
        try:
            box_loss = self.compute_box_regression_loss(outputs, targets['boxes'])
            print(f"   ✓ 框回歸損失計算完成: {box_loss.item():.4f}")
        except Exception as e:
            print(f"   ❌ 框回歸損失計算失敗: {e}")
        
        # 3. 教師知識蒸餾損失
        try:
            if 'teacher_outputs' in targets and targets['teacher_outputs'] is not None:
                teacher_outputs = targets['teacher_outputs']
                
                # 提取教師和學生的預測
                if isinstance(teacher_outputs, tuple) and len(teacher_outputs) >= 2:
                    teacher_scores = teacher_outputs[0]
                    teacher_boxes = teacher_outputs[1]
                    
                    if isinstance(outputs, dict):
                        student_scores = outputs.get('class_scores')
                        student_boxes = outputs.get('boxes')
                    elif isinstance(outputs, tuple) and len(outputs) >= 2:
                        student_scores = outputs[0]
                        student_boxes = outputs[1]
                    
                    # 確保預測形狀一致
                    if teacher_scores.dim() == student_scores.dim():
                        # 分類蒸餾損失 (KL 散度)
                        if student_scores.dim() <= 2:
                            cls_distill_loss = F.kl_div(
                                F.log_softmax(student_scores, dim=1),
                                F.softmax(teacher_scores, dim=1),
                                reduction='batchmean'
                            )
                        else:
                            # 處理密集格式 [B, C, 4, N]
                            b, c, f, n = student_scores.shape
                            student_flat = student_scores.view(b*c*n, f)
                            teacher_flat = teacher_scores.view(b*c*n, f)
                            cls_distill_loss = F.kl_div(
                                F.log_softmax(student_flat, dim=1),
                                F.softmax(teacher_flat, dim=1),
                                reduction='batchmean'
                            )
                        
                        # 框回歸蒸餾損失 (L2 損失)
                        box_distill_loss = F.mse_loss(student_boxes, teacher_boxes)
                        
                        # 組合蒸餾損失
                        teacher_loss = cls_distill_loss + box_distill_loss
                        print(f"   ✓ 教師損失計算完成: {teacher_loss.item():.4f}")
        except Exception as e:
            print(f"   ❌ 教師損失計算失敗: {e}")
        
        # 4. LCP 損失
        if use_lcp_loss and auxiliary_net is not None:
            try:
                # 獲取特徵圖
                if isinstance(outputs, dict) and 'images' in outputs:
                    feature_maps = self.get_feature_map(outputs['images'])
                    
                    # 使用輔助網絡計算 LCP 損失
                    if isinstance(feature_maps, torch.Tensor):
                        aux_outputs = auxiliary_net(feature_maps)
                        aux_cls_loss = self.compute_classification_loss(aux_outputs, targets['class_ids'])
                        lcp_loss = aux_cls_loss
                        print(f"   ✓ LCP 損失計算完成: {lcp_loss.item():.4f}")
            except Exception as e:
                print(f"   ❌ LCP 損失計算失敗: {e}")
        
        # 計算總損失
        total_loss = (
            loss_weights['cls'] * cls_loss +
            loss_weights['box_reg'] * box_loss +
            loss_weights['teacher'] * teacher_loss +
            loss_weights['lcp'] * lcp_loss
        )
        
        # 構建損失字典
        loss_dict = {
            'cls_loss': cls_loss,
            'box_loss': box_loss,
            'teacher_loss': teacher_loss,
            'lcp_loss': lcp_loss
        }
        
        return total_loss, loss_dict
    
    from tqdm import tqdm
    def train_one_epoch(self, train_loader, optimizer, 
                  auxiliary_net=None, device=None, 
                  print_freq=10, scheduler=None, 
                  loss_weights=None, use_lcp_loss=True, 
                  max_batches=None, max_predictions=100,
                  use_feature_pyramid=True,  # 特徵金字塔開關
                  pyramid_scales=[1.0, 0.75, 0.5],  # 金字塔尺度
                  nms_threshold=0.5,  # NMS IoU 閾值
                  apply_nms=True):  # 是否使用 NMS
        """
        訓練模型一個 epoch，支援 LCP 損失、特徵金字塔與 NMS 框數量控制
        """
        import torch
        import torch.nn.functional as F
        import torchvision.ops
        import time
        from tqdm import tqdm
        import traceback
        
        start_time = time.time()
        self.train()
        if auxiliary_net is not None:
            auxiliary_net.train()
            print(f"✓ 輔助網路設為訓練模式，輸入通道數: {auxiliary_net.get_current_channels()}")

        if device is None:
            device = self.device
        print(f"ℹ️ 使用設備: {device}")
        
        if loss_weights is None:
            loss_weights = {'cls': 1.0, 'box_reg': 1.0, 'teacher': 0.5, 'lcp': 0.1}
        
        # 獲取類別數量
        class_num = len(train_loader.dataset.get_class_ids())
        print(f"✓ 類別數量: {class_num}")
        
        # 確保模型輸出匹配類別數量
        if hasattr(self, 'classifier') and hasattr(self.classifier, 'out_features'):
            if self.classifier.out_features != class_num:
                print(f"⚠️ 更新分類器輸出維度: {self.classifier.out_features} → {class_num}")
            in_features = self.classifier.in_features
            self.classifier = nn.Linear(in_features, class_num).to(device)
        
        # 更新輔助網路（如果需要）
        if auxiliary_net is not None and hasattr(auxiliary_net, 'classifier') and hasattr(auxiliary_net.classifier, 'out_features'):
            if auxiliary_net.classifier.out_features != class_num:
                print(f"⚠️ 更新輔助網路分類器維度: {auxiliary_net.classifier.out_features} → {class_num}")
            aux_in_features = auxiliary_net.classifier.in_features
            auxiliary_net.classifier = nn.Linear(aux_in_features, class_num).to(device)
        
        # 初始化損失記錄和統計信息
        loss_history = []
        total_loss = 0
        total_cls_loss = 0
        total_box_loss = 0
        total_teacher_loss = 0
        total_lcp_loss = 0
        batch_count = 0
        
        # 確定要處理的批次數
        num_batches = len(train_loader)
        if max_batches is not None:
            num_batches = min(num_batches, max_batches)
        
        print(f"🔍 特徵金字塔狀態: {'啟用' if use_feature_pyramid else '停用'}")
        if use_feature_pyramid:
            print(f"📊 金字塔尺度: {pyramid_scales}")
        print(f"🧹 NMS 狀態: {'啟用' if apply_nms else '停用'}, 閾值: {nms_threshold}")
        
        # NMS 框框過濾函數
        def apply_nms_to_outputs(outputs, max_boxes=100, iou_threshold=0.5):
            """對模型輸出應用 NMS 以減少框數量"""
            if not isinstance(outputs, tuple) or len(outputs) < 2:
                return outputs
            
            class_scores, boxes = outputs[0], outputs[1]
            original_box_count = boxes.shape[0]
            
            # 如果框數量已經小於限制，直接返回
            if original_box_count <= max_boxes:
                return outputs
            
            # 提取分數用於 NMS
            if class_scores.dim() == 2:  # [N, C]
                nms_scores, _ = class_scores.max(dim=1)
            elif class_scores.dim() == 4:  # [B, C, 4, N] OS2D 密集格式
                # 密集格式轉換為標準格式
                scores_view = class_scores.view(class_scores.size(0), class_scores.size(1), -1)
                nms_scores, _ = scores_view.max(dim=2)
                nms_scores, _ = nms_scores.max(dim=1)
            else:
                # 預設分數
                nms_scores = torch.ones(boxes.size(0), device=boxes.device)
            
            # 應用 NMS
            keep_indices = torchvision.ops.nms(
                boxes,
                nms_scores,
                iou_threshold=iou_threshold
            )[:max_boxes]
            
            # 過濾結果
            filtered_scores = class_scores[keep_indices]
            filtered_boxes = boxes[keep_indices]
            
            # 重建輸出元組
            filtered_extras = tuple(
                extra[keep_indices] if isinstance(extra, torch.Tensor) and extra.shape[0] == original_box_count 
                else extra for extra in outputs[2:]
            ) if len(outputs) > 2 else ()
            
            filtered_outputs = (filtered_scores, filtered_boxes) + filtered_extras
            
            print(f"✓ NMS: 框數量 {original_box_count} → {len(keep_indices)} (閾值={iou_threshold}, 最大框數={max_boxes})")
            return filtered_outputs
        
        # 創建進度條
        with tqdm(range(num_batches), desc="訓練進度") as pbar:
            for batch_idx in pbar:
                try:
                    # 記錄批次開始時間
                    batch_start_time = time.time()
                    
                    # 獲取當前批次數據
                    batch_data = train_loader.get_batch(batch_idx)
                    images, class_images, loc_targets, class_targets, batch_class_ids, class_image_sizes, box_inverse_transform, batch_boxes, batch_img_size = batch_data
                    
                    print(f"✓ 批次 {batch_idx+1}/{num_batches} 數據載入完成")
                    print(f"   - 圖像形狀: {images.shape}")
                    print(f"   - 類別數量: {len(batch_class_ids)}")
                    print(f"   - 邊界框數量: {sum(1 for box in batch_boxes if isinstance(box, torch.Tensor) and box.numel() > 0)}")
                    
                    # 將數據移至指定設備
                    images = images.to(device)
                    
                    # 處理類別圖像
                    if isinstance(class_images, list):
                        class_images = [img.to(device) if isinstance(img, torch.Tensor) else img for img in class_images]
                    
                    # 更新輔助網路通道數（如果需要）
                    if auxiliary_net is not None:
                        feature_maps = self.get_feature_map(images)
                        if isinstance(feature_maps, torch.Tensor):
                            current_channels = feature_maps.size(1)
                            if auxiliary_net.get_current_channels() != current_channels:
                                print(f"✓ 更新輔助網路輸入通道: {auxiliary_net.get_current_channels()} → {current_channels}")
                                auxiliary_net.update_input_channels(current_channels)
                    
                    # 清零梯度
                    optimizer.zero_grad()
                    
                    # 使用特徵金字塔
                    if use_feature_pyramid:
                        print(f"\n📊 使用特徵金字塔進行推理，尺度: {pyramid_scales}")
                        
                        # ----- 教師模型特徵金字塔預測 -----
                        all_teacher_scores = []
                        all_teacher_boxes = []
                        all_teacher_extras = []
                        
                        with torch.no_grad():
                            for scale_idx, scale in enumerate(pyramid_scales):
                                scale_start = time.time()
                                print(f"\n🔍 處理教師模型尺度 {scale_idx+1}/{len(pyramid_scales)}: {scale}")
                                
                                # 縮放輸入圖像
                                if scale == 1.0:
                                    scaled_images = images
                                    scaled_class_images = class_images
                                else:
                                    # 縮放圖像
                                    if images.dim() == 4:  # [B, C, H, W]
                                        h, w = images.shape[2:]
                                        new_h, new_w = int(h * scale), int(w * scale)
                                        print(f"   縮放圖像: {h}x{w} → {new_h}x{new_w}")
                                        scaled_images = F.interpolate(
                                            images, size=(new_h, new_w), mode='bilinear', align_corners=False
                                        )
                                    else:
                                        scaled_images = images
                                        print(f"   跳過圖像縮放，維度不是4D: {images.dim()}")
                                    
                                    # 縮放類別圖像
                                    if isinstance(class_images, list) and class_images:
                                        scaled_class_images = []
                                        for idx, img in enumerate(class_images):
                                            if isinstance(img, torch.Tensor) and img.dim() >= 3:  # [C, H, W] 或 [B, C, H, W]
                                                h, w = img.shape[-2:]
                                                new_h, new_w = int(h * scale), int(w * scale)
                                                print(f"   縮放類別圖像 {idx+1}: {h}x{w} → {new_h}x{new_w}")
                                                scaled_img = F.interpolate(
                                                    img.unsqueeze(0) if img.dim() == 3 else img, 
                                                    size=(new_h, new_w), 
                                                    mode='bilinear', 
                                                    align_corners=False
                                                )
                                                scaled_img = scaled_img.squeeze(0) if img.dim() == 3 else scaled_img
                                                scaled_class_images.append(scaled_img)
                                            else:
                                                scaled_class_images.append(img)
                                                print(f"   跳過類別圖像 {idx+1} 縮放，不是有效張量")
                                    else:
                                        scaled_class_images = class_images
                                        print(f"   跳過類別圖像縮放，不是張量列表")
                                
                                # 教師模型預測
                                print(f"   運行教師模型推理中...")
                                teacher_outputs = self.teacher_model(scaled_images, class_images=scaled_class_images)
                                
                                # 應用 NMS 減少框數量
                                if apply_nms:
                                    teacher_outputs = apply_nms_to_outputs(
                                        teacher_outputs, 
                                        max_boxes=max_predictions, 
                                        iou_threshold=nms_threshold
                                    )
                                
                                if isinstance(teacher_outputs, tuple) and len(teacher_outputs) >= 2:
                                    teacher_scores, teacher_boxes = teacher_outputs[0], teacher_outputs[1]
                                    
                                    # 如果沒應用NMS，限制預測數量
                                    if not apply_nms and teacher_boxes.shape[0] > max_predictions:
                                        keep_idx = torch.randperm(teacher_boxes.shape[0])[:max_predictions]
                                        teacher_scores = teacher_scores[keep_idx]
                                        teacher_boxes = teacher_boxes[keep_idx]
                                        if len(teacher_outputs) > 2:
                                            teacher_extras = tuple(extra[keep_idx] if isinstance(extra, torch.Tensor) else extra 
                                                                for extra in teacher_outputs[2:])
                                        else:
                                            teacher_extras = teacher_outputs[2:] if len(teacher_outputs) > 2 else ()
                                    else:
                                        teacher_extras = teacher_outputs[2:] if len(teacher_outputs) > 2 else ()
                                    
                                    # 收集所有尺度的結果
                                    all_teacher_scores.append(teacher_scores)
                                    all_teacher_boxes.append(teacher_boxes)
                                    if teacher_extras:
                                        all_teacher_extras.append(teacher_extras)
                                    
                                    # 分析教師模型輸出
                                    output_info = self.analyze_os2d_outputs((teacher_scores, teacher_boxes))
                                    if output_info.get('dense_format', False):
                                        print(f"   教師模型尺度 {scale}: 密集格式 {teacher_scores.shape}, 位置數量: {output_info.get('num_positions', 'N/A')}")
                                    else:
                                        print(f"   教師模型尺度 {scale}: 檢測框數量: {teacher_boxes.shape[0]}, 分數形狀: {teacher_scores.shape}")
                                
                                scale_time = time.time() - scale_start
                                print(f"   尺度 {scale} 處理耗時: {scale_time:.2f}秒")
                        
                        # 合併教師模型結果
                        try:
                            if all_teacher_scores and all_teacher_boxes:
                                merge_start = time.time()
                                print(f"\n🔄 合併教師模型特徵金字塔結果...")
                                
                                # 檢查形狀一致性
                                all_shapes = [s.shape for s in all_teacher_scores]
                                print(f"   分數張量形狀: {all_shapes}")
                                
                                # 根據輸出形狀決定合併方式
                                dense_format = all_teacher_scores[0].dim() == 4
                                
                                if dense_format:
                                    # 密集格式 [B, C, 4, N]，在最後一個維度 (特徵位置) 合併
                                    print(f"💡 檢測到密集格式輸出，在特徵位置維度 (dim=3) 合併")
                                    try:
                                        teacher_scores = torch.cat(all_teacher_scores, dim=3)
                                        teacher_boxes = torch.cat(all_teacher_boxes, dim=2) # 通常 boxes 是 [B, C, N] 格式
                                        print(f"✓ 合併成功: scores {teacher_scores.shape}, boxes {teacher_boxes.shape}")
                                        
                                        # 正確合併額外輸出
                                        if all_teacher_extras:
                                            teacher_extras = []
                                            for i in range(len(all_teacher_extras[0])):
                                                extras = []
                                                for scale_extras in all_teacher_extras:
                                                    if i < len(scale_extras) and scale_extras[i] is not None:
                                                        extras.append(scale_extras[i])
                                                        
                                                if extras and all(isinstance(e, torch.Tensor) for e in extras):
                                                    # 確定合併維度
                                                    if extras[0].dim() == teacher_scores.dim():
                                                        teacher_extras.append(torch.cat(extras, dim=3)) # 與 scores 相同維度
                                                    elif extras[0].dim() == teacher_boxes.dim():
                                                        teacher_extras.append(torch.cat(extras, dim=2)) # 與 boxes 相同維度
                                                    else:
                                                        teacher_extras.append(extras[0]) # 無法確定，使用第一個
                                                else:
                                                    teacher_extras.append(None)
                                                    
                                            teacher_outputs = (teacher_scores, teacher_boxes) + tuple(e for e in teacher_extras if e is not None)
                                        else:
                                            teacher_outputs = (teacher_scores, teacher_boxes)
                                    except RuntimeError as e:
                                        print(f"❌ 密集格式合併失敗: {e}")
                                        # 使用第一個尺度的結果作為備選
                                        teacher_outputs = (all_teacher_scores[0], all_teacher_boxes[0])
                                        print(f"⚠️ 使用尺度 {pyramid_scales[0]} 的結果作為備選")
                                else:
                                    # 標準格式 [N, C] 和 [N, 4]，在第0維度 (樣本數) 合併
                                    print(f"💡 檢測到標準格式輸出，在樣本維度 (dim=0) 合併")
                                    try:
                                        teacher_scores = torch.cat(all_teacher_scores, dim=0)
                                        teacher_boxes = torch.cat(all_teacher_boxes, dim=0)
                                        print(f"✓ 合併成功: scores {teacher_scores.shape}, boxes {teacher_boxes.shape}")
                                        
                                        # 合併額外輸出
                                        if all_teacher_extras:
                                            teacher_extras = []
                                            for i in range(len(all_teacher_extras[0])):
                                                extras = [scale_extras[i] for scale_extras in all_teacher_extras 
                                                        if i < len(scale_extras) and scale_extras[i] is not None 
                                                        and isinstance(scale_extras[i], torch.Tensor)]
                                                if extras:
                                                    try:
                                                        teacher_extras.append(torch.cat(extras, dim=0))
                                                    except RuntimeError:
                                                        teacher_extras.append(extras[0])
                                                else:
                                                    first_valid = next((x for scale_extras in all_teacher_extras 
                                                                    if i < len(scale_extras) 
                                                                    for x in [scale_extras[i]] if x is not None), None)
                                                    teacher_extras.append(first_valid)
                                                    
                                            teacher_outputs = (teacher_scores, teacher_boxes) + tuple(e for e in teacher_extras if e is not None)
                                        else:
                                            teacher_outputs = (teacher_scores, teacher_boxes)
                                    except RuntimeError as e:
                                        print(f"❌ 標準格式合併失敗: {e}")
                                        # 使用第一個尺度的結果作為備選
                                        teacher_outputs = (all_teacher_scores[0], all_teacher_boxes[0])
                                        print(f"⚠️ 使用尺度 {pyramid_scales[0]} 的結果作為備選")
                                
                                # 應用 NMS
                                if apply_nms:
                                    teacher_outputs = apply_nms_to_outputs(
                                        teacher_outputs, 
                                        max_boxes=max_predictions, 
                                        iou_threshold=nms_threshold
                                    )
                                    
                                merge_time = time.time() - merge_start
                                print(f"✓ 教師模型特徵金字塔合併完成，耗時: {merge_time:.2f}秒")
                                
                            else:
                                print("⚠️ 教師模型未產生有效輸出，使用原始模式")
                                with torch.no_grad():
                                    teacher_outputs = self.teacher_model(images, class_images=class_images)
                                    if apply_nms:
                                        teacher_outputs = apply_nms_to_outputs(
                                            teacher_outputs, 
                                            max_boxes=max_predictions, 
                                            iou_threshold=nms_threshold
                                        )
                        except Exception as e:
                            print(f"❌ 合併教師模型結果失敗: {e}")
                            print(traceback.format_exc())
                            # 使用原始模式作為備選方案
                            with torch.no_grad():
                                teacher_outputs = self.teacher_model(images, class_images=class_images)
                                if apply_nms:
                                    teacher_outputs = apply_nms_to_outputs(
                                        teacher_outputs, 
                                        max_boxes=max_predictions, 
                                        iou_threshold=nms_threshold
                                    )
                        # ----- 學生模型特徵金字塔預測 -----
                        all_student_scores = []
                        all_student_boxes = []
                        all_student_extras = []
                        
                        for scale_idx, scale in enumerate(pyramid_scales):
                            scale_start = time.time()
                            print(f"\n🔍 處理學生模型尺度 {scale_idx+1}/{len(pyramid_scales)}: {scale}")
                            
                            # 縮放輸入圖像
                            if scale == 1.0:
                                scaled_images = images
                                scaled_class_images = class_images
                            else:
                                # 縮放圖像
                                if images.dim() == 4:  # [B, C, H, W]
                                    h, w = images.shape[2:]
                                    new_h, new_w = int(h * scale), int(w * scale)
                                    print(f"   縮放圖像: {h}x{w} → {new_h}x{new_w}")
                                    scaled_images = F.interpolate(
                                        images, size=(new_h, new_w), mode='bilinear', align_corners=False
                                    )
                                else:
                                    scaled_images = images
                                    print(f"   跳過圖像縮放，維度不是4D: {images.dim()}")
                                
                                # 縮放類別圖像
                                if isinstance(class_images, list) and class_images:
                                    scaled_class_images = []
                                    for idx, img in enumerate(class_images):
                                        if isinstance(img, torch.Tensor) and img.dim() >= 3:  # [C, H, W] 或 [B, C, H, W]
                                            h, w = img.shape[-2:]
                                            new_h, new_w = int(h * scale), int(w * scale)
                                            print(f"   縮放類別圖像 {idx+1}: {h}x{w} → {new_h}x{new_w}")
                                            scaled_img = F.interpolate(
                                                img.unsqueeze(0) if img.dim() == 3 else img, 
                                                size=(new_h, new_w), 
                                                mode='bilinear', 
                                                align_corners=False
                                            )
                                            scaled_img = scaled_img.squeeze(0) if img.dim() == 3 else scaled_img
                                            scaled_class_images.append(scaled_img)
                                        else:
                                            scaled_class_images.append(img)
                                            print(f"   跳過類別圖像 {idx+1} 縮放，不是有效張量")
                                else:
                                    scaled_class_images = class_images
                                    print(f"   跳過類別圖像縮放，不是張量列表")
                            
                            # 學生模型預測
                            print(f"   運行學生模型推理中...")
                            outputs = self(scaled_images, class_images=scaled_class_images)
                            
                            # 應用 NMS 減少框數量
                            if apply_nms:
                                outputs = apply_nms_to_outputs(
                                    outputs, 
                                    max_boxes=max_predictions, 
                                    iou_threshold=nms_threshold
                                )
                            
                            if isinstance(outputs, tuple) and len(outputs) >= 2:
                                student_scores, student_boxes = outputs[0], outputs[1]
                                
                                # 如果沒應用 NMS，限制預測數量
                                if not apply_nms and student_boxes.shape[0] > max_predictions:
                                    keep_idx = torch.randperm(student_boxes.shape[0])[:max_predictions]
                                    student_scores = student_scores[keep_idx]
                                    student_boxes = student_boxes[keep_idx]
                                    if len(outputs) > 2:
                                        student_extras = tuple(extra[keep_idx] if isinstance(extra, torch.Tensor) else extra 
                                                            for extra in outputs[2:])
                                    else:
                                        student_extras = outputs[2:] if len(outputs) > 2 else ()
                                else:
                                    student_extras = outputs[2:] if len(outputs) > 2 else ()
                                
                                # 收集所有尺度的結果
                                all_student_scores.append(student_scores)
                                all_student_boxes.append(student_boxes)
                                if student_extras:
                                    all_student_extras.append(student_extras)
                                
                                # 分析學生模型輸出
                                output_info = self.analyze_os2d_outputs((student_scores, student_boxes))
                                if output_info.get('dense_format', False):
                                    print(f"   學生模型尺度 {scale}: 密集格式 {student_scores.shape}, 位置數量: {output_info.get('num_positions', 'N/A')}")
                                else:
                                    print(f"   學生模型尺度 {scale}: 檢測框數量: {student_boxes.shape[0]}, 分數形狀: {student_scores.shape}")
                            
                            scale_time = time.time() - scale_start
                            print(f"   尺度 {scale} 處理耗時: {scale_time:.2f}秒")
                        
                        # 合併學生模型結果
                        try:
                            if all_student_scores and all_student_boxes:
                                merge_start = time.time()
                                print(f"\n🔄 合併學生模型特徵金字塔結果...")
                                
                                # 檢查形狀一致性
                                all_shapes = [s.shape for s in all_student_scores]
                                print(f"   分數張量形狀: {all_shapes}")
                                
                                # 根據輸出形狀決定合併方式
                                dense_format = all_student_scores[0].dim() == 4
                                
                                if dense_format:
                                    # 密集格式 [B, C, 4, N]，在最後一個維度 (特徵位置) 合併
                                    print(f"💡 檢測到密集格式輸出，在特徵位置維度 (dim=3) 合併")
                                    try:
                                        student_scores = torch.cat(all_student_scores, dim=3)
                                        student_boxes = torch.cat(all_student_boxes, dim=2) # 通常 boxes 是 [B, C, N] 格式
                                        print(f"✓ 合併成功: scores {student_scores.shape}, boxes {student_boxes.shape}")
                                        
                                        # 正確合併額外輸出
                                        if all_student_extras:
                                            student_extras = []
                                            for i in range(len(all_student_extras[0])):
                                                extras = []
                                                for scale_extras in all_student_extras:
                                                    if i < len(scale_extras) and scale_extras[i] is not None:
                                                        extras.append(scale_extras[i])
                                                        
                                                if extras and all(isinstance(e, torch.Tensor) for e in extras):
                                                    # 確定合併維度
                                                    if extras[0].dim() == student_scores.dim():
                                                        student_extras.append(torch.cat(extras, dim=3)) # 與 scores 相同維度
                                                    elif extras[0].dim() == student_boxes.dim():
                                                        student_extras.append(torch.cat(extras, dim=2)) # 與 boxes 相同維度
                                                    else:
                                                        student_extras.append(extras[0]) # 無法確定，使用第一個
                                                else:
                                                    student_extras.append(None)
                                                    
                                            outputs = (student_scores, student_boxes) + tuple(e for e in student_extras if e is not None)
                                        else:
                                            outputs = (student_scores, student_boxes)
                                    except RuntimeError as e:
                                        print(f"❌ 密集格式合併失敗: {e}")
                                        # 使用第一個尺度的結果作為備選
                                        outputs = (all_student_scores[0], all_student_boxes[0])
                                        print(f"⚠️ 使用尺度 {pyramid_scales[0]} 的結果作為備選")
                                else:
                                    # 標準格式 [N, C] 和 [N, 4]，在第0維度 (樣本數) 合併
                                    print(f"💡 檢測到標準格式輸出，在樣本維度 (dim=0) 合併")
                                    try:
                                        student_scores = torch.cat(all_student_scores, dim=0)
                                        student_boxes = torch.cat(all_student_boxes, dim=0)
                                        print(f"✓ 合併成功: scores {student_scores.shape}, boxes {student_boxes.shape}")
                                        
                                        # 合併額外輸出
                                        if all_student_extras:
                                            student_extras = []
                                            for i in range(len(all_student_extras[0])):
                                                extras = [scale_extras[i] for scale_extras in all_student_extras 
                                                        if i < len(scale_extras) and scale_extras[i] is not None 
                                                        and isinstance(scale_extras[i], torch.Tensor)]
                                                if extras:
                                                    try:
                                                        student_extras.append(torch.cat(extras, dim=0))
                                                    except RuntimeError:
                                                        student_extras.append(extras[0])
                                                else:
                                                    first_valid = next((x for scale_extras in all_student_extras 
                                                                    if i < len(scale_extras) 
                                                                    for x in [scale_extras[i]] if x is not None), None)
                                                    student_extras.append(first_valid)
                                                    
                                            outputs = (student_scores, student_boxes) + tuple(e for e in student_extras if e is not None)
                                        else:
                                            outputs = (student_scores, student_boxes)
                                    except RuntimeError as e:
                                        print(f"❌ 標準格式合併失敗: {e}")
                                        # 使用第一個尺度的結果作為備選
                                        outputs = (all_student_scores[0], all_student_boxes[0])
                                        print(f"⚠️ 使用尺度 {pyramid_scales[0]} 的結果作為備選")
                                
                                # 應用 NMS
                                if apply_nms:
                                    outputs = apply_nms_to_outputs(
                                        outputs, 
                                        max_boxes=max_predictions, 
                                        iou_threshold=nms_threshold
                                    )
                                    
                                merge_time = time.time() - merge_start
                                print(f"✓ 學生模型特徵金字塔合併完成，耗時: {merge_time:.2f}秒")
                                
                            else:
                                print("⚠️ 學生模型未產生有效輸出，使用原始模式")
                                outputs = self(images, class_images=class_images)
                                if apply_nms:
                                    outputs = apply_nms_to_outputs(
                                        outputs, 
                                        max_boxes=max_predictions, 
                                        iou_threshold=nms_threshold
                                    )
                        except Exception as e:
                            print(f"❌ 合併學生模型結果失敗: {e}")
                            print(traceback.format_exc())
                            # 使用原始模式作為備選方案
                            outputs = self(images, class_images=class_images)
                            if apply_nms:
                                outputs = apply_nms_to_outputs(
                                    outputs, 
                                    max_boxes=max_predictions, 
                                    iou_threshold=nms_threshold
                                )
                    
                    else:
                        # 不使用特徵金字塔的標準處理方式
                        print(f"\n⚡ 使用標準模式 (無特徵金字塔)")
                        
                        # 教師模型預測（知識蒸餾）
                        with torch.no_grad():
                            teacher_outputs = self.teacher_model(images, class_images=class_images)
                            if apply_nms:
                                teacher_outputs = apply_nms_to_outputs(
                                    teacher_outputs, 
                                    max_boxes=max_predictions, 
                                    iou_threshold=nms_threshold
                                )
                        
                        # 學生模型預測
                        outputs = self(images, class_images=class_images)
                        if apply_nms:
                            outputs = apply_nms_to_outputs(
                                outputs, 
                                max_boxes=max_predictions, 
                                iou_threshold=nms_threshold
                            )
                    
                    # 分析和顯示輸出信息
                    print("\n📊 模型輸出分析:")
                    
                    if isinstance(outputs, tuple) and len(outputs) >= 2:
                        student_boxes = outputs[1]
                        student_scores = outputs[0]
                        
                        # 輸出張量形狀信息
                        student_shape_info = f"張量形狀: scores {student_scores.shape}, boxes {student_boxes.shape}"
                        
                        # 分析輸出維度信息
                        if student_scores.dim() == 2:
                            print(f"✓ 學生模型輸出: {len(outputs)} 元素，框數量: {student_boxes.shape[0]}")
                            print(f"   - 每框類別數: {student_scores.shape[1]}")
                            print(f"   - {student_shape_info}")
                        elif student_scores.dim() == 4:
                            print(f"✓ 學生模型輸出: {len(outputs)} 元素，批次大小: {student_scores.shape[0]}")
                            print(f"   - 類別數: {student_scores.shape[1]}")
                            print(f"   - 特徵位置數: {student_scores.shape[3]}")
                            print(f"   - {student_shape_info}")
                        else:
                            print(f"✓ 學生模型輸出: {len(outputs)} 元素，形狀: {student_shape_info}")

                    if isinstance(teacher_outputs, tuple) and len(teacher_outputs) >= 2:
                        teacher_boxes = teacher_outputs[1]
                        teacher_scores = teacher_outputs[0]
                        
                        # 輸出張量形狀信息
                        teacher_shape_info = f"張量形狀: scores {teacher_scores.shape}, boxes {teacher_boxes.shape}"
                        
                        # 分析輸出維度信息
                        if teacher_scores.dim() == 2:
                            print(f"✓ 教師模型輸出: {len(teacher_outputs)} 元素，框數量: {teacher_boxes.shape[0]}")
                            print(f"   - 每框類別數: {teacher_scores.shape[1]}")
                            print(f"   - {teacher_shape_info}")
                        elif teacher_scores.dim() == 4:
                            print(f"✓ 教師模型輸出: {len(teacher_outputs)} 元素，批次大小: {teacher_scores.shape[0]}")
                            print(f"   - 類別數: {teacher_scores.shape[1]}")
                            print(f"   - 特徵位置數: {teacher_scores.shape[3]}")
                            print(f"   - {teacher_shape_info}")
                        else:
                            print(f"✓ 教師模型輸出: {len(teacher_outputs)} 元素，形狀: {teacher_shape_info}")
                    
                    # 構建 targets 字典
                    targets = {
                        'class_ids': batch_class_ids,
                        'boxes': batch_boxes,
                        'images': images,
                        'class_targets': class_targets,
                        'loc_targets': loc_targets,
                        'teacher_outputs': teacher_outputs
                    }
                    
                    # 處理 outputs 結構
                    if not isinstance(outputs, dict):
                        if isinstance(outputs, tuple) and len(outputs) >= 2:
                            outputs_dict = {
                                'class_scores': outputs[0],
                                'boxes': outputs[1],
                                'images': images,
                                'class_images': class_images,
                                'feature_pyramid': use_feature_pyramid,  # 添加標記以便損失計算時知道使用了特徵金字塔
                                'pyramid_scales': pyramid_scales  # 添加使用的尺度信息
                            }
                        else:
                            outputs_dict = {
                                'class_scores': outputs,
                                'images': images,
                                'class_images': class_images
                            }
                    else:
                        outputs_dict = outputs
                        outputs_dict['feature_pyramid'] = use_feature_pyramid
                        outputs_dict['pyramid_scales'] = pyramid_scales
                    
                    # 計算損失
                    print(f"\n📊 開始計算損失...")
                    loss_start = time.time()
                    
                    loss, loss_dict = self.compute_losses(
                        outputs_dict, 
                        targets,
                        class_num=class_num,
                        auxiliary_net=auxiliary_net,
                        use_lcp_loss=use_lcp_loss,
                        loss_weights=loss_weights
                    )
                    
                    loss_time = time.time() - loss_start
                    print(f"✓ 損失計算完成，耗時: {loss_time:.2f}秒")
                    print(f"   - 分類損失: {loss_dict['cls_loss'].item():.4f}")
                    print(f"   - 框回歸損失: {loss_dict['box_loss'].item():.4f}")
                    print(f"   - 教師損失: {loss_dict['teacher_loss'].item():.4f}")
                    print(f"   - LCP損失: {loss_dict['lcp_loss'].item():.4f}")
                    
                    # 反向傳播
                    backprop_start = time.time()
                    print(f"\n🔄 開始反向傳播...")
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
                    if auxiliary_net is not None:
                        torch.nn.utils.clip_grad_norm_(auxiliary_net.parameters(), max_norm=10.0)
                    
                    # 優化器步進
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    
                    backprop_time = time.time() - backprop_start
                    print(f"✓ 反向傳播完成，耗時: {backprop_time:.2f}秒")
                    
                    # 記錄損失
                    loss_value = loss.item()
                    loss_history.append(loss_value)
                    
                    # 更新總損失統計
                    total_loss += loss_value
                    total_cls_loss += loss_dict['cls_loss'].item()
                    total_box_loss += loss_dict['box_loss'].item()
                    total_teacher_loss += loss_dict['teacher_loss'].item()
                    total_lcp_loss += loss_dict['lcp_loss'].item()
                    batch_count += 1
                    
                    # 更新進度條描述
                    pbar.set_description(
                        f"Loss: {loss_value:.4f} (cls: {loss_dict['cls_loss'].item():.4f}, "
                        f"box: {loss_dict['box_loss'].item():.4f})"
                    )
                    
                    # 打印詳細信息
                    if print_freq > 0 and (batch_idx % print_freq == 0 or batch_idx == num_batches - 1):
                        print(f"\n批次 {batch_idx+1}/{num_batches} 摘要:")
                        print(f"  分類損失: {loss_dict['cls_loss'].item():.4f}")
                        print(f"  框回歸損失: {loss_dict['box_loss'].item():.4f}")
                        print(f"  教師損失: {loss_dict['teacher_loss'].item():.4f}")
                        print(f"  LCP損失: {loss_dict['lcp_loss'].item():.4f}")
                        print(f"  總損失: {loss_value:.4f}")
                    
                    # 計算批次總耗時
                    batch_time = time.time() - batch_start_time
                    print(f"\n✓ 批次 {batch_idx+1}/{num_batches} 完成，總耗時: {batch_time:.2f}秒\n")
                    print("-" * 80)
                
                except Exception as e:
                    print(f"❌ 批次 {batch_idx+1} 處理出錯: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # 計算平均損失
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_cls_loss = total_cls_loss / batch_count if batch_count > 0 else 0
        avg_box_loss = total_box_loss / batch_count if batch_count > 0 else 0
        avg_teacher_loss = total_teacher_loss / batch_count if batch_count > 0 else 0
        avg_lcp_loss = total_lcp_loss / batch_count if batch_count > 0 else 0
        
        # 輸出訓練結果摘要
        import datetime
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 50)
        print(f"✅ 訓練完成! 處理了 {batch_count}/{num_batches} 批次")
        print(f"✅ 總耗時: {elapsed_time:.2f}秒 ({datetime.timedelta(seconds=int(elapsed_time))})")
        print(f"✅ 特徵金字塔: {'啟用' if use_feature_pyramid else '停用'}")
        if use_feature_pyramid:
            print(f"✅ 特徵金字塔尺度: {pyramid_scales}")
        print(f"✅ NMS: {'啟用' if apply_nms else '停用'}, 閾值: {nms_threshold}")
        print(f"✅ 平均損失: {avg_loss:.4f}")
        print(f"   - 平均分類損失: {avg_cls_loss:.4f}")
        print(f"   - 平均框回歸損失: {avg_box_loss:.4f}")
        print(f"   - 平均教師損失: {avg_teacher_loss:.4f}")
        print(f"   - 平均LCP損失: {avg_lcp_loss:.4f}")
        print("=" * 50)
        
        return loss_history
    def finetune(self, train_loader, auxiliary_net, prune_layers, prune_ratio=0.3,
                optimizer=None, device=None, epochs_per_layer=1, print_freq=1, max_batches=3):
        pass

    
    def load_checkpoint(self, checkpoint_path, device=None):
        pass

    
    def save_checkpoint(self, checkpoint_path):
        pass

    def _eval(self, dataloader, iou_thresh=0.5, batch_size=4, cfg=None, criterion=None, print_per_class_results=False):
        pass