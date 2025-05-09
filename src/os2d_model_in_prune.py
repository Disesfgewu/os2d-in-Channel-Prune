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
                 pretrained_path='./models/os2d_v1-train.pth', pruned_checkpoint=None, **kwargs):
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
        """計算訓練損失的組合（分類、框回歸、教師蒸餾和 LCP）"""
        # 預設損失權重
        if loss_weights is None:
            loss_weights = {'cls': 1.0, 'box_reg': 1.0, 'teacher': 0.5, 'lcp': 0.1}
        
        # 初始化損失 - 使用 requires_grad=True 以確保正確的梯度計算
        cls_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        box_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        teacher_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        lcp_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 先處理框數量不匹配問題
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            pred_boxes = outputs[1]
            target_boxes = self._prepare_target_boxes(targets['boxes'], pred_boxes.device)
            
            # 處理 OS2D 密集格式
            if pred_boxes.dim() > 2:
                pred_boxes = self._convert_dense_boxes_to_standard(pred_boxes, outputs[0])
            
            # 處理框數量不匹配情況
            if pred_boxes.size(0) != target_boxes.size(0):
                pred_boxes, target_boxes = self._match_box_counts(pred_boxes, target_boxes, outputs[0])
                
                # 將選好的框保存到 targets 中，供其他損失函數使用
                targets['selected_boxes'] = pred_boxes
        
        # 1. 分類損失
        try:
            cls_loss = self._compute_classification_loss(outputs, targets['class_ids'])
            print(f"   ✓ 分類損失計算完成: {cls_loss.item():.4f}")
        except Exception as e:
            print(f"   ❌ 分類損失計算失敗: {e}")
        
        # 2. 框回歸損失
        try:
            box_loss = self._compute_box_regression_loss(outputs, targets['boxes'])
            print(f"   ✓ 框回歸損失計算完成: {box_loss.item():.4f}")
        except Exception as e:
            print(f"   ❌ 框回歸損失計算失敗: {e}")
        
        # 3. 教師知識蒸餾損失
        try:
            if 'teacher_outputs' in targets and targets['teacher_outputs'] is not None:
                teacher_loss = self._compute_teacher_distillation_loss(outputs, targets)
                print(f"   ✓ 教師損失計算完成: {teacher_loss.item():.4f}")
            else:
                print(f"⚠️ 教師輸出無效，跳過教師損失計算")
        except Exception as e:
            print(f"   ❌ 教師損失計算失敗: {e}")
            import traceback
            print(traceback.format_exc())
        
        # 4. LCP 損失
        if use_lcp_loss and auxiliary_net is not None:
            try:
                lcp_loss = self._compute_lcp_loss(outputs, targets, auxiliary_net)
                print(f"   ✓ 最終 LCP 損失: {lcp_loss.item():.4f}")
            except Exception as e:
                print(f"   ❌ LCP 損失計算失敗: {e}")
                import traceback
                print(traceback.format_exc())
        
        # 計算總損失前進行損失平衡
        scaled_losses = self._scale_losses(cls_loss, box_loss, teacher_loss, lcp_loss)
        scaled_cls_loss, scaled_box_loss, scaled_teacher_loss, scaled_lcp_loss = scaled_losses
        
        # 計算總損失 - 使用新的張量而非就地操作
        total_loss = (
            loss_weights['cls'] * scaled_cls_loss +
            loss_weights['box_reg'] * scaled_box_loss +
            loss_weights['teacher'] * scaled_teacher_loss +
            loss_weights['lcp'] * scaled_lcp_loss
        )
        
        # 構建損失字典 - 保存原始損失值
        loss_dict = {
            'cls_loss': cls_loss.clone(),  # 使用 clone 避免就地操作
            'box_loss': box_loss.clone(),
            'teacher_loss': teacher_loss.clone(),
            'lcp_loss': lcp_loss.clone()
        }
        
        # 保存到模型中供外部訪問
        self.last_losses = {k: v.item() for k, v in loss_dict.items()}
        
        return total_loss, loss_dict

    def _compute_classification_loss(self, outputs, class_ids):
        """計算分類損失"""
        # 從輸出中獲取分類分數
        if isinstance(outputs, dict) and 'class_scores' in outputs:
            class_scores = outputs['class_scores']
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            # 處理來自輔助網路的輸出元組，通常第一個元素是分類分數
            class_scores = outputs[0]
            print(f"✓ 從輸出元組中提取分類分數: {type(class_scores)}")
        else:
            class_scores = outputs
        
        # 確保 class_scores 是張量
        if not isinstance(class_scores, torch.Tensor):
            print(f"⚠️ class_scores 不是張量，而是 {type(class_scores)}，創建預設張量")
            # 創建一個預設的張量，防止錯誤
            class_scores = torch.zeros((1, max(1, len(class_ids) if hasattr(class_ids, '__len__') else 1)),
                                    device=self.device)
        
        # 處理 OS2D 特殊的 4D 輸出格式
        if class_scores.dim() > 2:
            class_scores = self._convert_4d_to_2d_scores(class_scores)
        
        # 處理目標類別
        target = self._prepare_classification_targets(class_ids, class_scores.size(1))
        
        # 使用交叉熵損失
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(class_scores, target)

    def _convert_4d_to_2d_scores(self, class_scores):
        """將 [B, C, H, W] 格式的分數轉換為 [B, C] 格式"""
        # 獲取批次大小和類別數
        batch_size = class_scores.size(0)
        num_classes = class_scores.size(1)
        
        # 將 [B, C, H, W] 轉換為 [B, C] 通過平均池化或最大池化
        class_scores = class_scores.view(batch_size, num_classes, -1)
        class_scores = class_scores.mean(dim=2)  # 或使用 max(dim=2)[0]
        
        print(f"✓ 將 class_scores 從 4D 轉換為 2D: {class_scores.shape}")
        return class_scores

    def _prepare_classification_targets(self, class_ids, num_classes):
        """準備分類目標"""
        # 只使用每個圖像的主要類別
        if isinstance(class_ids, list):
            main_class_ids = []
            for cls_id in class_ids:
                if isinstance(cls_id, torch.Tensor) and cls_id.numel() > 0:
                    main_class_ids.append(cls_id[0].unsqueeze(0))
            
            if main_class_ids:
                target = torch.cat(main_class_ids).long()
            else:
                target = torch.zeros(1, dtype=torch.long, device=self.device)
        else:
            target = class_ids.long()
        
        # 確保目標在有效範圍內
        if target.max() >= num_classes:
            target = torch.clamp(target, max=num_classes-1)
        
        return target.to(self.device)

    def _compute_box_regression_loss(self, outputs, boxes):
        """計算邊界框回歸損失"""
        # 從輸出中獲取預測框
        if isinstance(outputs, dict) and 'boxes' in outputs:
            pred_boxes = outputs['boxes']
            class_scores = outputs.get('class_scores', None)
        elif isinstance(outputs, tuple):
            pred_boxes = outputs[1] if len(outputs) > 1 else torch.zeros(1, 4, device=self.device)
            class_scores = outputs[0] if len(outputs) > 0 else None
        else:
            pred_boxes = outputs
            class_scores = None
        
        # 確保 pred_boxes 是張量
        if not isinstance(pred_boxes, torch.Tensor):
            print(f"警告: pred_boxes 不是張量，而是 {type(pred_boxes)}，使用預設值")
            pred_boxes = torch.zeros(1, 4, device=self.device)
        
        # 處理目標框
        target_boxes = self._prepare_target_boxes(boxes, pred_boxes.device)
        
        # 打印 debug 信息
        print(f"📊 Box 回歸內容檢查:")
        print(f" pred_boxes: 形狀 {pred_boxes.shape}, 類型 {pred_boxes.dtype}, 裝置 {pred_boxes.device}")
        print(f" target_boxes: 形狀 {target_boxes.shape}, 類型 {target_boxes.dtype}, 裝置 {target_boxes.device}")
        
        # 如果沒有有效的目標框，返回零損失
        if target_boxes.numel() == 0:
            print(f"⚠️ 目標框為空，返回零損失")
            return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
        
        # 處理 OS2D 密集格式
        if pred_boxes.dim() > 2:
            pred_boxes = self._convert_dense_boxes_to_standard(pred_boxes, class_scores)
        
        # 處理框數量不匹配情況
        pred_boxes, target_boxes = self._match_box_counts(pred_boxes, target_boxes, class_scores)
        
        # 計算損失並檢查是否需要縮放
        try:
            raw_loss = F.smooth_l1_loss(pred_boxes, target_boxes, reduction='none')
            per_box_loss = raw_loss.sum(dim=1)
            
            # 檢測異常值
            if per_box_loss.mean() > 10.0:
                scale_factor = min(1.0, 10.0 / per_box_loss.mean().item())
                loss = raw_loss.mean() * scale_factor
                print(f"✓ 損失縮放 (因子={scale_factor:.6f}), 縮放後損失: {loss.item():.4f}")
            else:
                loss = raw_loss.mean()
                print(f"✓ 計算框回歸損失: {loss.item():.4f}")
            
            return loss
        except Exception as e:
            print(f"⚠️ 計算損失時出錯: {e}")
            return torch.tensor(1.0, device=pred_boxes.device, requires_grad=True)

    def _prepare_target_boxes(self, boxes, device):
        """準備目標框"""
        if isinstance(boxes, list):
            target_boxes = self._cat_boxes_list(boxes, device=device)
        else:
            if hasattr(boxes, 'bbox_xyxy'):
                target_boxes = boxes.bbox_xyxy
            elif hasattr(boxes, 'bbox') and hasattr(boxes, 'size'):
                target_boxes = boxes.bbox
            else:
                target_boxes = boxes
        
        # 確保張量在同一設備上
        if isinstance(target_boxes, torch.Tensor):
            target_boxes = target_boxes.to(device)
        else:
            print(f"警告: target_boxes 不是張量，使用預設值")
            target_boxes = torch.zeros(1, 4, device=device)
        
        return target_boxes

    def _convert_dense_boxes_to_standard(self, pred_boxes, class_scores):
        """將密集格式的框轉換為標準格式"""
        try:
            # 處理密集格式 [B, C, N] 或 [B, C, 4, N]
            if pred_boxes.dim() == 4:  # [B, C, 4, N]
                b, c, four, n = pred_boxes.shape
                if four == 4:
                    # 如果是 [B, C, 4, N]，轉換為 [B*C*N, 4]
                    pred_boxes = pred_boxes.permute(0, 1, 3, 2).reshape(-1, 4)
                else:
                    pred_boxes = pred_boxes.reshape(-1, 4)
            elif pred_boxes.dim() == 3:  # [B, C, N]
                b, c, n = pred_boxes.shape
                
                # 特別處理 OS2D 特定的 [1, 9, 56661] 形狀
                if pred_boxes.shape[2] % 4 != 0:
                    print(f"⚠️ 偵測到 OS2D 特殊格式: {pred_boxes.shape}")
                    
                    # 獲取分數信息 - 用於選擇最佳框
                    confidence_scores = None
                    if class_scores is not None:
                        if class_scores.dim() == 4:  # [B, C, 4, N]
                            # 提取每個位置的最高分數
                            confidence_scores = class_scores.max(dim=2)[0]  # [B, C, N]
                            confidence_scores = confidence_scores.view(-1)  # [B*C*N]
                    
                    # 將數據重整為可用的格式
                    # 如果有四個坐標，每個框需要4個值
                    pred_boxes_flat = pred_boxes.reshape(-1)  # [B*C*N]
                    n_boxes = len(pred_boxes_flat) // 4
                    
                    # 確保可以被4整除
                    valid_elements = 4 * n_boxes
                    pred_boxes_flat = pred_boxes_flat[:valid_elements]
                    pred_boxes = pred_boxes_flat.reshape(n_boxes, 4)
                    
                    # 現在 pred_boxes 是 [N, 4] 形狀
                    
                    # 按信心分數排序選擇框
                    if confidence_scores is not None and confidence_scores.numel() >= n_boxes:
                        confidence_scores = confidence_scores[:n_boxes]
                        
                        # 按置信度排序
                        _, indices = torch.sort(confidence_scores, descending=True)
                        pred_boxes = pred_boxes[indices]
                    
                    print(f"✓ 重整後形狀: {pred_boxes.shape}")
                else:
                    # 標準情況，每一列是4個坐標值
                    pred_boxes = pred_boxes.reshape(-1, 4)
            
            return pred_boxes
        except Exception as e:
            print(f"⚠️ 重塑預測框時出錯: {e}")
            return torch.zeros(1, 4, device=pred_boxes.device, requires_grad=True)

    def _match_box_counts(self, pred_boxes, target_boxes, class_scores=None):
        """處理框數量不匹配的情況"""
        # 處理框數量不匹配情況
        if pred_boxes.size(0) != target_boxes.size(0):
            print(f"⚠️ 預測框和目標框數量不匹配: {pred_boxes.size(0)} vs {target_boxes.size(0)}")
            
            if pred_boxes.size(0) > target_boxes.size(0):
                # 獲取目標框數量
                target_size = target_boxes.size(0)
                print(f"🎯 將預測框數量從 {pred_boxes.size(0)} 減少到 {target_size}")
                
                # 選擇框的策略
                pred_boxes_selected = self._select_boxes_by_confidence(
                    pred_boxes, target_size, class_scores)
                
                return pred_boxes_selected, target_boxes
            else:  # pred_boxes.size(0) < target_boxes.size(0)
                # 目標框數量更多，使用部分目標框
                print(f"🔄 預測框數量少於目標框: {pred_boxes.size(0)} < {target_boxes.size(0)}")
                
                # 策略：選擇與預測框相同數量的目標框
                pred_size = pred_boxes.size(0)
                target_indices = torch.randperm(target_boxes.size(0), device=target_boxes.device)[:pred_size]
                target_boxes_sampled = target_boxes[target_indices]
                
                return pred_boxes, target_boxes_sampled
        
        return pred_boxes, target_boxes

    def _select_boxes_by_confidence(self, boxes, target_size, class_scores=None):
        """根據置信度選擇框"""
        # 獲取置信度分數 (如果可用)
        if class_scores is not None:
            try:
                # 根據輸出形狀計算置信度
                confidence_scores = None
                if isinstance(class_scores, torch.Tensor):
                    if class_scores.dim() == 2:  # [N, C] 標準格式
                        # 每個框的最高類別分數
                        confidence_scores, _ = class_scores.max(dim=1)  # [N]
                    elif class_scores.dim() == 4:  # [B, C, 4, N] 密集格式
                        # 轉換為 [B*C*N]
                        confidence_scores = class_scores.max(dim=2)[0]  # [B, C, N]
                        confidence_scores = confidence_scores.view(-1)  # [B*C*N]
                    
                    # 確保長度匹配
                    confidence_scores = confidence_scores[:boxes.size(0)]
                
                if confidence_scores is not None and confidence_scores.numel() == boxes.size(0):
                    # 按置信度排序
                    _, indices = torch.sort(confidence_scores, descending=True)
                    
                    # 選取前 target_size 個框
                    top_indices = indices[:target_size]
                    boxes_selected = boxes[top_indices]
                    print(f"✓ 按信心分數選擇了 {target_size} 個最高得分框")
                    return boxes_selected
                else:
                    # 隨機選擇
                    indices = torch.randperm(boxes.size(0), device=boxes.device)[:target_size]
                    boxes_selected = boxes[indices]
                    print(f"✓ 無有效置信度分數，隨機選擇了 {target_size} 個框")
                    return boxes_selected
            except Exception as e:
                print(f"⚠️ 按信心分數選擇框時出錯: {e}")
        
        # 退回到隨機選擇
        indices = torch.randperm(boxes.size(0), device=boxes.device)[:target_size]
        boxes_selected = boxes[indices]
        print(f"✓ 退回到隨機選擇 {target_size} 個框")
        return boxes_selected

    def _compute_teacher_distillation_loss(self, outputs, targets):
        """計算教師知識蒸餾損失"""
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
            if teacher_scores is not None and student_scores is not None:
                # 處理形狀不一致的情況
                if teacher_scores.dim() != student_scores.dim():
                    teacher_scores, student_scores = self._align_score_dimensions(
                        teacher_scores, student_scores)
                
                # 分類蒸餾損失 (KL 散度)
                cls_distill_loss = self._compute_classification_distillation_loss(
                    teacher_scores, student_scores)
                
                # 框回歸蒸餾損失 (L2 損失)
                # 關鍵修改：使用與框回歸損失相同的框數量
                if 'selected_boxes' in targets and targets['selected_boxes'] is not None:
                    # 使用已經選好的框
                    student_boxes = targets['selected_boxes']
                    # 同樣選擇相同數量的教師框
                    if teacher_boxes.size(0) != student_boxes.size(0):
                        teacher_boxes = self._select_boxes_by_confidence(
                            teacher_boxes, student_boxes.size(0), teacher_scores)
                
                box_distill_loss = self._compute_box_distillation_loss(
                    teacher_boxes, student_boxes, teacher_scores, student_scores)
                
                # 組合蒸餾損失
                teacher_loss = cls_distill_loss + box_distill_loss
                return teacher_loss
        
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _align_score_dimensions(self, teacher_scores, student_scores):
        """對齊教師和學生分數的維度"""
        if teacher_scores.dim() == 4 and student_scores.dim() == 2:
            # 將密集格式轉為標準格式
            b, c, f, n = teacher_scores.shape
            teacher_scores = teacher_scores.view(b, c, -1).mean(dim=2)
        elif student_scores.dim() == 4 and teacher_scores.dim() == 2:
            b, c, f, n = student_scores.shape
            student_scores = student_scores.view(b, c, -1).mean(dim=2)
        
        return teacher_scores, student_scores

    def _compute_classification_distillation_loss(self, teacher_scores, student_scores):
        """計算分類蒸餾損失"""
        if student_scores.dim() <= 2 and teacher_scores.dim() <= 2:
            # 現有的標準格式代碼保持不變
            teacher_probs = F.softmax(teacher_scores, dim=1)
            student_log_probs = F.log_softmax(student_scores, dim=1)
            
            if teacher_probs.shape == student_log_probs.shape:
                cls_distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            else:
                print(f"⚠️ 分數形狀不匹配: student {student_log_probs.shape}, teacher {teacher_probs.shape}")
                min_classes = min(student_scores.size(1), teacher_scores.size(1))
                
                # 創建新張量而非就地修改
                student_scores_subset = student_scores[:, :min_classes].clone()
                teacher_scores_subset = teacher_scores[:, :min_classes].clone()
                
                cls_distill_loss = F.kl_div(
                    F.log_softmax(student_scores_subset, dim=1),
                    F.softmax(teacher_scores_subset, dim=1),
                    reduction='batchmean'
                )
        elif student_scores.dim() == 4 and teacher_scores.dim() == 4:
            # 處理 4D 密集格式
            cls_distill_loss = self._compute_4d_classification_distillation_loss(
                teacher_scores, student_scores)
        else:
            # 維度不一致的情況
            print(f"⚠️ 需要進行維度轉換: student {student_scores.dim()}D, teacher {teacher_scores.dim()}D")
            
            # 將 4D 格式轉為 2D
            if student_scores.dim() == 4:  # [B, C, 4, N] -> [B, C]
                b, c, f, n = student_scores.shape
                student_scores = student_scores.view(b, c, -1).mean(dim=2)
                print(f"✓ 學生分數: 4D → 2D {student_scores.shape}")
            
            if teacher_scores.dim() == 4:  # [B, C, 4, N] -> [B, C]
                b, c, f, n = teacher_scores.shape
                teacher_scores = teacher_scores.view(b, c, -1).mean(dim=2)
                print(f"✓ 教師分數: 4D → 2D {teacher_scores.shape}")
            
            # 現在都是 2D 格式，可以正常計算 KL 散度
            if student_scores.dim() <= 2 and teacher_scores.dim() <= 2:
                teacher_probs = F.softmax(teacher_scores, dim=1)
                student_log_probs = F.log_softmax(student_scores, dim=1)
                
                # 確保類別數匹配
                if teacher_probs.shape[1] != student_log_probs.shape[1]:
                    min_classes = min(teacher_probs.shape[1], student_log_probs.shape[1])
                    
                    # 創建新張量而非就地修改
                    student_scores_subset = student_scores[:, :min_classes].clone()
                    teacher_scores_subset = teacher_scores[:, :min_classes].clone()
                    
                    cls_distill_loss = F.kl_div(
                        F.log_softmax(student_scores_subset, dim=1),
                        F.softmax(teacher_scores_subset, dim=1),
                        reduction='batchmean'
                    )
                else:
                    cls_distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                
                print(f"✓ 維度轉換後的蒸餾損失: {cls_distill_loss.item():.4f}")
            else:
                # 如果轉換後仍然無法處理，則跳過
                print(f"❌ 維度不兼容，無法計算蒸餾損失")
                cls_distill_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return cls_distill_loss

    def _compute_4d_classification_distillation_loss(self, teacher_scores, student_scores):
        """計算 4D 密集格式的分類蒸餾損失"""
        print(f"🔄 處理 4D 密集格式蒸餾損失: student {student_scores.shape}, teacher {teacher_scores.shape}")
        
        # 1. 獲取批次大小和類別數
        batch_size = student_scores.size(0)
        student_classes = student_scores.size(1)
        teacher_classes = teacher_scores.size(1)
        
        # 2. 確保學生和教師分數具有相同的特徵位置數量
        student_positions = student_scores.size(3)
        teacher_positions = teacher_scores.size(3)
        
        if student_positions != teacher_positions:
            print(f"⚠️ 特徵位置數量不匹配: student {student_positions}, teacher {teacher_positions}")
            
            # 決定統一的目標特徵位置數量
            common_positions = min(student_positions, teacher_positions)
            print(f"✓ 統一特徵位置數量: {common_positions}")
            
            # 選擇學生模型的高置信度特徵位置
            if student_positions > common_positions:
                student_scores, student_boxes = self._select_top_feature_positions(
                    student_scores, None, common_positions)
            
            # 選擇教師模型的高置信度特徵位置
            if teacher_positions > common_positions:
                teacher_scores, teacher_boxes = self._select_top_feature_positions(
                    teacher_scores, None, common_positions)
        
        # 3. 確保類別數匹配
        min_classes = min(student_classes, teacher_classes)
        
        # 創建新張量而非就地修改
        student_scores_subset = student_scores[:, :min_classes].clone()
        teacher_scores_subset = teacher_scores[:, :min_classes].clone()
        
        # 4. 展平 4D 張量以執行 KL 散度計算
        # 展平為 [B*4*N, C] 格式
        student_flat = student_scores_subset.permute(0, 2, 3, 1).reshape(-1, min_classes)
        teacher_flat = teacher_scores_subset.permute(0, 2, 3, 1).reshape(-1, min_classes)
        
        # 5. 計算 KL 散度
        teacher_probs = F.softmax(teacher_flat, dim=1)
        student_log_probs = F.log_softmax(student_flat, dim=1)
        cls_distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        print(f"✓ 4D 蒸餾損失計算完成: {cls_distill_loss.item():.4f}")
        return cls_distill_loss

    def _select_top_feature_positions(self, scores, boxes, target_positions):
        """選擇頂部特徵位置"""
        b, c, four, _ = scores.shape
        confidence = scores.max(dim=2)[0]  # [B, C, N]
        
        new_scores = torch.zeros(b, c, four, target_positions, device=scores.device)
        new_boxes = None
        
        if boxes is not None:
            new_boxes = torch.zeros(b, c, target_positions, device=boxes.device)
        
        for bi in range(b):
            for ci in range(c):
                conf = confidence[bi, ci]
                _, indices = torch.topk(conf, k=target_positions)
                for i, idx in enumerate(indices):
                    new_scores[bi, ci, :, i] = scores[bi, ci, :, idx]
                    if boxes is not None:
                        new_boxes[bi, ci, i] = boxes[bi, ci, idx]
        
        return new_scores, new_boxes

    def _compute_box_distillation_loss(self, teacher_boxes, student_boxes, teacher_scores=None, student_scores=None):
        """計算框蒸餾損失"""
        # 將框標準化為相同格式
        if student_boxes.dim() > 2 or teacher_boxes.dim() > 2:
            # 獲取有效的標準格式框
            if student_boxes.dim() > 2:
                student_boxes = self._standardize_boxes(student_boxes)
            
            if teacher_boxes.dim() > 2:
                teacher_boxes = self._standardize_boxes(teacher_boxes)
        
        # 處理數量不同的框
        student_size, teacher_size = student_boxes.size(0), teacher_boxes.size(0)
        if student_size != teacher_size:
            min_size = min(student_size, teacher_size)
            
            if student_size > teacher_size:
                # 只使用信心分數最高的框
                student_boxes = self._select_boxes_by_confidence(
                    student_boxes, min_size, student_scores)
            else:
                # 只使用信心分數最高的目標框
                teacher_boxes = self._select_boxes_by_confidence(
                    teacher_boxes, min_size, teacher_scores)
        
        # 進行規范化避免大數值造成損失爆炸
        box_distill_loss = F.mse_loss(student_boxes, teacher_boxes)
        
        if box_distill_loss > 100:
            scale = min(1.0, 10.0 / box_distill_loss.item())
            box_distill_loss = box_distill_loss * scale
            print(f"⚠️ 框蒸餾損失過大，縮放後: {box_distill_loss.item():.4f}")
        
        return box_distill_loss

    def _standardize_boxes(self, boxes):
        """將框標準化為 [N, 4] 格式"""
        if boxes.dim() == 3 and boxes.size(2) % 4 == 0:
            boxes = boxes.view(-1, 4)
        else:
            # 複雜形狀，使用扁平化後的前N個元素
            flat_boxes = boxes.view(-1)
            elements = (flat_boxes.numel() // 4) * 4
            boxes = flat_boxes[:elements].view(-1, 4)
        
        return boxes

    def _compute_lcp_loss(self, outputs, targets, auxiliary_net):
        """計算 LCP 損失"""
        # 獲取特徵圖
        feature_maps = self._get_feature_maps_for_lcp(outputs, targets)
        
        if feature_maps is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 處理框數據，轉換為標準格式 [K, 5]
        aux_boxes = self._get_boxes_for_lcp(outputs)
        
        if aux_boxes is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 確保框格式正確 - 轉換為 [K, 5] 格式
        roi_format_boxes = self._prepare_roi_format_boxes(aux_boxes)
        
        if roi_format_boxes is None or roi_format_boxes.size(0) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 調用輔助網絡
        aux_outputs = auxiliary_net(feature_maps, roi_format_boxes)
        
        # 計算 LCP 損失
        aux_cls_loss = self._compute_classification_loss(aux_outputs, targets['class_ids'])
        
        # 診斷與輔助網絡輸出有關的信息
        print(f"   🔍 輔助網絡輸出形狀: {[x.shape if isinstance(x, torch.Tensor) else type(x) for x in aux_outputs if x is not None]}")
        print(f"   🔍 目標類別IDs: {targets['class_ids'][:5] if isinstance(targets['class_ids'], list) and len(targets['class_ids']) > 0 else targets['class_ids']}")
        print(f"   🔍 原始 LCP 損失值: {aux_cls_loss.item():.6f}")
        
        # 強化 LCP 損失
        enhanced_lcp_loss = self._enhance_lcp_loss(aux_outputs, aux_cls_loss, targets)
        
        return enhanced_lcp_loss

    def _get_feature_maps_for_lcp(self, outputs, targets):
        """獲取用於 LCP 的特徵圖"""
        feature_maps = None
        
        if isinstance(outputs, dict):
            if 'images' in outputs:
                feature_maps = self.get_feature_map(outputs['images'])
            elif 'feature_maps' in outputs:
                feature_maps = outputs['feature_maps']
        
        # 如果沒有找到特徵圖，嘗試使用輸入重新生成
        if feature_maps is None and 'images' in targets:
            feature_maps = self.get_feature_map(targets['images'])
        
        return feature_maps

    def _get_boxes_for_lcp(self, outputs):
        """獲取用於 LCP 的框"""
        aux_boxes = None
        
        # 獲取輸出框
        if isinstance(outputs, dict) and 'boxes' in outputs:
            aux_boxes = outputs['boxes']
        elif isinstance(outputs, tuple) and len(outputs) >= 2:
            aux_boxes = outputs[1]
        
        return aux_boxes

    def _prepare_roi_format_boxes(self, aux_boxes):
        """準備 ROI 格式的框 [K, 5]"""
        try:
            print(f"⚠️ LCP: 原始框形狀 {aux_boxes.shape}")
            
            # 首先將框轉為標準 [N, 4] 格式
            if aux_boxes.dim() > 2:
                # OS2D 密集格式框處理
                if aux_boxes.dim() == 3:
                    # 形狀 [B, C, N] 特殊格式
                    flat_boxes = aux_boxes.reshape(-1)
                    n_elements = 4 * (flat_boxes.numel() // 4)
                    flat_boxes = flat_boxes[:n_elements]
                    aux_boxes = flat_boxes.reshape(-1, 4)
                elif aux_boxes.dim() == 4:
                    # 形狀 [B, C, 4, N]
                    b, c, four, n = aux_boxes.shape
                    aux_boxes = aux_boxes.permute(0, 1, 3, 2).reshape(-1, 4)
                else:
                    aux_boxes = aux_boxes.reshape(-1, 4)
            
            # 確保形狀是 [N, 4] 且數值有效
            if aux_boxes.shape[-1] != 4:
                print(f"⚠️ 調整框維度: {aux_boxes.shape} → [N, 4]")
                aux_boxes = aux_boxes.reshape(-1, 4)
            
            # 限制框數量，避免 OOM
            MAX_BOXES = 500  # 減少到 500 以保證效率
            if aux_boxes.shape[0] > MAX_BOXES:
                # 使用隨機採樣
                indices = torch.randperm(aux_boxes.shape[0], device=aux_boxes.device)[:MAX_BOXES]
                aux_boxes = aux_boxes[indices]
                print(f"✓ LCP: 隨機選擇了 {MAX_BOXES} 個框")
            
            # 檢查並修正框坐標
            aux_boxes = torch.nan_to_num(aux_boxes, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 確保 x1 < x2, y1 < y2
            x1, y1, x2, y2 = aux_boxes.unbind(1)
            
            # 將x1,y1設為min，將x2,y2設為max - 使用新張量而非就地操作
            new_x1 = torch.min(x1, x2)
            new_x2 = torch.max(x1, x2)
            new_y1 = torch.min(y1, y2)
            new_y2 = torch.max(y1, y2)
            
            # 確保最小尺寸
            min_size = 1e-3
            new_x2 = torch.max(new_x2, new_x1 + min_size)
            new_y2 = torch.max(new_y2, new_y1 + min_size)
            
            # 重組框 - 使用新張量而非就地操作
            aux_boxes = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)
            
            # 添加批次索引形成 [N, 5]
            batch_indices = torch.zeros(aux_boxes.shape[0], 1, device=aux_boxes.device)
            roi_format_boxes = torch.cat([batch_indices, aux_boxes], dim=1)
            
            # 驗證並修正ROI格式
            if roi_format_boxes.size(1) != 5:
                print(f"⚠️ ROI格式不正確: {roi_format_boxes.shape}，調整為 [K, 5]")
                
                if roi_format_boxes.size(1) > 5:
                    # 如果列數過多，截取前5列
                    roi_format_boxes = roi_format_boxes[:, :5]
                elif roi_format_boxes.size(1) < 5:
                    # 如果列數不足，擴展到5列
                    padding = torch.zeros(roi_format_boxes.size(0), 5 - roi_format_boxes.size(1), 
                                        device=roi_format_boxes.device)
                    roi_format_boxes = torch.cat([roi_format_boxes, padding], dim=1)
            
            # 確認最終形狀
            print(f"✓ LCP: 最終框形狀 {roi_format_boxes.shape}, 符合 [K, 5] 格式")
            return roi_format_boxes
        except Exception as e:
            print(f"   ❌ LCP 框處理失敗: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def _enhance_lcp_loss(self, aux_outputs, aux_cls_loss, targets):
        """增強 LCP 損失以確保有效的梯度"""
        # 1. 先提取分類分數並確保數值穩定
        if isinstance(aux_outputs, tuple) and len(aux_outputs) > 0:
            aux_scores = aux_outputs[0]
            aux_scores = torch.nan_to_num(aux_scores, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # 2. 計算分類矩陣的統計數據
            batch_size = aux_scores.size(0)
            num_classes = aux_scores.size(1)
            
            # 3. 強化 LCP 損失計算策略
            if aux_cls_loss.item() < 0.01:  # 損失太小時採用多樣化策略
                print(f"   ⚠️ 偵測到較小的 LCP 損失 ({aux_cls_loss.item():.6f})，應用混合強化技術")
                
                # 創建新的損失張量
                enhanced_loss = aux_cls_loss.clone()
                
                # 3.1 L2 正則化 - 針對低活躍度的分類器
                l2_loss = 0.01 * torch.mean(aux_scores.pow(2))
                
                # 3.2 熵正則化 - 鼓勵更多樣化的預測
                log_probs = F.log_softmax(aux_scores, dim=1)
                probs = torch.exp(log_probs)
                entropy_loss = -0.05 * torch.mean(torch.sum(probs * log_probs, dim=1))
                
                # 3.3 Focal Loss 項 - 專注於難分類樣本
                gt_labels = self._prepare_gt_labels_for_lcp(targets['class_ids'], batch_size, num_classes)
                
                # 應用 one-hot 編碼
                gt_one_hot = F.one_hot(gt_labels.long() % num_classes, num_classes=num_classes).float()
                
                # Focal Loss 項
                focal_weight = (1 - probs) ** 2  # 對低置信度預測給予更高權重
                focal_loss = 0.1 * torch.mean(focal_weight * F.cross_entropy(aux_scores, gt_labels.long() % num_classes, reduction='none'))
                
                # 組合所有損失
                print(f"   ✓ L2 正則化: {l2_loss.item():.6f}")
                print(f"   ✓ 熵正則化: {entropy_loss.item():.6f}")
                print(f"   ✓ Focal Loss: {focal_loss.item():.6f}")
                
                # 組合所有損失 - 確保最終值在合理範圍
                enhanced_loss = enhanced_loss + l2_loss + entropy_loss + focal_loss
                print(f"   ✓ 混合 LCP 損失: {enhanced_loss.item():.6f}")
            else:
                print(f"   ✓ 原始 LCP 損失值足夠大，無需增強: {aux_cls_loss.item():.6f}")
                enhanced_loss = aux_cls_loss
            
            # 4. 確保損失值在合理範圍 (不要太大也不要太小)
            if enhanced_loss.item() < 0.1:
                # 如果太小，確保至少有一個最小值
                min_loss = torch.tensor(0.1, device=enhanced_loss.device)
                enhanced_loss = torch.max(enhanced_loss, min_loss)
                print(f"   ⚠️ 應用最小 LCP 損失閾值: 0.1")
            elif enhanced_loss.item() > 5.0:
                # 如果太大，進行縮放
                enhanced_loss = 5.0 * enhanced_loss / enhanced_loss.item()
                print(f"   ⚠️ 應用 LCP 損失上限: 5.0")
            
            return enhanced_loss
        
        return aux_cls_loss

    def _prepare_gt_labels_for_lcp(self, class_ids, batch_size, num_classes):
        """準備用於 LCP 的 GT 標籤"""
        # 確保 gt_labels 是張量並且形狀正確
        if isinstance(class_ids, list):
            if len(class_ids) > 0 and isinstance(class_ids[0], torch.Tensor):
                gt_labels = class_ids[0]  # 取第一個張量
            else:
                gt_labels = torch.tensor(class_ids[0] if len(class_ids) > 0 else 0, device=self.device)
        else:
            gt_labels = class_ids
        
        if not isinstance(gt_labels, torch.Tensor):
            gt_labels = torch.tensor(gt_labels, device=self.device)
        
        # 擴展 gt_labels 以匹配 batch_size
        if gt_labels.numel() == 1:
            gt_labels = gt_labels.expand(batch_size)
        elif gt_labels.numel() < batch_size:
            # 循環擴展到 batch_size
            repeats = batch_size // gt_labels.numel() + 1
            gt_labels = gt_labels.repeat(repeats)[:batch_size]
        
        return gt_labels

    def _scale_losses(self, cls_loss, box_loss, teacher_loss, lcp_loss):
        """縮放損失以避免異常值"""
        max_cls_loss = 10.0
        max_box_loss = 10.0
        max_teacher_loss = 10.0
        max_lcp_loss = 10.0
        
        # 損失縮放 - 避免使用 inplace 操作，創建新的張量
        scaled_cls_loss = torch.min(cls_loss, torch.tensor(max_cls_loss, device=cls_loss.device))
        scaled_box_loss = torch.min(box_loss, torch.tensor(max_box_loss, device=box_loss.device))
        scaled_teacher_loss = torch.min(teacher_loss, torch.tensor(max_teacher_loss, device=teacher_loss.device))
        scaled_lcp_loss = torch.min(lcp_loss, torch.tensor(max_lcp_loss, device=lcp_loss.device))
        
        # 檢測異常值並打印
        if cls_loss > max_cls_loss or box_loss > max_box_loss or teacher_loss > max_teacher_loss or lcp_loss > max_lcp_loss:
            print(f"⚠️ 檢測到損失異常值，進行損失縮放")
            print(f"   分類損失: {cls_loss.item():.4f} → {scaled_cls_loss.item():.4f}")
            print(f"   框回歸損失: {box_loss.item():.4f} → {scaled_box_loss.item():.4f}")
            print(f"   教師損失: {teacher_loss.item():.4f} → {scaled_teacher_loss.item():.4f}")
            print(f"   LCP損失: {lcp_loss.item():.4f} → {scaled_lcp_loss.item():.4f}")
        
        return scaled_cls_loss, scaled_box_loss, scaled_teacher_loss, scaled_lcp_loss

    
    from tqdm import tqdm
    def train_one_epoch(self, train_loader, optimizer, 
                  auxiliary_net=None, device=None, 
                  print_freq=10, scheduler=None, 
                  loss_weights=None, use_lcp_loss=True, 
                  max_batches=None, max_predictions=100,
                  use_feature_pyramid=True,  
                  pyramid_scales=[1.0, 0.75, 0.5],  
                  nms_threshold=0.5,  
                  apply_nms=True):
        """
        訓練模型一個 epoch，支援 LCP 損失、特徵金字塔與 NMS 框數量控制
        """
        import torch
        import time
        from tqdm import tqdm
        
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
        
        # 更新分類器以匹配類別數量
        self._update_classifier_for_classes(class_num, device)
        
        # 更新輔助網路（如果需要）
        if auxiliary_net is not None:
            self._update_auxiliary_classifier(auxiliary_net, class_num, device)
        
        # 初始化損失記錄和統計信息
        loss_history = []
        loss_stats = {'total': 0, 'cls': 0, 'box_reg': 0, 'teacher': 0, 'lcp': 0}
        batch_count = 0
        
        # 確定要處理的批次數
        num_batches = len(train_loader)
        if max_batches is not None:
            num_batches = min(num_batches, max_batches)
        
        # 打印配置信息
        self._print_training_config(use_feature_pyramid, pyramid_scales, apply_nms, nms_threshold)
        
        # 設定全局統一的目標特徵位置數量
        target_feature_count = min(100, max_predictions)
        print(f"📊 設置統一特徵位置數量: {target_feature_count}")

        # 創建進度條
        with tqdm(range(num_batches), desc="訓練進度") as pbar:
            for batch_idx in pbar:
                try:
                    # 記錄批次開始時間
                    batch_start_time = time.time()
                    
                    # 獲取當前批次數據
                    batch_data = train_loader.get_batch(batch_idx)
                    images, class_images, loc_targets, class_targets, batch_class_ids, class_image_sizes, box_inverse_transform, batch_boxes, batch_img_size = batch_data
                    
                    # 打印批次信息
                    self._print_batch_info(batch_idx, num_batches, images, batch_class_ids, batch_boxes)
                    
                    # 將數據移至指定設備
                    images = images.to(device)
                    class_images = [img.to(device) if isinstance(img, torch.Tensor) else img for img in class_images] if isinstance(class_images, list) else class_images
                    
                    # 更新輔助網路通道數（如果需要）
                    if auxiliary_net is not None:
                        self._update_auxiliary_channels(auxiliary_net, images, device)
                    
                    # 清零梯度
                    optimizer.zero_grad()
                    
                    # 使用特徵金字塔或標準模式進行預測
                    if use_feature_pyramid:
                        print(f"\n📊 使用特徵金字塔進行推理，尺度: {pyramid_scales}")
                        
                        # 教師模型特徵金字塔預測
                        teacher_outputs = self._run_teacher_feature_pyramid(
                            images, class_images, pyramid_scales, 
                            max_predictions, nms_threshold, apply_nms, 
                            target_feature_count
                        )
                        
                        # 學生模型特徵金字塔預測
                        outputs = self._run_student_feature_pyramid(
                            images, class_images, pyramid_scales, 
                            max_predictions, nms_threshold, apply_nms, 
                            target_feature_count, batch_boxes
                        )
                    else:
                        # 不使用特徵金字塔的標準處理方式
                        print(f"\n⚡ 使用標準模式 (無特徵金字塔)")
                        teacher_outputs = self._run_standard_inference(
                            self.teacher_model, images, class_images, 
                            max_predictions, nms_threshold, apply_nms
                        )
                        outputs = self._run_standard_inference(
                            self, images, class_images, 
                            max_predictions, nms_threshold, apply_nms
                        )
                    
                    # 分析和顯示輸出信息
                    self._analyze_model_outputs(outputs, teacher_outputs)
                    
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
                    outputs_dict = self._prepare_outputs_dict(outputs, images, class_images, use_feature_pyramid, pyramid_scales)
                    
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
                    
                    # 打印損失信息
                    loss_time = time.time() - loss_start
                    self._print_loss_info(loss_dict, loss_time)
                    
                    # 反向傳播
                    backprop_start = time.time()
                    print(f"\n🔄 開始反向傳播...")

                    # 使用 clone 避免修改原始損失
                    loss_for_backward = loss.clone()

                    # 使用 set_detect_anomaly 幫助定位問題
                    with torch.autograd.set_detect_anomaly(True):
                        loss_for_backward.backward()

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
                    
                    # 更新損失統計
                    loss_stats['total'] += loss_value
                    loss_stats['cls'] += loss_dict['cls_loss'].item()
                    loss_stats['box_reg'] += loss_dict['box_loss'].item()
                    loss_stats['teacher'] += loss_dict['teacher_loss'].item()
                    loss_stats['lcp'] += loss_dict['lcp_loss'].item()
                    batch_count += 1
                    
                    # 更新進度條描述
                    pbar.set_description(
                        f"Loss: {loss_value:.4f} (cls: {loss_dict['cls_loss'].item():.4f}, "
                        f"box: {loss_dict['box_loss'].item():.4f})"
                    )
                    
                    # 打印詳細信息
                    if print_freq > 0 and (batch_idx % print_freq == 0 or batch_idx == num_batches - 1):
                        self._print_batch_summary(batch_idx, num_batches, loss_dict, loss_value)
                    
                    # 計算批次總耗時
                    batch_time = time.time() - batch_start_time
                    print(f"\n✓ 批次 {batch_idx+1}/{num_batches} 完成，總耗時: {batch_time:.2f}秒\n")
                    print("-" * 80)
                
                except Exception as e:
                    print(f"❌ 批次 {batch_idx+1} 處理出錯: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # 計算並打印訓練結果摘要
        self._print_training_summary(
            batch_count, num_batches, loss_stats, 
            start_time, use_feature_pyramid, pyramid_scales, 
            apply_nms, nms_threshold
        )
        
        return loss_history

    def _update_classifier_for_classes(self, class_num, device):
        """更新分類器以匹配類別數量"""
        if hasattr(self, 'classifier') and hasattr(self.classifier, 'out_features'):
            if self.classifier.out_features != class_num:
                print(f"⚠️ 更新分類器輸出維度: {self.classifier.out_features} → {class_num}")
            in_features = self.classifier.in_features
            self.classifier = nn.Linear(in_features, class_num).to(device)

    def _update_auxiliary_classifier(self, auxiliary_net, class_num, device):
        """更新輔助網路分類器"""
        if hasattr(auxiliary_net, 'classifier') and hasattr(auxiliary_net.classifier, 'out_features'):
            if auxiliary_net.classifier.out_features != class_num:
                print(f"⚠️ 更新輔助網路分類器維度: {auxiliary_net.classifier.out_features} → {class_num}")
            aux_in_features = auxiliary_net.classifier.in_features
            auxiliary_net.classifier = nn.Linear(aux_in_features, class_num).to(device)

    def _print_training_config(self, use_feature_pyramid, pyramid_scales, apply_nms, nms_threshold):
        """打印訓練配置信息"""
        print(f"🔍 特徵金字塔狀態: {'啟用' if use_feature_pyramid else '停用'}")
        if use_feature_pyramid:
            print(f"📊 金字塔尺度: {pyramid_scales}")
        print(f"🧹 NMS 狀態: {'啟用' if apply_nms else '停用'}, 閾值: {nms_threshold}")

    def _print_batch_info(self, batch_idx, num_batches, images, batch_class_ids, batch_boxes):
        """打印批次基本信息"""
        print(f"✓ 批次 {batch_idx+1}/{num_batches} 數據載入完成")
        print(f"   - 圖像形狀: {images.shape}")
        print(f"   - 類別數量: {len(batch_class_ids)}")
        print(f"   - 邊界框數量: {sum(1 for box in batch_boxes if isinstance(box, torch.Tensor) and box.numel() > 0)}")

    def _update_auxiliary_channels(self, auxiliary_net, images, device):
        """更新輔助網路的輸入通道數"""
        feature_maps = self.get_feature_map(images)
        if isinstance(feature_maps, torch.Tensor):
            current_channels = feature_maps.size(1)
            if auxiliary_net.get_current_channels() != current_channels:
                print(f"✓ 更新輔助網路輸入通道: {auxiliary_net.get_current_channels()} → {current_channels}")
                auxiliary_net.update_input_channels(current_channels)

    def _run_standard_inference(self, model, images, class_images, max_predictions, nms_threshold, apply_nms):
        """執行標準模式的推理"""
        import torch
        
        with torch.no_grad() if model != self else torch.enable_grad():
            outputs = model(images, class_images=class_images)
            if apply_nms:
                outputs = self._apply_nms_to_outputs(
                    outputs, 
                    max_boxes=max_predictions, 
                    iou_threshold=nms_threshold
                )
        return outputs

    def _run_teacher_feature_pyramid(self, images, class_images, pyramid_scales, max_predictions, nms_threshold, apply_nms, target_feature_count):
        """執行教師模型的特徵金字塔推理"""
        import torch
        import torch.nn.functional as F
        import time
        
        all_teacher_scores = []
        all_teacher_boxes = []
        all_teacher_extras = []
        
        with torch.no_grad():
            for scale_idx, scale in enumerate(pyramid_scales):
                scale_start = time.time()
                print(f"\n🔍 處理教師模型尺度 {scale_idx+1}/{len(pyramid_scales)}: {scale}")
                
                # 縮放輸入
                scaled_images, scaled_class_images = self._scale_inputs(images, class_images, scale)
                
                # 教師模型預測
                print(f"   運行教師模型推理中...")
                teacher_outputs = self.teacher_model(scaled_images, class_images=scaled_class_images)
                
                # 應用 NMS 減少框數量
                if apply_nms:
                    teacher_outputs = self._apply_nms_to_outputs(
                        teacher_outputs, 
                        max_boxes=max_predictions, 
                        iou_threshold=nms_threshold
                    )
                
                if isinstance(teacher_outputs, tuple) and len(teacher_outputs) >= 2:
                    teacher_scores, teacher_boxes = teacher_outputs[0], teacher_outputs[1]
                    
                    # 限制預測數量
                    if not apply_nms and teacher_boxes.shape[0] > max_predictions:
                        keep_idx = torch.randperm(teacher_boxes.shape[0])[:max_predictions]
                        teacher_scores = teacher_scores[keep_idx]
                        teacher_boxes = teacher_boxes[keep_idx]
                        teacher_extras = tuple(extra[keep_idx] if isinstance(extra, torch.Tensor) else extra 
                                            for extra in teacher_outputs[2:]) if len(teacher_outputs) > 2 else ()
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
                    teacher_outputs = self._merge_dense_outputs(
                        all_teacher_scores, all_teacher_boxes, all_teacher_extras, 
                        target_feature_count, apply_nms, max_predictions, nms_threshold
                    )
                else:
                    teacher_outputs = self._merge_standard_outputs(
                        all_teacher_scores, all_teacher_boxes, all_teacher_extras, 
                        target_feature_count, apply_nms, max_predictions, nms_threshold
                    )
                
                merge_time = time.time() - merge_start
                print(f"✓ 教師模型特徵金字塔合併完成，耗時: {merge_time:.2f}秒")
                
            else:
                print("⚠️ 教師模型未產生有效輸出，使用原始模式")
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images, class_images=class_images)
                    if apply_nms:
                        teacher_outputs = self._apply_nms_to_outputs(
                            teacher_outputs, 
                            max_boxes=max_predictions, 
                            iou_threshold=nms_threshold
                        )
        except Exception as e:
            print(f"❌ 合併教師模型結果失敗: {e}")
            import traceback
            traceback.print_exc()
            # 使用原始模式作為備選方案
            with torch.no_grad():
                teacher_outputs = self.teacher_model(images, class_images=class_images)
                if apply_nms:
                    teacher_outputs = self._apply_nms_to_outputs(
                        teacher_outputs, 
                        max_boxes=max_predictions, 
                        iou_threshold=nms_threshold
                    )
        
        return teacher_outputs

    def _run_student_feature_pyramid(self, images, class_images, pyramid_scales, max_predictions, nms_threshold, apply_nms, target_feature_count, batch_boxes):
        """執行學生模型的特徵金字塔推理"""
        import torch
        import torch.nn.functional as F
        import time
        
        # 從 batch_boxes 確定目標框數量
        target_size = sum(1 for box in batch_boxes if isinstance(box, torch.Tensor) and box.numel() > 0)
        common_feature_count = min(max(target_size, 30), 100) if target_size > 0 else min(target_feature_count, 50)
        print(f"🔄 將使用 {common_feature_count} 個高置信度特徵進行特徵金字塔處理")
        
        all_student_scores = []
        all_student_boxes = []
        all_student_extras = []
        
        for scale_idx, scale in enumerate(pyramid_scales):
            scale_start = time.time()
            print(f"\n🔍 處理學生模型尺度 {scale_idx+1}/{len(pyramid_scales)}: {scale}")
            
            # 縮放輸入
            scaled_images, scaled_class_images = self._scale_inputs(images, class_images, scale)
            
            # 學生模型預測
            print(f"   運行學生模型推理中...")
            outputs = self(scaled_images, class_images=scaled_class_images)
            
            # 應用 NMS 減少框數量
            if apply_nms:
                outputs = self._apply_nms_to_outputs(
                    outputs, 
                    max_boxes=max_predictions, 
                    iou_threshold=nms_threshold
                )
            
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                student_scores, student_boxes = outputs[0], outputs[1]
                
                # 限制預測數量
                if not apply_nms and student_boxes.shape[0] > max_predictions:
                    keep_idx = torch.randperm(student_boxes.shape[0])[:max_predictions]
                    student_scores = student_scores[keep_idx]
                    student_boxes = student_boxes[keep_idx]
                    student_extras = tuple(extra[keep_idx] if isinstance(extra, torch.Tensor) else extra 
                                        for extra in outputs[2:]) if len(outputs) > 2 else ()
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
                    outputs = self._merge_dense_outputs(
                        all_student_scores, all_student_boxes, all_student_extras, 
                        common_feature_count, apply_nms, max_predictions, nms_threshold
                    )
                else:
                    outputs = self._merge_standard_outputs(
                        all_student_scores, all_student_boxes, all_student_extras, 
                        common_feature_count, apply_nms, max_predictions, nms_threshold
                    )
                
                merge_time = time.time() - merge_start
                print(f"✓ 學生模型特徵金字塔合併完成，耗時: {merge_time:.2f}秒")
            else:
                print("⚠️ 學生模型未產生有效輸出，使用原始模式")
                outputs = self(images, class_images=class_images)
        except Exception as e:
            print(f"❌ 合併學生模型結果失敗: {e}")
            import traceback
            traceback.print_exc()
            outputs = self(images, class_images=class_images)
            if apply_nms:
                outputs = self._apply_nms_to_outputs(
                    outputs, 
                    max_boxes=max_predictions, 
                    iou_threshold=nms_threshold
                )
        
        return outputs

    def _scale_inputs(self, images, class_images, scale):
        """縮放輸入圖像和類別圖像"""
        import torch.nn.functional as F
        
        if scale == 1.0:
            return images, class_images
        
        # 縮放圖像
        scaled_images = images
        if images.dim() == 4:  # [B, C, H, W]
            h, w = images.shape[2:]
            new_h, new_w = int(h * scale), int(w * scale)
            print(f"   縮放圖像: {h}x{w} → {new_h}x{new_w}")
            scaled_images = F.interpolate(
                images, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
        else:
            print(f"   跳過圖像縮放，維度不是4D: {images.dim()}")
        
        # 縮放類別圖像
        scaled_class_images = class_images
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
            print(f"   跳過類別圖像縮放，不是張量列表")
        
        return scaled_images, scaled_class_images

    def _merge_dense_outputs(self, all_scores, all_boxes, all_extras, target_feature_count, apply_nms, max_predictions, nms_threshold):
        """合併密集格式的輸出"""
        import torch
        
        try:
            scores = torch.cat(all_scores, dim=3)
            boxes = torch.cat(all_boxes, dim=2)
            print(f"✓ 合併成功: scores {scores.shape}, boxes {boxes.shape}")
            
            # 根據置信度過濾，保留固定數量的高置信度特徵位置
            if scores.size(3) > target_feature_count:
                print(f"⚠️ 特徵位置過多 ({scores.size(3)}), 將基於置信度選擇 {target_feature_count} 個")
                
                # 計算每個位置的置信度 (每個位置的最大分類分數)
                b, c, four, n = scores.shape
                confidence_scores = scores.max(dim=2)[0]  # [B, C, N]
                
                # 為每個批次和類別選擇最高置信度的位置
                filtered_scores = torch.zeros(b, c, four, target_feature_count, device=scores.device)
                filtered_boxes = torch.zeros(b, c, target_feature_count, device=boxes.device)
                
                for bi in range(b):
                    for ci in range(c):
                        # 選擇此批次此類別中置信度最高的位置
                        conf = confidence_scores[bi, ci]  # [N]
                        _, indices = torch.topk(conf, k=min(target_feature_count, conf.size(0)))
                        
                        # 複製選定的位置
                        for i, idx in enumerate(indices):
                            if i < target_feature_count:
                                filtered_scores[bi, ci, :, i] = scores[bi, ci, :, idx]
                                filtered_boxes[bi, ci, i] = boxes[bi, ci, idx]
                
                # 更新結果
                scores = filtered_scores
                boxes = filtered_boxes
                print(f"✓ 特徵位置過濾完成: {n} → {target_feature_count}")
            
            # 合併額外輸出
            extras = []
            if all_extras:
                for i in range(len(all_extras[0])):
                    extra_tensors = []
                    for scale_extras in all_extras:
                        if i < len(scale_extras) and scale_extras[i] is not None:
                            extra_tensors.append(scale_extras[i])
                    
                    if extra_tensors and all(isinstance(e, torch.Tensor) for e in extra_tensors):
                        # 確定合併維度
                        if extra_tensors[0].dim() == scores.dim():
                            extras.append(torch.cat(extra_tensors, dim=3))  # 與 scores 相同維度
                        elif extra_tensors[0].dim() == boxes.dim():
                            extras.append(torch.cat(extra_tensors, dim=2))  # 與 boxes 相同維度
                        else:
                            extras.append(extra_tensors[0])  # 無法確定，使用第一個
                    else:
                        extras.append(None)
            
            outputs = (scores, boxes) + tuple(e for e in extras if e is not None)
            
            # 應用 NMS
            if apply_nms:
                outputs = self._apply_nms_to_outputs(
                    outputs, 
                    max_boxes=max_predictions, 
                    iou_threshold=nms_threshold
                )
            
            return outputs
        except RuntimeError as e:
            print(f"❌ 密集格式合併失敗: {e}")
            # 使用第一個尺度的結果作為備選
            return (all_scores[0], all_boxes[0])

    def _merge_standard_outputs(self, all_scores, all_boxes, all_extras, target_feature_count, apply_nms, max_predictions, nms_threshold):
        """合併標準格式的輸出"""
        import torch
        
        try:
            scores = torch.cat(all_scores, dim=0)
            boxes = torch.cat(all_boxes, dim=0)
            print(f"✓ 合併成功: scores {scores.shape}, boxes {boxes.shape}")
            
            # 合併後根據置信度篩選
            target_size = target_feature_count
            
            if apply_nms:
                # 使用 NMS 過濾框
                outputs = self._apply_nms_to_outputs(
                    (scores, boxes), 
                    max_boxes=target_size, 
                    iou_threshold=nms_threshold
                )
            else:
                # 獲取每個框的最高類別分數作為置信度
                confidence_scores, _ = scores.max(dim=1)
                
                # 按置信度選擇前 target_size 個框
                if boxes.size(0) > target_size:
                    _, indices = torch.topk(confidence_scores, k=target_size)
                    filtered_scores = scores[indices]
                    filtered_boxes = boxes[indices]
                    
                    # 處理額外輸出
                    if all_extras:
                        filtered_extras = []
                        for extra in all_extras:
                            if isinstance(extra, torch.Tensor) and extra.size(0) == scores.size(0):
                                filtered_extras.append(extra[indices])
                            else:
                                filtered_extras.append(extra)
                        outputs = (filtered_scores, filtered_boxes) + tuple(filtered_extras)
                    else:
                        outputs = (filtered_scores, filtered_boxes)
                    
                    print(f"✓ 按信心分數選擇了 {target_size} 個最高得分框 (共 {boxes.size(0)} 個框)")
                else:
                    outputs = (scores, boxes)
            
            return outputs
        except RuntimeError as e:
            print(f"❌ 標準格式合併失敗: {e}")
            return (all_scores[0], all_boxes[0])

    def _apply_nms_to_outputs(self, outputs, max_boxes=100, iou_threshold=0.5):
        """對模型輸出應用 NMS 以減少框數量"""
        import torch
        import torchvision.ops
        
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            return outputs
        
        class_scores, boxes = outputs[0], outputs[1]
        
        # 處理 OS2D 密集格式
        if boxes.dim() > 2:
            print(f"⚠️ NMS: 處理密集格式框張量 {boxes.shape}")
            
            # 如果是密集格式 [1, 9, 56661]，先將其轉換為標準格式
            if boxes.dim() == 3:  # [B, C, N] 格式
                # 獲取每個位置的最高類別分數
                if class_scores.dim() == 4:  # [B, C, 4, N]
                    confidence_scores = class_scores.max(dim=2)[0]  # [B, C, N]
                    confidence_flat = confidence_scores.view(-1)  # [B*C*N]
                else:
                    confidence_flat = torch.ones(boxes.numel() // 4, device=boxes.device)
                    
                # 展平框張量
                b, c, n = boxes.shape
                boxes_flat = boxes.view(-1)  # [B*C*N]
                num_boxes = len(boxes_flat) // 4
                
                # 確保能被4整除
                valid_elements = 4 * num_boxes
                boxes_flat = boxes_flat[:valid_elements]
                boxes_2d = boxes_flat.view(num_boxes, 4)  # [N, 4]
                
                # 按置信度選擇框
                if confidence_flat.numel() >= num_boxes:
                    confidence_flat = confidence_flat[:num_boxes]
                    # 按置信度排序
                    _, indices = torch.sort(confidence_flat, descending=True)
                    boxes_2d = boxes_2d[indices]
                    
                print(f"✓ NMS: 轉換密集格式為標準格式 {boxes.shape} → {boxes_2d.shape}")
                
                # 使用標準 NMS
                keep_indices = torchvision.ops.nms(
                    boxes_2d,
                    torch.ones(boxes_2d.size(0), device=boxes_2d.device),  # 使用同樣的分數
                    iou_threshold=iou_threshold
                )[:max_boxes]
                
                # 獲取結果
                nms_boxes = boxes_2d[keep_indices]
                
                # 重新構建輸出 - 注意這裡我們需要保持原有的維度結構
                # 由於格式不兼容，我們只能返回過濾後的框集合
                if class_scores.dim() == 4:  # [B, C, 4, N]
                    # 選擇對應的分數
                    # 選擇最高分數的前 k 個類別和位置
                    b, c, four, n = class_scores.shape
                    scores_flat = class_scores.view(b, c, -1)
                    _, top_idxs = scores_flat.max(dim=2)[0].view(-1).topk(min(len(keep_indices), c))
                    filtered_scores = torch.zeros(1, c, 4, len(keep_indices), device=class_scores.device)
                    # 保留最高分數的類別
                    for i, idx in enumerate(top_idxs):
                        if i < len(keep_indices):
                            filtered_scores[0, idx % c, :, i] = 1.0
                    
                    # 構建新的框張量
                    filtered_boxes = torch.zeros(1, c, len(keep_indices), device=boxes.device)
                    for i in range(len(keep_indices)):
                        if i < nms_boxes.size(0):
                            filtered_boxes[0, :, i] = nms_boxes[i].sum()  # 簡化處理
                else:
                    # 簡化：直接使用第一個結果
                    filtered_scores = class_scores
                    filtered_boxes = nms_boxes
                
                # 重建輸出元組
                filtered_extras = outputs[2:] if len(outputs) > 2 else ()
                outputs = (filtered_scores, filtered_boxes) + filtered_extras
                
                print(f"✓ NMS: 框數量 {num_boxes} → {len(keep_indices)} (閾值={iou_threshold}, 最大框數={max_boxes})")
                return outputs
        
        # 標準格式 NMS 處理
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

    def _analyze_model_outputs(self, outputs, teacher_outputs):
        """分析和顯示模型輸出信息"""
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

    def _prepare_outputs_dict(self, outputs, images, class_images, use_feature_pyramid, pyramid_scales):
        """處理 outputs 結構，準備用於損失計算"""
        if not isinstance(outputs, dict):
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                outputs_dict = {
                    'class_scores': outputs[0],
                    'boxes': outputs[1],
                    'images': images,
                    'class_images': class_images,
                    'feature_pyramid': use_feature_pyramid,
                    'pyramid_scales': pyramid_scales
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
        
        return outputs_dict

    def _print_loss_info(self, loss_dict, loss_time):
        """打印損失信息"""
        print(f"✓ 損失計算完成，耗時: {loss_time:.2f}秒")
        print(f"   - 分類損失: {loss_dict['cls_loss'].item():.4f}")
        print(f"   - 框回歸損失: {loss_dict['box_loss'].item():.4f}")
        print(f"   - 教師損失: {loss_dict['teacher_loss'].item():.4f}")
        print(f"   - LCP損失: {loss_dict['lcp_loss'].item():.4f}")

    def _clip_gradients(self, model, auxiliary_net=None):
        """梯度裁剪"""
        import torch
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        if auxiliary_net is not None:
            torch.nn.utils.clip_grad_norm_(auxiliary_net.parameters(), max_norm=10.0)

    def _print_batch_summary(self, batch_idx, num_batches, loss_dict, loss_value):
        """打印批次摘要信息"""
        print(f"\n批次 {batch_idx+1}/{num_batches} 摘要:")
        print(f"  分類損失: {loss_dict['cls_loss'].item():.4f}")
        print(f"  框回歸損失: {loss_dict['box_loss'].item():.4f}")
        print(f"  教師損失: {loss_dict['teacher_loss'].item():.4f}")
        print(f"  LCP損失: {loss_dict['lcp_loss'].item():.4f}")
        print(f"  總損失: {loss_value:.4f}")

    def _print_training_summary(self, batch_count, num_batches, loss_stats, start_time, use_feature_pyramid, pyramid_scales, apply_nms, nms_threshold):
        """打印訓練結果摘要"""
        import time
        import datetime
        
        # 計算平均損失
        avg_loss = loss_stats['total'] / batch_count if batch_count > 0 else 0
        avg_cls_loss = loss_stats['cls'] / batch_count if batch_count > 0 else 0
        avg_box_loss = loss_stats['box_reg'] / batch_count if batch_count > 0 else 0
        avg_teacher_loss = loss_stats['teacher'] / batch_count if batch_count > 0 else 0
        avg_lcp_loss = loss_stats['lcp'] / batch_count if batch_count > 0 else 0
        
        # 輸出訓練結果摘要
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


    def finetune(self, train_loader, auxiliary_net, prune_layers, prune_ratio=0.3,
                optimizer=None, device=None, epochs_per_layer=1, print_freq=1, max_batches=3):
        pass

    
    def load_checkpoint(self, checkpoint_path, device=None):
        pass

    
    def save_checkpoint(self, checkpoint_path):
        pass

    def _eval(self, dataloader, iou_thresh=0.5, batch_size=4, cfg=None, criterion=None, print_per_class_results=False):
        pass