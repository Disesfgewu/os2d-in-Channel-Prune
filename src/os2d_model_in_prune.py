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
from collections import defaultdict
from os2d.modeling.model import Os2dModel
from os2d.modeling.feature_extractor import build_feature_extractor
from src.lcp_channel_selector import OS2DChannelSelector
from src.gIoU_loss import GIoULoss

class Os2dModelInPrune(Os2dModel):
    """
    擴展 OS2D 模型以支持通道剪枝功能
    """
    def __init__(self, logger=None, is_cuda=False, backbone_arch="resnet50", 
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
        """
        計算分類損失
        
        Args:
            outputs: 模型輸出
            class_ids: 目標類別 ID
            
        Returns:
            torch.Tensor: 分類損失
        """
        # 從輸出中獲取分類分數
        if isinstance(outputs, dict) and 'class_scores' in outputs:
            class_scores = outputs['class_scores']
        else:
            # 如果輸出不包含分類分數，嘗試使用整個輸出作為分類分數
            class_scores = outputs

        # 將目標轉換為適當的格式
        if isinstance(class_ids, list):
            # 確保列表中的每個元素都是張量，並將它們拼接
            tensor_items = [item for item in class_ids if isinstance(item, torch.Tensor)]
            if tensor_items:
                target = torch.cat(tensor_items).long()
            else:
                # 如果列表中沒有張量，創建一個默認張量
                target = torch.zeros(1).long()
        elif isinstance(class_ids, tuple):
            # 處理元組類型，將其轉換為列表並再次處理
            tensor_items = [item for item in class_ids if isinstance(item, torch.Tensor)]
            if tensor_items:
                target = torch.cat(tensor_items).long()
            else:
                target = torch.zeros(1).long()
        else:
            target = class_ids.long()
        

        device = next(self.parameters()).device
        target = target.to(device)
        
        # 使用交叉熵損失
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # 處理 class_scores 可能是元組的情況
        if isinstance(class_scores, tuple):
            # 使用第一個元素，通常包含分類分數
            class_scores = class_scores[0]
        elif not isinstance(class_scores, torch.Tensor):
            print(f"Warning: class_scores 類型為 {type(class_scores)}，無法計算分類損失")
            return torch.tensor(0.0, device=device)
        
        # 檢查並修正批次大小不匹配的問題
        batch_size = class_scores.size(0)
        
        # 確保 target 至少是 1D 張量
        if target.dim() == 0:
            target = target.unsqueeze(0)
            
        if target.size(0) != batch_size:
            # print(f"Warning: 修正目標批次大小 ({target.size(0)}) 不匹配輸出批次大小 ({batch_size})")
            if target.size(0) > batch_size:
                # 如果目標批次較大，截斷以匹配輸出批次
                target = target[:batch_size]
            else:
                # 如果目標批次較小，使用重複來擴展
                repeats = (batch_size + target.size(0) - 1) // target.size(0)
                target = target.repeat(repeats)[:batch_size]
        
        # 確保 class_scores 形狀正確 (batch_size, num_classes)
        if class_scores.dim() > 2:
            class_scores = class_scores.view(batch_size, -1)
            
        return loss_fn(class_scores, target)
    
    def compute_box_regression_loss(self, outputs, boxes):
        """
        計算邊界框回歸損失
        
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
            # 如果沒有明確的邊界框輸出，使用默認值
            pred_boxes = torch.zeros(1, 4).to(self.device)
            
        # 將目標框轉換為適當的格式
        if isinstance(boxes, list):
            # 處理 BoxList 對象
            valid_boxes = []
            for b in boxes:
                if b is None:
                    continue
                    
                # 檢查是否為 BoxList 對象
                if hasattr(b, 'bbox') and hasattr(b, 'size'):  # BoxList 通常有這些屬性
                    box_tensor = b.bbox  # 獲取底層張量
                    if isinstance(box_tensor, torch.Tensor) and box_tensor.numel() > 0:
                        valid_boxes.append(box_tensor)
                # 檢查是否為張量
                elif isinstance(b, torch.Tensor) and b.numel() > 0:
                    valid_boxes.append(b)
                    
            if not valid_boxes:
                return torch.tensor(0.0, device=pred_boxes.device)
            
            target_boxes = torch.cat(valid_boxes)
        else:
            # 處理單個 BoxList 對象
            if hasattr(boxes, 'bbox') and hasattr(boxes, 'size'):
                target_boxes = boxes.bbox
            else:
                target_boxes = boxes
            
        # 確保張量在同一設備上
        target_boxes = target_boxes.to(pred_boxes.device)
        
        # 如果沒有有效的目標框，返回零損失
        if not isinstance(target_boxes, torch.Tensor) or target_boxes.numel() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
            
        # 確保預測框和目標框的形狀匹配
        if pred_boxes.size(-1) != 4:
            pred_boxes = pred_boxes.view(-1, 4)
        if target_boxes.size(-1) != 4:
            target_boxes = target_boxes.view(-1, 4)
            
        # 調整批次大小以匹配
        if pred_boxes.size(0) != target_boxes.size(0):
            # 如果預測框比目標框多，取前N個
            if pred_boxes.size(0) > target_boxes.size(0):
                pred_boxes = pred_boxes[:target_boxes.size(0)]
            # 如果目標框比預測框多，取前N個
            else:
                target_boxes = target_boxes[:pred_boxes.size(0)]
                
        # 使用 L1 損失，因為它更穩定
        loss_fn = torch.nn.L1Loss()
        try:
            loss = loss_fn(pred_boxes, target_boxes)
        except RuntimeError as e:
            print(f"警告: L1損失計算失敗 - pred_boxes: {pred_boxes.shape}, target_boxes: {target_boxes.shape}")
            return torch.tensor(0.0, device=pred_boxes.device)
            
        return loss
    

    from tqdm import tqdm

    def train_one_epoch(self, train_loader, optimizer, 
                    auxiliary_net=None, device=None, 
                    print_freq=10, scheduler=None, 
                    loss_weights=None, use_lcp_loss=True, 
                    max_batches=None):
        """
        訓練模型一個 epoch
        """
        import torch
        import traceback

        self.train()
        if auxiliary_net is not None:
            auxiliary_net.train()

        if device is None:
            device = self.device
        print( "Start training...")
        print(f"Using device: {device}")
        if loss_weights is None:
            loss_weights = {'cls': 1.0, 'box_reg': 1.0, 'teacher': 0.5}

        giou_loss = GIoULoss()
        loss_history = []

        num_batches = len(train_loader)
        if max_batches is not None:
            num_batches = min(num_batches, max_batches)

        pbar = tqdm(range(num_batches), total=num_batches)

        for batch_idx in pbar:
            try:
                batch_data = train_loader.get_batch(batch_idx)
                # 依據 OS2D dataloader 的 batch 結構
                images, class_images, loc_targets, class_targets, batch_class_ids, class_image_sizes, box_inverse_transform, batch_boxes, batch_img_size = batch_data

                # 移至設備
                images = images.to(device)
                # # class_images 是 list of tensors [B_class, 3, 64, 64]
                # class_images_tensor = torch.stack(class_images).to(device) if isinstance(class_images, list) and isinstance(class_images[0], torch.Tensor) else class_images
                if auxiliary_net is not None:
                    # 取 backbone 輸出 feature map channel
                    feature_maps = self.get_feature_map(images)
                    if isinstance(feature_maps, torch.Tensor):
                        current_channels = feature_maps.shape[1]
                        if auxiliary_net.get_current_channels() != current_channels:
                            auxiliary_net.update_input_channels(current_channels)
                # 處理 boxes 和 class_targets
                boxes = batch_boxes
                class_ids = batch_class_ids

                optimizer.zero_grad()
                print("教師模型預測...")
                # 教師模型預測（可選）
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images, class_images=class_images)
                # 學生模型預測
                print("學生模型預測...")
                # 直接將 class_images (list of tensor) 傳給 model
                outputs = self(images, class_images=class_images)


                # 分類損失
                cls_loss = self.compute_classification_loss(outputs, class_ids)
                # 回歸損失
                box_loss = self.compute_box_regression_loss(outputs, boxes)
                # 教師-學生損失
                if isinstance(outputs, dict) and isinstance(teacher_outputs, dict):
                    teacher_loss = torch.nn.functional.mse_loss(outputs['class_scores'], teacher_outputs['class_scores']) + \
                                torch.nn.functional.mse_loss(outputs['boxes'], teacher_outputs['boxes'])
                elif isinstance(outputs, tuple) and isinstance(teacher_outputs, tuple):
                    teacher_loss = torch.nn.functional.mse_loss(outputs[0], teacher_outputs[0])
                    if len(outputs) > 1 and len(teacher_outputs) > 1:
                        teacher_loss += torch.nn.functional.mse_loss(outputs[1], teacher_outputs[1])
                else:
                    teacher_loss = torch.tensor(0.0, device=device)
                # LCP loss
                lcp_loss = 0
                if use_lcp_loss and auxiliary_net is not None:
                    feature_maps = self.get_feature_map(images)
                    if isinstance(feature_maps, torch.Tensor):
                        aux_outputs = auxiliary_net(feature_maps, boxes, gt_boxes=boxes)
                        if isinstance(aux_outputs, tuple):
                            lcp_loss = torch.nn.functional.mse_loss(aux_outputs[1], torch.cat([b for b in boxes if b.numel() > 0], dim=0))
                        else:
                            lcp_loss = torch.nn.functional.mse_loss(aux_outputs, torch.cat([b for b in boxes if b.numel() > 0], dim=0))
                    elif isinstance(feature_maps, dict):
                        last_feature = list(feature_maps.values())[-1]
                        aux_outputs = auxiliary_net(last_feature, boxes, gt_boxes=boxes)
                        if isinstance(aux_outputs, tuple):
                            lcp_loss = torch.nn.functional.mse_loss(aux_outputs[1], torch.cat([b for b in boxes if b.numel() > 0], dim=0))
                        else:
                            lcp_loss = torch.nn.functional.mse_loss(aux_outputs, torch.cat([b for b in boxes if b.numel() > 0], dim=0))
                    else:
                        last_feature = feature_maps[-1] if isinstance(feature_maps, (tuple, list)) else feature_maps
                        aux_outputs = auxiliary_net(last_feature, boxes, gt_boxes=boxes)
                        if isinstance(aux_outputs, tuple):
                            lcp_loss = torch.nn.functional.mse_loss(aux_outputs[1], torch.cat([b for b in boxes if b.numel() > 0], dim=0))
                        else:
                            lcp_loss = torch.nn.functional.mse_loss(aux_outputs, torch.cat([b for b in boxes if b.numel() > 0], dim=0))
                print("計算損失...")
                loss = loss_weights['cls'] * cls_loss + \
                    loss_weights['box_reg'] * box_loss + \
                    loss_weights['teacher'] * teacher_loss + \
                    0.1 * lcp_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
                if auxiliary_net is not None:
                    torch.nn.utils.clip_grad_norm_(auxiliary_net.parameters(), max_norm=10.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                loss_value = loss.item()
                loss_history.append(loss_value)
                if print_freq > 0 and (batch_idx % print_freq == 0 or batch_idx == num_batches - 1):
                    pbar.set_description(
                        f"Loss: {loss_value:.4f} (cls: {cls_loss.item():.4f}, box: {box_loss.item():.4f}, "
                        f"teacher: {teacher_loss.item():.4f}, lcp: {lcp_loss if isinstance(lcp_loss, float) else lcp_loss.item():.4f})"
                    )
            except Exception as e:
                print(f"批次 {batch_idx} 處理失敗: {e}")
                traceback.print_exc()
                continue

        return loss_history
    
    def finetune(self, train_loader, auxiliary_net, prune_layers, prune_ratio=0.3,
                optimizer=None, device=None, epochs_per_layer=1, print_freq=1, max_batches=3):
        pass

    
    def load_checkpoint(self, checkpoint_path, device=None):
        pass

    
    def save_checkpoint(self, checkpoint_path):
        pass

    def _eval(self, dataloader, iou_thresh=0.5, batch_size=4, cfg=None, criterion=None, print_per_class_results=False):
        """
        使用 os2d.engine.evaluate.evaluate 進行自動化 mAP 評估
        """
        pass