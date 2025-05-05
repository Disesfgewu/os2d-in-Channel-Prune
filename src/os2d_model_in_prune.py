import torch
import torch.nn as nn
import os
import time
import datetime
import logging
import traceback
import copy
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
        
    def forward(self, images, class_images=None, **kwargs):
        """支援接收 class_images 參數的前向傳播"""
        if class_images is not None:
            return super().forward(images, class_images=class_images, **kwargs)
        else:
            return super().forward(images, **kwargs)
    
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
            boxes: 目標邊界框
            
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
            valid_boxes = [b for b in boxes if b is not None and b.numel() > 0]
            if not valid_boxes:
                return torch.tensor(0.0, device=pred_boxes.device)
            target_boxes = torch.cat(valid_boxes)
        else:
            target_boxes = boxes
            
        # 確保張量在同一設備上
        target_boxes = target_boxes.to(pred_boxes.device)
        
        # 如果沒有有效的目標框，返回零損失
        if target_boxes.numel() == 0:
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
    def train_one_epoch(self, train_loader, optimizer, 
                   auxiliary_net=None, device=None, 
                   print_freq=10, scheduler=None, 
                   loss_weights=None, use_lcp_loss=True, 
                   max_batches=None):
        """
        訓練模型一個 epoch
        
        Args:
            train_loader: 資料加載器
            optimizer: 優化器
            auxiliary_net: 輔助網絡，用於通道剪枝 (可選)
            device: 計算設備 (可選，默認為模型設備)
            print_freq: 打印頻率 (可選，默認每10個批次)
            scheduler: 學習率調度器 (可選)
            loss_weights: 損失權重字典 {'cls': 1.0, 'reg': 1.0} (可選)
            use_lcp_loss: 是否使用 LCP 論文中的重建損失 (可選)
            max_batches: 每個 epoch 處理的最大批次數 (可選)
            
        Returns:
            avg_loss: 平均損失值
            loss_components: 損失組件字典 {'cls': cls_loss, 'reg': reg_loss}
        """
        # 1. 初始化訓練環境和統計數據
        device, loss_weights, stats = self._init_training_environment(
            auxiliary_net, device, loss_weights, use_lcp_loss)
        
        # 2. 設置進度條
        progress_bar, num_batches = self._setup_progress_bar(train_loader, max_batches)
        
        # 3. 迭代訓練資料
        for batch_idx, batch_data in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
                
            try:
                # 3.1 解析批次資料
                images, boxes, class_ids = self._parse_batch_data(batch_data, device)
                
                # 3.2 提取類別圖像
                class_images = self._extract_class_images(images, boxes, device)
                
                # 3.3 前向傳播和計算損失
                loss, cls_loss, reg_loss, recon_loss = self._forward_and_compute_loss(
                    images, class_images, boxes, class_ids, 
                    optimizer, auxiliary_net, loss_weights, use_lcp_loss, device)
                
                # 3.4 反向傳播和優化
                self._backward_and_optimize(loss, optimizer, scheduler)
                
                # 3.5 更新統計資料
                stats = self._update_training_stats(
                    stats, loss, cls_loss, reg_loss, recon_loss, use_lcp_loss)
                
                # 3.6 更新進度條
                self._update_progress_bar(
                    progress_bar, batch_idx, print_freq, loss, cls_loss, reg_loss, optimizer)
                
            except Exception as e:
                print(f"⚠️ 處理批次 {batch_idx} 時出錯: {e}")
                traceback.print_exc()
                continue
        
        # 4. 結束訓練並返回結果
        return self._finalize_training(progress_bar, stats, use_lcp_loss)

    def _init_training_environment(self, auxiliary_net, device, loss_weights, use_lcp_loss):
        """初始化訓練環境和統計數據"""
        # 設置模型為訓練模式
        self.train()
        print(f"ℹ️ 主模型設為訓練模式")
        
        # 檢查並設置輔助網絡為訓練模式 (如果存在)
        if auxiliary_net is not None:
            auxiliary_net = auxiliary_net.train()
            print(f"ℹ️ 輔助網絡設為訓練模式，輸入通道數: {auxiliary_net.get_current_channels()}")
        
        # 設置設備
        if device is None:
            device = next(self.parameters()).device
            print(f"ℹ️ 使用模型預設設備: {device}")
        else:
            print(f"ℹ️ 使用指定設備: {device}")
        
        # 設置損失權重
        if loss_weights is None:
            loss_weights = {'cls': 1.0, 'reg': 1.0}
            if use_lcp_loss:
                loss_weights['recon'] = 0.1
        print(f"ℹ️ 損失權重設置: {loss_weights}")
        
        # 初始化統計數據
        stats = {
            'start_time': time.time(),
            'batch_count': 0,
            'total_loss': 0.0,
            'cls_loss_total': 0.0,
            'reg_loss_total': 0.0,
            'recon_loss_total': 0.0 if use_lcp_loss else None
        }
        
        return device, loss_weights, stats

    def _setup_progress_bar(self, train_loader, max_batches):
        """設置進度條"""
        from tqdm import tqdm
        num_batches = len(train_loader)
        if max_batches is not None:
            num_batches = min(max_batches, num_batches)
        progress_bar = tqdm(total=num_batches, desc="Training", unit="batch")
        return progress_bar, num_batches

    def _parse_batch_data(self, batch_data, device):
        """解析批次資料"""
        # 根據批次數據結構解析數據
        if len(batch_data) == 4:  # (images, boxes, labels, class_images)
            images, boxes, class_ids, _ = batch_data  # 忽略原始的 class_images
        elif len(batch_data) == 3:  # (images, boxes, labels)
            images, boxes, class_ids = batch_data
        else:
            raise ValueError(f"不支持的批次數據格式: 長度為 {len(batch_data)}")
        
        # 將數據移至目標設備
        if isinstance(boxes, list):
            boxes = [box.to(device) if isinstance(box, torch.Tensor) else box for box in boxes]
        else:
            boxes = boxes.to(device)
        
        if isinstance(class_ids, list):
            class_ids = [cls_id.to(device) if isinstance(cls_id, torch.Tensor) else cls_id for cls_id in class_ids]
        else:
            class_ids = class_ids.to(device) if isinstance(class_ids, torch.Tensor) else class_ids
        
        # 確保 images 是批次張量
        if isinstance(images, list):
            images = torch.stack(images).to(device)
        else:
            images = images.to(device)
            
        return images, boxes, class_ids

    def _extract_class_images(self, images, boxes, device, class_size=(64, 64)):
        """從圖像和邊界框提取類別圖像"""
        class_images = []
        
        for i in range(images.shape[0]):
            img = images[i]  # 當前圖像
            
            # 獲取當前圖像的邊界框
            current_boxes = boxes[i] if isinstance(boxes, list) else boxes[i] if boxes.dim() > 1 else boxes
            
            # 如果有有效的邊界框
            if current_boxes is not None and current_boxes.numel() > 0:
                class_img = self._crop_class_image_from_box(img, current_boxes, class_size)
            else:
                # 無邊界框 - 使用圖像中心
                class_img = self._crop_class_image_from_center(img, class_size)
            
            class_images.append(class_img)
        
        # 將類別圖像堆疊為批次張量
        class_images = torch.stack(class_images).to(device)
        return class_images

    def _crop_class_image_from_box(self, img, boxes, class_size=(64, 64)):
        """從邊界框中裁剪類別圖像"""
        # 使用第一個框作為類別圖像源
        x1, y1, x2, y2 = boxes[0].cpu().int().tolist()
        
        # 確保座標有效
        h, w = img.shape[1], img.shape[2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            # 提取類別圖像區域
            class_img = img[:, y1:y2, x1:x2].clone()
        else:
            # 無效框區域 - 使用圖像中心
            class_img = self._crop_class_image_from_center(img, class_size)
            return class_img
            
        # 調整尺寸為標準類別圖像尺寸
        class_img = torch.nn.functional.interpolate(
            class_img.unsqueeze(0),
            size=class_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return class_img

    def _crop_class_image_from_center(self, img, class_size=(64, 64)):
        """從圖像中心裁剪類別圖像"""
        h, w = img.shape[1], img.shape[2]
        center_h, center_w = h // 2, w // 2
        size_h, size_w = h // 4, w // 4
        
        y1, y2 = max(0, center_h - size_h), min(h, center_h + size_h)
        x1, x2 = max(0, center_w - size_w), min(w, center_w + size_w)
        
        class_img = img[:, y1:y2, x1:x2].clone()
        class_img = torch.nn.functional.interpolate(
            class_img.unsqueeze(0),
            size=class_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return class_img

    def _forward_and_compute_loss(self, images, class_images, boxes, class_ids, 
                                optimizer, auxiliary_net, loss_weights, use_lcp_loss, device):
        """前向傳播和計算損失"""
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向傳播
        outputs = self(images, class_images=class_images)
        
        # 計算分類損失和回歸損失
        cls_loss = self.compute_classification_loss(outputs, class_ids)
        reg_loss = self.compute_box_regression_loss(outputs, boxes)
        
        # 計算重建損失（如果使用 LCP）
        recon_loss = torch.tensor(0.0, device=device)
        if use_lcp_loss and auxiliary_net is not None:
            recon_loss = self._compute_reconstruction_loss(images, boxes, auxiliary_net, device)
        
        # 計算總損失
        loss = loss_weights['cls'] * cls_loss + loss_weights['reg'] * reg_loss
        if use_lcp_loss:
            # 確保 recon_loss 是標量張量
            if recon_loss.dim() == 0:
                recon_loss = recon_loss.unsqueeze(0)
            loss = loss + loss_weights['recon'] * recon_loss.float()
        
        return loss, cls_loss, reg_loss, recon_loss

    def _compute_reconstruction_loss(self, images, boxes, auxiliary_net, device):
        """
        LCP 論文標準重建損失：學生和教師模型同層 feature map 的 MSE
        """
        # 學生模型 feature map 
        student_feature_maps = self.get_feature_map(images)
        
        # 教師模型 feature map（不需梯度）
        with torch.no_grad():
            teacher_feature_maps = self.teacher_model.get_feature_map(images)
        
        # 計算 MSE 
        criterion = nn.MSELoss()
        recon_loss = criterion(student_feature_maps, teacher_feature_maps)
        
        # 確保損失值有效
        if torch.isnan(recon_loss) or torch.isinf(recon_loss):
            print("⚠️ 重建損失無效，使用備用值")
            recon_loss = torch.tensor(0.1, device=device)
        
        return recon_loss

    def _resize_feature_maps(self, feature_maps, target_size):
        """調整特徵圖大小"""
        # 檢查並確保輸入張量具有正確的維度 [N, C, H, W]
        if feature_maps.dim() < 4:
            # 如果維度不足，添加必要的維度
            if feature_maps.dim() == 2:
                feature_maps = feature_maps.unsqueeze(0).unsqueeze(0)
            elif feature_maps.dim() == 3:
                feature_maps = feature_maps.unsqueeze(0)
        
        # 使用雙線性插值調整大小
        resized_maps = torch.nn.functional.interpolate(
            feature_maps,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        
        return resized_maps

    def _backward_and_optimize(self, loss, optimizer, scheduler=None):
        """反向傳播和優化"""
        loss.backward()
        optimizer.step()
        
        # 更新學習率（如果有調度器）
        if scheduler is not None:
            scheduler.step()

    def _update_training_stats(self, stats, loss, cls_loss, reg_loss, recon_loss, use_lcp_loss):
        """更新訓練統計資料"""
        stats['batch_count'] += 1
        stats['total_loss'] += loss.item()
        stats['cls_loss_total'] += cls_loss.item()
        stats['reg_loss_total'] += reg_loss.item()
        
        if use_lcp_loss:
            stats['recon_loss_total'] += recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss
        
        return stats

    def _update_progress_bar(self, progress_bar, batch_idx, print_freq, loss, cls_loss, reg_loss, optimizer):
        """更新進度條"""
        # 更新進度條
        if batch_idx % print_freq == 0:
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls_loss': f'{cls_loss.item():.4f}', 
                'reg_loss': f'{reg_loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        progress_bar.update(1)

    def _finalize_training(self, progress_bar, stats, use_lcp_loss):
        """結束訓練並返回結果"""
        # 關閉進度條
        progress_bar.close()
        
        # 計算平均損失
        batch_count = stats['batch_count']
        avg_loss = stats['total_loss'] / batch_count if batch_count > 0 else float('inf')
        
        loss_components = {
            'cls': stats['cls_loss_total'] / batch_count if batch_count > 0 else float('inf'),
            'reg': stats['reg_loss_total'] / batch_count if batch_count > 0 else float('inf')
        }
        
        if use_lcp_loss:
            loss_components['recon'] = stats['recon_loss_total'] / batch_count if batch_count > 0 else float('inf')
        
        # 打印訓練統計
        elapsed_time = time.time() - stats['start_time']
        print(f"\n✓ 訓練完成: {batch_count} 批次，平均損失: {avg_loss:.4f}，耗時: {elapsed_time:.2f}秒")
        print(f"  分類損失: {loss_components['cls']:.4f}, 回歸損失: {loss_components['reg']:.4f}")
        
        if use_lcp_loss:
            print(f"  重建損失: {loss_components['recon']:.4f}")
        
        return avg_loss, loss_components
    
    def finetune(self, train_loader, auxiliary_net, prune_layers, prune_ratio=0.3,
                optimizer=None, device=None, epochs_per_layer=1, print_freq=1, max_batches=3):
        """
        逐層剪枝+每層剪枝後微調（LCP論文流程）
        """
        import torch

        if device is None:
            device = self.device if hasattr(self, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if optimizer is None:
            optimizer = torch.optim.Adam(list(self.parameters()) + list(auxiliary_net.parameters()), lr=1e-3)

        self.train()
        auxiliary_net.train()

        for layer_name in prune_layers:
            print(f"\n🔪 剪枝層: {layer_name}")
            # 取一個 batch 作為剪枝依據
            images, boxes, labels, class_images = next(iter(train_loader))
            images = self._normalize_batch_images(images, device=device)
            class_images = self._normalize_batch_images(class_images, device=device, target_size=(64, 64))
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # 剪枝
            success = self.prune_channel(
                layer_name=layer_name,
                prune_ratio=prune_ratio,
                images=images,
                boxes=boxes,
                labels=labels,
                auxiliary_net=auxiliary_net
            )
            assert success, f"剪枝 {layer_name} 失敗"

            # 剪枝後微調
            for epoch in range(epochs_per_layer):
                print(f"  微調 Epoch {epoch+1}/{epochs_per_layer}")
                avg_loss, loss_components = self.train_one_epoch(
                    train_loader=train_loader,
                    optimizer=optimizer,
                    auxiliary_net=auxiliary_net,
                    device=device,
                    print_freq=print_freq,
                    max_batches=max_batches
                )
                print(f"  微調完成，平均損失: {avg_loss:.4f}，損失組件: {loss_components}")
        self.save_checkpoint("finetune_checkpoint.pth")
        print("\n✅ LCP finetune pipeline 完成")
    
    def load_checkpoint(self, checkpoint_path, device=None):
        """
        從檢查點載入學生模型，包括處理剪枝後的結構
        
        Args:
            checkpoint_path: 檢查點文件路徑
            device: 設備 (CPU/GPU)
            
        Returns:
            成功載入返回 True，否則 False
        """
        if device is None:
            device = self.device if hasattr(self, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        try:
            # 載入檢查點
            print(f"📂 開始從 {checkpoint_path} 載入檢查點...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 檢查是否有模型結構資訊
            if 'model_structure' not in checkpoint:
                print(f"⚠️ 檢查點中沒有模型結構資訊，嘗試直接載入...")
                try:
                    # 直接載入，但允許不匹配的鍵值
                    # 只載入學生模型相關的參數
                    student_state_dict = {}
                    for k, v in checkpoint['model_state_dict'].items():
                        if not k.startswith('teacher_model'):
                            student_state_dict[k] = v
                    
                    result = self.load_state_dict(student_state_dict, strict=False)
                    # 如果有缺失或意外的鍵值，輸出警告
                    if result.missing_keys or result.unexpected_keys:
                        print(f"⚠️ 載入時發現匹配問題:")
                        print(f"   缺失鍵值: {len(result.missing_keys)} 個")
                        print(f"   多餘鍵值: {len(result.unexpected_keys)} 個")
                    print(f"⚠️ 模型已載入但可能存在參數不匹配問題")
                    return True
                except Exception as e:
                    print(f"❌ 直接載入失敗: {e}")
                    return False
                
            # 根據檢查點中的結構重構模型
            print("🔄 根據保存的結構重建模型...")
            success = self._reconstruct_model_from_structure(checkpoint['model_structure'])
            if not success:
                print("❌ 模型重構失敗")
                return False
            
            # 現在模型結構應該匹配，可以載入權重
            print("⏳ 載入權重...")
            
            # 確保只載入學生模型的參數
            student_state_dict = {}
            for k, v in checkpoint['model_state_dict'].items():
                # 過濾掉教師模型的參數
                if not k.startswith('teacher_model'):
                    student_state_dict[k] = v
                    
            # 載入過濾後的狀態字典
            self.load_state_dict(student_state_dict, strict=False)
            
            # 如果存在教師模型，需要重新初始化
            if hasattr(self, 'teacher_model'):
                print("🔄 正在重新初始化教師模型...")
                # 使用當前模型（學生模型）的狀態創建新的教師模型
                self.teacher_model = copy.deepcopy(self)
                self.teacher_model.eval()
                for p in self.teacher_model.parameters():
                    p.requires_grad = False
            
            # 如果需要，也可以載入輔助網絡
            if hasattr(self, 'auxiliary_net') and 'auxiliary_net_state_dict' in checkpoint and checkpoint['auxiliary_net_state_dict'] is not None:
                self.auxiliary_net.load_state_dict(checkpoint['auxiliary_net_state_dict'])
                print("✓ 已載入輔助網絡")
                
            # 計算並顯示學生模型的參數量
            student_params = sum(p.numel() for name, p in self.named_parameters() 
                            if not name.startswith('teacher_model'))
                
            print(f"✅ 成功從 {checkpoint_path} 載入模型")
            print(f"載入後學生模型參數量: {student_params:,}")
            return True
            
        except Exception as e:
            print(f"❌ 載入模型時出錯: {e}")
            traceback.print_exc()
            return False

    def _reconstruct_model_from_structure(self, structure):
        """
        根據保存的結構資訊重建模型
        
        Args:
            structure: 模型結構字典
        
        Returns:
            重建是否成功
        """
        # 首先找出所有需要調整的卷積層
        conv_layers_to_adjust = {}
        
        for name, config in structure.items():
            if 'type' not in config or config['type'] != 'Conv2d':
                continue
                
            # 檢查這個層是否存在於當前模型中
            try:
                module = self._get_module_by_name(name)
                if module is not None and isinstance(module, nn.Conv2d):
                    # 檢查通道數是否不同
                    if module.out_channels != config['out_channels']:
                        conv_layers_to_adjust[name] = config
            except (AttributeError, IndexError):
                print(f"⚠️ 無法找到層 {name}，跳過重建")
                continue
        
        # 輸出所有需要重建的層
        print(f"📊 需要調整的卷積層數量: {len(conv_layers_to_adjust)}")
        
        # 按照名稱排序重建層，確保按正確順序重建
        for name in sorted(conv_layers_to_adjust.keys()):
            config = conv_layers_to_adjust[name]
            print(f"  調整層 {name}: 輸出通道從 {self._get_module_by_name(name).out_channels} 到 {config['out_channels']}")
            
            # 獲取backbone相對路徑
            if "backbone." in name:
                layer_name = name.replace("backbone.", "")
            elif "net_feature_maps." in name:
                layer_name = name.replace("net_feature_maps.", "")
            else:
                layer_name = name
                
            # 使用set_layer_out_channels方法調整通道數
            success = self.set_layer_out_channels(layer_name, config['out_channels'])
            if not success:
                print(f"❌ 調整層 {layer_name} 失敗")
                return False
        
        print(f"✅ 模型結構重建完成")
        return True

    def _get_module_by_name(self, name):
        """通過名稱獲取模組"""
        parts = name.split('.')
        module = self
        
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
                
        return module
    
    def save_checkpoint(self, checkpoint_path):
        """保存模型檢查點，確保與 OS2D 框架完全相容"""
        import traceback
        import logging
        import os
        
        # 創建臨時logger，類似於父類中的logger
        temp_logger = logging.getLogger("OS2D.save_checkpoint")
        
        # 定義保存路徑
        pruned_path = checkpoint_path.replace('.pth', '_pruned.pth')  # 剪枝版本
        os2d_path = checkpoint_path  # OS2D 兼容版本
        
        # 收集剪枝後的模型結構資訊
        model_structure = self._get_model_structure(exclude_teacher=True)
        
        # 準備檢查點字典 - 只包含學生模型的參數
        student_state_dict = {
            k: v for k, v in self.state_dict().items() 
            if not k.startswith('teacher_model')
        }
        
        # 檢查是否有 optimizer 屬性
        optimizer_state = None
        if hasattr(self, 'optimizer'):
            optimizer_state = self.optimizer.state_dict()
        
        # 使用與原始 OS2D 完全相同的檢查點結構
        os2d_checkpoint = {
            'net': student_state_dict,  # OS2D init_model_from_file 首先查找的鍵
            'optimizer': optimizer_state,
            'scheduler': None,
            'iteration': 0,
            'epoch': self.epoch if hasattr(self, 'epoch') else 0,
            'loss': 0.0,  # OS2D _load_network 方法可能需要這個
            'config': {
                'model': {
                    'backbone': {'arch': "resnet50"},
                    'merge_branch_parameters': True,
                    'use_group_norm': False,
                    'use_inverse_geom_model': False,
                    'use_simplified_affine_model': True
                },
            },
            'best_score': 0.0,
            
            # 保留我們自己的額外資訊
            'model_state_dict': student_state_dict,
            'optimizer_state_dict': optimizer_state,
            'auxiliary_net_state_dict': self.auxiliary_net.state_dict() if hasattr(self, 'auxiliary_net') else None,
            'model_structure': model_structure,
            'backbone_arch': self.backbone_arch if hasattr(self, 'backbone_arch') else "resnet50"
        }
        
        # 1. 保存剪枝版本 (給我們自己用)
        pruned_checkpoint = {
            'model_state_dict': student_state_dict,
            'model_structure': model_structure,
            'optimizer_state_dict': optimizer_state,
            'auxiliary_net_state_dict': self.auxiliary_net.state_dict() if hasattr(self, 'auxiliary_net') else None,
            'backbone_arch': self.backbone_arch if hasattr(self, 'backbone_arch') else "resnet50",
            'epoch': self.epoch if hasattr(self, 'epoch') else 0
        }
        
        try:
            # 保存兩個檢查點
            torch.save(os2d_checkpoint, os2d_path)
            torch.save(pruned_checkpoint, pruned_path)
            
            print(f"\n✅ 檢查點已保存:")
            print(f"  - OS2D 相容檢查點: {os2d_path}")
            print(f"  - 剪枝模型檢查點: {pruned_path}")
            
            # 計算參數量
            student_params = sum(p.numel() for name, p in self.named_parameters() 
                            if not name.startswith('teacher_model'))
            print(f"  - 學生模型參數量: {student_params:,}")
            
            # 測試能否加載 (嚴格的測試)
            print("\n🧪 測試檢查點相容性...")
            os2d_compat_result = False
            pruned_compat_result = False
            
            print("\n1. 測試OS2D框架相容性 (使用父類 Os2dModel):")
            try:
                from os2d.modeling.model import Os2dModel
                # 創建原始 OS2D 模型實例
                os2d_model = Os2dModel(logger=temp_logger, is_cuda=self.is_cuda)
                # 使用父類的 init_model_from_file 方法測試
                optimizer_result = os2d_model.init_model_from_file(os2d_path)
                print(f"✅ OS2D 框架相容性測試: ✓ 通過")
                os2d_compat_result = True
            except Exception as e:
                print(f"❌ OS2D 框架相容性測試: ✗ 失敗")
                print(f"   錯誤原因: {e}")
            
            print("\n2. 測試剪枝模型載入 (使用 Os2dModelInPrune):")
            try:
                # 自己的模型載入測試
                pruned_model = type(self)(pretrained_path=None, is_cuda=self.is_cuda)
                success = pruned_model.load_checkpoint(pruned_path)
                if success:
                    print(f"✅ 剪枝模型載入測試: ✓ 通過")
                    pruned_compat_result = True
                else:
                    print(f"❌ 剪枝模型載入測試: ✗ 失敗")
            except Exception as e:
                print(f"❌ 剪枝模型載入測試: ✗ 失敗")
                print(f"   錯誤原因: {e}")
            
            print("\n3. 測試使用本類的 init_model_from_file:")
            try:
                temp_model2 = type(self)(pretrained_path=None, is_cuda=self.is_cuda)
                optimizer_result = temp_model2.init_model_from_file(os2d_path)
                print(f"✅ init_model_from_file 測試: ✓ 通過")
                if optimizer_result is not None:
                    print("   優化器狀態也成功載入")
            except Exception as e:
                print(f"❌ init_model_from_file 測試: ✗ 失敗")
                print(f"   錯誤原因: {e}")
                    
            print("\n===== 相容性測試摘要 =====")
            print(f"OS2D 框架相容性: {'✅ 通過' if os2d_compat_result else '❌ 失敗'}")
            print(f"剪枝模型載入測試: {'✅ 通過' if pruned_compat_result else '❌ 失敗'}")
            print("==========================")
            
            return True
        except Exception as e:
            print(f"❌ 保存檢查點時發生錯誤: {str(e)}")
            traceback.print_exc()
            return False

    def _get_model_structure(self, exclude_teacher=True):
        """獲取詳細的模型結構資訊（可選排除教師模型）"""
        structure = {}
        
        # 記錄所有層的結構資訊
        for name, module in self.named_modules():
            # 如果設定排除教師模型，則跳過教師模型相關的層
            if exclude_teacher and name.startswith('teacher_model'):
                continue
                
            if isinstance(module, nn.Conv2d):
                structure[name] = {
                    'type': 'Conv2d',
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                    'dilation': module.dilation,
                    'groups': module.groups,
                    'bias': module.bias is not None
                }
            elif isinstance(module, nn.BatchNorm2d):
                structure[name] = {
                    'type': 'BatchNorm2d',
                    'num_features': module.num_features,
                    'eps': module.eps,
                    'momentum': module.momentum,
                    'affine': module.affine,
                    'track_running_stats': module.track_running_stats
                }
            elif isinstance(module, nn.Linear):
                structure[name] = {
                    'type': 'Linear',
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'bias': module.bias is not None
                }
        
        return structure    