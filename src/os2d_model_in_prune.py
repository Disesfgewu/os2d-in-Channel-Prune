import torch
import torch.nn as nn
import logging
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
                 pretrained_path=None, **kwargs):
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
    
    def _handle_downsample_connection(self, layer_name, keep_indices):
        """處理 downsample 連接"""
        print(f"\n🔍 處理 downsample 連接: {layer_name}")
        
        # 解析層名稱
        parts = layer_name.split('.')
        if len(parts) < 3:
            print(f"⚠️ 無效的層名稱格式: {layer_name}")
            return False
            
        layer_str, block_idx = parts[0], int(parts[1])
        
        # 獲取當前 block 和 downsample
        layer = getattr(self.backbone, layer_str)
        current_block = layer[block_idx]
        
        if not hasattr(current_block, 'downsample') or current_block.downsample is None:
            return True
            
        downsample = current_block.downsample
        old_conv = downsample[0]  # downsample 的第一層是 conv
        old_bn = downsample[1]    # 第二層是 bn
        
        # 更新 downsample 的 conv 層
        new_conv = nn.Conv2d(
            old_conv.in_channels,
            len(keep_indices),  # 新的輸出通道數
            old_conv.kernel_size,
            old_conv.stride,
            old_conv.padding,
            old_conv.dilation,
            old_conv.groups,
            bias=old_conv.bias is not None
        ).to(old_conv.weight.device)
        
        # 更新權重
        new_conv.weight.data = old_conv.weight.data[keep_indices].clone()
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data[keep_indices].clone()
        
        # 更新 downsample 的 bn 層
        new_bn = nn.BatchNorm2d(len(keep_indices)).to(old_bn.weight.device)
        new_bn.weight.data = old_bn.weight.data[keep_indices].clone()
        new_bn.bias.data = old_bn.bias.data[keep_indices].clone()
        new_bn.running_mean = old_bn.running_mean[keep_indices].clone()
        new_bn.running_var = old_bn.running_var[keep_indices].clone()
        
        # 創建新的 downsample sequential
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
        
    
    def get_feature_map(self, x):
        """獲取特徵圖"""
        feature_maps = self.backbone(x)
        return feature_maps
    
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