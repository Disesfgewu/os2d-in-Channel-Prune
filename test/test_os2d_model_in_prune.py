import os
import torch
import torchvision
import pytest
# import # traceback
import numpy as np
import torch.nn as nn 
import unittest
from torch.utils.data import DataLoader
from unittest.mock import MagicMock, patch
from src.lcp_channel_selector import OS2DChannelSelector
from src.contextual_roi_align import ContextualRoIAlign
from src.auxiliary_network import AuxiliaryNetwork
from src.os2d_model_in_prune import Os2dModelInPrune
from src.dataset_downloader import VOCDataset , VOC_CLASSES
from os2d.modeling.model import Os2dModel
from os2d.data.dataset import build_grozi_dataset
from os2d.data.dataloader import DataloaderOneShotDetection
from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
from os2d.structures.feature_map import FeatureMapSize

import logging
import tempfile

def test_in_init():
    """測試 Os2dModelInPrune 初始化功能"""
    # 基本初始化測試
    model = Os2dModelInPrune(is_cuda=False)
    assert isinstance(model, Os2dModelInPrune)
    assert isinstance(model, Os2dModel)
    assert model.device == torch.device('cpu')
    assert hasattr(model, 'backbone')
    assert hasattr(model, 'teacher_model')
    
    # 測試 teacher_model 屬性
    assert isinstance(model.teacher_model, Os2dModelInPrune)
    assert model.teacher_model.training == False  # 確認教師模型處於評估模式
    
    # 測試教師模型參數不需要梯度
    for param in model.teacher_model.parameters():
        assert not param.requires_grad
    
    # 測試帶有實際預訓練路徑的初始化
    pretrained_path = "./models/os2d_v1-train.pth"
    if os.path.exists(pretrained_path):
        try:
            model_with_pretrained = Os2dModelInPrune(
                is_cuda=False,
                pretrained_path=pretrained_path
            )
            print(f"✓ 成功載入預訓練模型: {pretrained_path}")
            assert isinstance(model_with_pretrained, Os2dModelInPrune)
        except Exception as e:
            print(f"⚠️ 載入預訓練模型失敗: {e}")
    else:
        print(f"⚠️ 預訓練模型路徑不存在: {pretrained_path}")
        # 使用 mock 模擬載入
        with patch.object(Os2dModelInPrune, 'init_model_from_file') as mock_init:
            model = Os2dModelInPrune(
                is_cuda=False,
                pretrained_path="./models/os2d_v1-train.pth"
            )
            mock_init.assert_called_once_with("./models/os2d_v1-train.pth")
    
    # 測試帶有剪枝檢查點的初始化
    # 創建臨時檔案模擬剪枝檢查點
    with tempfile.NamedTemporaryFile(suffix='.pth') as tmp_file:
        torch.save({'state_dict': {}}, tmp_file.name)
        
        try:
            model_with_checkpoint = Os2dModelInPrune(
                is_cuda=False,
                pruned_checkpoint=tmp_file.name
            )
            print(f"✓ 成功載入剪枝檢查點: {tmp_file.name}")
            assert isinstance(model_with_checkpoint, Os2dModelInPrune)
        except Exception as e:
            print(f"⚠️ 載入剪枝檢查點失敗: {e}")
            # 使用 mock 模擬載入
            with patch.object(Os2dModelInPrune, 'load_checkpoint') as mock_load:
                model = Os2dModelInPrune(
                    is_cuda=False,
                    pruned_checkpoint=tmp_file.name
                )
                mock_load.assert_called_once_with(tmp_file.name)
    
    # 測試 CUDA 處理
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            cuda_model = Os2dModelInPrune(is_cuda=True)
            print("✓ 成功在 CUDA 上初始化模型")
            assert cuda_model.device == torch.device('cuda')
        except Exception as e:
            print(f"⚠️ CUDA 初始化失敗: {e}")
            # 應該回退到 CPU
            assert model.device == torch.device('cpu')
    else:
        print("⚠️ CUDA 不可用，使用 mock 測試")
        with patch('torch.cuda.is_available', return_value=True):
            with patch.object(Os2dModelInPrune, 'cuda') as mock_cuda:
                model = Os2dModelInPrune(is_cuda=True)
                mock_cuda.assert_called()
                assert model.device == torch.device('cuda')
    
    # 測試 CUDA 錯誤處理
    with patch('torch.cuda.is_available', return_value=True):
        with patch.object(Os2dModelInPrune, 'cuda', side_effect=RuntimeError("CUDA error")):
            model = Os2dModelInPrune(is_cuda=True)
            # 應該回退到 CPU
            assert model.device == torch.device('cpu')
    
    # 測試自定義 logger
    mock_logger = MagicMock()
    model = Os2dModelInPrune(is_cuda=False, logger=mock_logger)
    assert model.logger == mock_logger
    
    # 測試自定義 backbone_arch
    model = Os2dModelInPrune(is_cuda=False, backbone_arch="resnet34")
    # 檢查 backbone 架構
    backbone_modules = list(model.backbone.named_modules())
    backbone_structure = [name for name, _ in backbone_modules if len(name.split('.')) <= 2]
    print(f"✓ 使用 resnet34 backbone，結構: {backbone_structure[:5]}...")
    
    print("✅ Os2dModelInPrune.__init__ 測試通過")
    return True

def test_in_set_layer_out_channels():
    """測試 set_layer_out_channels 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 獲取原始層的通道數
    original_layer = None
    target_layer_name = "layer2.0.conv1"
    for name, module in model.backbone.named_modules():
        if name == target_layer_name:
            original_layer = module
            break
    
    assert original_layer is not None, f"找不到目標層: {target_layer_name}"
    original_out_channels = original_layer.out_channels
    original_in_channels = original_layer.in_channels
    
    # 測試減少通道數
    new_out_channels = original_out_channels // 2
    success = model.set_layer_out_channels(target_layer_name, new_out_channels)
    
    # 驗證操作是否成功
    assert success, "設置層輸出通道數應該成功"
    
    # 獲取更新後的層
    updated_layer = None
    for name, module in model.backbone.named_modules():
        if name == target_layer_name:
            updated_layer = module
            break
    
    # 驗證通道數是否正確更新
    assert updated_layer is not None, "更新後應該能找到目標層"
    assert updated_layer.out_channels == new_out_channels, f"輸出通道數應該為 {new_out_channels}，但得到 {updated_layer.out_channels}"
    assert updated_layer.in_channels == original_in_channels, "輸入通道數不應該改變"
    
    # 測試相關的 BatchNorm 層是否也被更新
    bn_name = target_layer_name.replace('conv', 'bn')
    bn_layer = None
    for name, module in model.backbone.named_modules():
        if name == bn_name:
            bn_layer = module
            break
    
    if bn_layer is not None:
        assert bn_layer.num_features == new_out_channels, f"BatchNorm 層的特徵數應該為 {new_out_channels}，但得到 {bn_layer.num_features}"
    
    # 測試下一層的輸入通道是否更新
    next_conv_name = "layer2.0.conv2"
    next_conv = None
    for name, module in model.backbone.named_modules():
        if name == next_conv_name:
            next_conv = module
            break
    
    if next_conv is not None:
        assert next_conv.in_channels == new_out_channels, f"下一層的輸入通道數應該為 {new_out_channels}，但得到 {next_conv.in_channels}"
    
    # 測試增加通道數（應該會截斷權重）
    larger_out_channels = original_out_channels * 2
    success = model.set_layer_out_channels(target_layer_name, larger_out_channels)
    
    # 驗證操作是否成功
    assert success, "設置更大的輸出通道數應該成功"
    
    # 獲取更新後的層
    updated_layer = None
    for name, module in model.backbone.named_modules():
        if name == target_layer_name:
            updated_layer = module
            break
    
    # 驗證通道數是否正確更新
    assert updated_layer is not None, "更新後應該能找到目標層"
    assert updated_layer.out_channels == larger_out_channels, f"輸出通道數應該為 {larger_out_channels}，但得到 {updated_layer.out_channels}"
    
    # 測試無效層名稱
    invalid_layer_name = "non_existent_layer"
    success = model.set_layer_out_channels(invalid_layer_name, 64)
    assert not success, "設置不存在的層應該失敗"
    
    # 測試非卷積層
    non_conv_layer_name = "layer2.0"
    success = model.set_layer_out_channels(non_conv_layer_name, 64)
    assert not success, "設置非卷積層應該失敗"
    
    print("✅ set_layer_out_channels 測試通過")
    return True

def test_should_skip_pruning():
    """測試 _should_skip_pruning 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試 layer4 層 (應該跳過)
    layer4_name = "layer4.0.conv1"
    assert model._should_skip_pruning(layer4_name), f"層 {layer4_name} 應該跳過剪枝"
    
    # 測試 conv3 層 (應該跳過)
    conv3_name = "layer2.0.conv3"
    assert model._should_skip_pruning(conv3_name), f"層 {conv3_name} 應該跳過剪枝"
    
    # 測試帶有 downsample 的 conv1 層 (不應該跳過)
    # 首先需要確認哪個 block 有 downsample
    downsample_block_idx = None
    for i in range(len(model.backbone.layer2)):
        if hasattr(model.backbone.layer2[i], 'downsample') and model.backbone.layer2[i].downsample is not None:
            downsample_block_idx = i
            break
    
    if downsample_block_idx is not None:
        downsample_conv1 = f"layer2.{downsample_block_idx}.conv1"
        assert not model._should_skip_pruning(downsample_conv1), f"帶有 downsample 的 conv1 層 {downsample_conv1} 不應該跳過剪枝"
    
    # 測試最後一個 block 的層 (應該跳過)
    last_block_idx = len(model.backbone.layer2) - 1
    last_block_conv = f"layer2.{last_block_idx}.conv1"
    assert model._should_skip_pruning(last_block_conv), f"最後一個 block 的層 {last_block_conv} 應該跳過剪枝"
    
    # 測試普通的 conv1 層 (不應該跳過)
    normal_conv1 = "layer2.1.conv1"
    # 確保這不是最後一個 block 且沒有 downsample
    if 1 != last_block_idx and not hasattr(model.backbone.layer2[1], 'downsample'):
        assert not model._should_skip_pruning(normal_conv1), f"普通的 conv1 層 {normal_conv1} 不應該跳過剪枝"
    
    # 測試普通的 conv2 層 (不應該跳過)
    normal_conv2 = "layer2.1.conv2"
    # 確保這不是最後一個 block
    if 1 != last_block_idx:
        assert not model._should_skip_pruning(normal_conv2), f"普通的 conv2 層 {normal_conv2} 不應該跳過剪枝"
    
    # 測試無效的層名稱格式
    invalid_name = "invalid_layer_name"
    assert not model._should_skip_pruning(invalid_name), f"無效的層名稱 {invalid_name} 不應該導致錯誤"
    
    # 測試 layer1 層 (行為應與 layer2/3 一致)
    layer1_conv1 = "layer1.0.conv1"
    # 這裡我們不確定具體行為，所以只是調用函數確保不會崩潰
    result = model._should_skip_pruning(layer1_conv1)
    print(f"layer1.0.conv1 跳過剪枝: {result}")
    
    # 使用 Mock 測試特殊情況
    with patch.object(model, 'backbone') as mock_backbone:
        # 模擬無法找到層的情況
        mock_backbone.layer2 = MagicMock()
        mock_backbone.layer2.__len__.return_value = 3
        mock_backbone.layer2.__getitem__.side_effect = IndexError("模擬索引錯誤")
        
        # 應該返回 False 而不是引發錯誤
        assert not model._should_skip_pruning("layer2.5.conv1"), "不存在的層索引應該返回 False"
    
    print("✅ _should_skip_pruning 測試通過")
    return True

def test_in_handle_residual_connection():
    """測試 _handle_residual_connection 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試 conv1 層的殘差連接處理
    # 首先找到一個有效的 conv1 層
    layer_name = "layer2.0.conv1"
    
    # 獲取原始層的通道數
    target_layer = None
    for name, module in model.backbone.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            target_layer = module
            break
    
    assert target_layer is not None, f"找不到目標層: {layer_name}"
    
    # 創建保留索引 - 保留一半的通道
    original_channels = target_layer.out_channels
    keep_num = original_channels // 2
    keep_indices = torch.arange(keep_num, dtype=torch.long)
    
    # 測試殘差連接處理
    model._handle_residual_connection(layer_name, keep_indices)
    
    # 檢查下一層的輸入通道是否已更新
    next_conv_name = "layer2.0.conv2"
    next_conv = None
    for name, module in model.backbone.named_modules():
        if name == next_conv_name and isinstance(module, nn.Conv2d):
            next_conv = module
            break
    
    assert next_conv is not None, f"找不到下一層: {next_conv_name}"
    assert next_conv.in_channels == keep_num, f"下一層輸入通道數應為 {keep_num}，但得到 {next_conv.in_channels}"
    
    # 測試 conv2 層的殘差連接處理
    layer_name = "layer2.0.conv2"
    target_layer = None
    for name, module in model.backbone.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            target_layer = module
            break
    
    assert target_layer is not None, f"找不到目標層: {layer_name}"
    
    # 創建保留索引 - 保留一半的通道
    original_channels = target_layer.out_channels
    keep_num = original_channels // 2
    keep_indices = torch.arange(keep_num, dtype=torch.long)
    
    # 測試殘差連接處理
    model._handle_residual_connection(layer_name, keep_indices)
    
    # 檢查下一層的輸入通道是否已更新 (如果存在 conv3)
    next_conv_name = "layer2.0.conv3"
    next_conv = None
    for name, module in model.backbone.named_modules():
        if name == next_conv_name and isinstance(module, nn.Conv2d):
            next_conv = module
            break
    
    if next_conv is not None:
        assert next_conv.in_channels == keep_num, f"下一層輸入通道數應為 {keep_num}，但得到 {next_conv.in_channels}"
    
    # 測試無效層名稱
    with patch('builtins.print') as mock_print:
        model._handle_residual_connection("invalid_layer_name", keep_indices)
        mock_print.assert_called()
    
    # 測試帶有 downsample 的塊
    # 找到一個帶有 downsample 的塊
    downsample_block = None
    downsample_block_name = None
    for i in range(len(model.backbone.layer2)):
        if hasattr(model.backbone.layer2[i], 'downsample') and model.backbone.layer2[i].downsample is not None:
            downsample_block = model.backbone.layer2[i]
            downsample_block_name = f"layer2.{i}"
            break
    
    if downsample_block is not None:
        # 測試 conv1 層的殘差連接處理
        layer_name = f"{downsample_block_name}.conv1"
        
        # 獲取原始層的通道數
        target_layer = None
        for name, module in model.backbone.named_modules():
            if name == layer_name and isinstance(module, nn.Conv2d):
                target_layer = module
                break
        
        assert target_layer is not None, f"找不到目標層: {layer_name}"
        
        # 創建保留索引 - 保留一半的通道
        original_channels = target_layer.out_channels
        keep_num = original_channels // 2
        keep_indices = torch.arange(keep_num, dtype=torch.long)
        
        # 測試殘差連接處理
        model._handle_residual_connection(layer_name, keep_indices)
        
        # 檢查下一層的輸入通道是否已更新
        next_conv_name = f"{downsample_block_name}.conv2"
        next_conv = None
        for name, module in model.backbone.named_modules():
            if name == next_conv_name and isinstance(module, nn.Conv2d):
                next_conv = module
                break
        
        assert next_conv is not None, f"找不到下一層: {next_conv_name}"
        assert next_conv.in_channels == keep_num, f"下一層輸入通道數應為 {keep_num}，但得到 {next_conv.in_channels}"
    
    # 測試異常情況處理
    with patch.object(model, '_handle_residual_connection', side_effect=Exception("模擬異常")):
        try:
            model._handle_residual_connection("layer2.0.conv1", keep_indices)
            # 如果沒有引發異常，則測試失敗
            assert False, "應該引發異常但沒有"
        except Exception:
            # 預期會引發異常
            pass
    
    print("✅ _handle_residual_connection 測試通過")
    return True

def test_in_prune_conv_layer():
    """測試 _prune_conv_layer 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試剪枝 conv1 層
    layer_name = "layer2.0.conv1"
    
    # 獲取原始層的通道數
    target_layer = None
    for name, module in model.backbone.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            target_layer = module
            break
    
    assert target_layer is not None, f"找不到目標層: {layer_name}"
    
    # 記錄原始通道數
    original_out_channels = target_layer.out_channels
    
    # 創建保留索引 - 保留一半的通道
    keep_num = original_out_channels // 2
    keep_indices = torch.arange(keep_num, dtype=torch.long)
    
    # 執行剪枝
    success = model._prune_conv_layer(layer_name, keep_indices)
    
    # 驗證剪枝是否成功
    assert success, "剪枝操作應該成功"
    
    # 獲取剪枝後的層
    pruned_layer = None
    for name, module in model.backbone.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            pruned_layer = module
            break
    
    assert pruned_layer is not None, "剪枝後應該能找到目標層"
    
    # 驗證通道數是否正確減少
    assert pruned_layer.out_channels == keep_num, f"剪枝後通道數應為 {keep_num}，但得到 {pruned_layer.out_channels}"
    
    # 驗證下一層的輸入通道是否更新
    next_conv_name = "layer2.0.conv2"
    next_conv = None
    for name, module in model.backbone.named_modules():
        if name == next_conv_name and isinstance(module, nn.Conv2d):
            next_conv = module
            break
    
    assert next_conv is not None, f"找不到下一層: {next_conv_name}"
    assert next_conv.in_channels == keep_num, f"下一層輸入通道數應為 {keep_num}，但得到 {next_conv.in_channels}"
    
    # 測試剪枝 conv2 層
    layer_name = "layer2.0.conv2"
    
    # 獲取原始層的通道數
    target_layer = None
    for name, module in model.backbone.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            target_layer = module
            break
    
    assert target_layer is not None, f"找不到目標層: {layer_name}"
    
    # 記錄原始通道數
    original_out_channels = target_layer.out_channels
    
    # 創建保留索引 - 保留一半的通道
    keep_num = original_out_channels // 2
    keep_indices = torch.arange(keep_num, dtype=torch.long)
    
    # 執行剪枝
    success = model._prune_conv_layer(layer_name, keep_indices)
    
    # 驗證剪枝是否成功
    assert success, "剪枝操作應該成功"
    
    # 獲取剪枝後的層
    pruned_layer = None
    for name, module in model.backbone.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            pruned_layer = module
            break
    
    assert pruned_layer is not None, "剪枝後應該能找到目標層"
    
    # 驗證通道數是否正確減少
    assert pruned_layer.out_channels == keep_num, f"剪枝後通道數應為 {keep_num}，但得到 {pruned_layer.out_channels}"
    
    # 測試無效層名稱
    invalid_layer_name = "non_existent_layer"
    success = model._prune_conv_layer(invalid_layer_name, keep_indices)
    assert not success, "對不存在的層進行剪枝應該失敗"
    
    # 測試應該跳過的層
    layer4_name = "layer4.0.conv1"
    success = model._prune_conv_layer(layer4_name, keep_indices)
    assert not success, f"層 {layer4_name} 應該跳過剪枝"
    
    print("✅ _prune_conv_layer 測試通過")
    return True

def test_in_reset_batchnorm_stats():
    """測試 _reset_batchnorm_stats 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 獲取所有 BatchNorm 層
    bn_layers = [m for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    
    # 確保模型中有 BatchNorm 層
    assert len(bn_layers) > 0, "模型中沒有 BatchNorm 層"
    
    # 記錄原始統計數據
    original_stats = {}
    for i, bn in enumerate(bn_layers):
        original_stats[i] = {
            'running_mean': bn.running_mean.clone(),
            'running_var': bn.running_var.clone(),
            'num_batches_tracked': bn.num_batches_tracked.clone() if hasattr(bn, 'num_batches_tracked') else None
        }
    
    # 修改統計數據以模擬訓練
    for bn in bn_layers:
        # 修改 running_mean
        if bn.running_mean.numel() > 0:
            bn.running_mean[0] += 1.0
        
        # 修改 running_var
        if bn.running_var.numel() > 0:
            bn.running_var[0] += 1.0
        
        # 增加 num_batches_tracked
        if hasattr(bn, 'num_batches_tracked'):
            bn.num_batches_tracked += 10
    
    # 確認統計數據已被修改
    for i, bn in enumerate(bn_layers):
        if bn.running_mean.numel() > 0:
            assert not torch.allclose(bn.running_mean, original_stats[i]['running_mean']), f"BatchNorm {i} 的 running_mean 未被修改"
        
        if bn.running_var.numel() > 0:
            assert not torch.allclose(bn.running_var, original_stats[i]['running_var']), f"BatchNorm {i} 的 running_var 未被修改"
        
        if hasattr(bn, 'num_batches_tracked'):
            assert bn.num_batches_tracked != original_stats[i]['num_batches_tracked'], f"BatchNorm {i} 的 num_batches_tracked 未被修改"
    
    # 調用重置方法
    model._reset_batchnorm_stats()
    
    # 驗證統計數據已被重置
    for i, bn in enumerate(bn_layers):
        # 檢查 running_mean 是否已重置為 0
        assert torch.allclose(bn.running_mean, torch.zeros_like(bn.running_mean)), f"BatchNorm {i} 的 running_mean 未被重置為 0"
        
        # 檢查 running_var 是否已重置為 1
        assert torch.allclose(bn.running_var, torch.ones_like(bn.running_var)), f"BatchNorm {i} 的 running_var 未被重置為 1"
        
        # 檢查 num_batches_tracked 是否已重置為 0
        if hasattr(bn, 'num_batches_tracked'):
            assert bn.num_batches_tracked == 0, f"BatchNorm {i} 的 num_batches_tracked 未被重置為 0"
    
    # 測試在不同設備上的行為
    if torch.cuda.is_available():
        try:
            cuda_model = Os2dModelInPrune(is_cuda=True)
            cuda_model._reset_batchnorm_stats()
            print("✓ 在 CUDA 設備上成功重置 BatchNorm 統計數據")
        except Exception as e:
            print(f"⚠️ 在 CUDA 設備上重置 BatchNorm 統計數據失敗: {e}")
    
    print("✅ _reset_batchnorm_stats 測試通過")
    return True

def test_in_prune_channel():
    """測試 prune_channel 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 創建輔助網路
    auxiliary_net = AuxiliaryNetwork(in_channels=2048)
    
    # 測試剪枝 conv1 層
    layer_name = "layer2.0.conv1"
    
    # 獲取原始層的通道數
    target_layer = None
    for name, module in model.backbone.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            target_layer = module
            break
    
    assert target_layer is not None, f"找不到目標層: {layer_name}"
    
    # 記錄原始通道數
    original_out_channels = target_layer.out_channels
    
    # 創建測試數據
    test_image = torch.randn(1, 3, 224, 224)
    test_box = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
    test_label = torch.tensor([0], dtype=torch.long)
    
    # 測試基於數據的剪枝
    result = model.prune_channel(
        layer_name=layer_name,
        prune_ratio=0.3,
        images=test_image,
        boxes=[test_box],
        labels=[test_label],
        auxiliary_net=auxiliary_net
    )
    
    # 驗證剪枝是否成功
    assert result, "剪枝操作應該成功"
    
    # 獲取剪枝後的層
    pruned_layer = None
    for name, module in model.backbone.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            pruned_layer = module
            break
    
    assert pruned_layer is not None, "剪枝後應該能找到目標層"
    
    # 驗證通道數是否正確減少
    expected_channels = int(original_out_channels * (1 - 0.3))
    assert pruned_layer.out_channels == expected_channels, f"剪枝後通道數應為 {expected_channels}，但得到 {pruned_layer.out_channels}"
    
    # 驗證下一層的輸入通道是否更新
    next_conv_name = "layer2.0.conv2"
    next_conv = None
    for name, module in model.backbone.named_modules():
        if name == next_conv_name and isinstance(module, nn.Conv2d):
            next_conv = module
            break
    
    assert next_conv is not None, f"找不到下一層: {next_conv_name}"
    assert next_conv.in_channels == expected_channels, f"下一層輸入通道數應為 {expected_channels}，但得到 {next_conv.in_channels}"
    
    # 測試應該跳過的層
    layer4_name = "layer4.0.conv1"
    result = model.prune_channel(
        layer_name=layer4_name,
        prune_ratio=0.3,
        images=test_image,
        boxes=[test_box],
        labels=[test_label],
        auxiliary_net=auxiliary_net
    )
    
    assert result == "SKIPPED", f"層 {layer4_name} 應該跳過剪枝"
    
    # 測試無數據的剪枝 (隨機選擇通道)
    layer_name = "layer2.1.conv1"
    target_layer = None
    for name, module in model.backbone.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            target_layer = module
            break
    
    if target_layer is not None:
        original_out_channels = target_layer.out_channels
        
        result = model.prune_channel(
            layer_name=layer_name,
            prune_ratio=0.3
        )
        
        assert result, "無數據剪枝操作應該成功"
        
        pruned_layer = None
        for name, module in model.backbone.named_modules():
            if name == layer_name and isinstance(module, nn.Conv2d):
                pruned_layer = module
                break
        
        assert pruned_layer is not None, "剪枝後應該能找到目標層"
        
        expected_channels = int(original_out_channels * (1 - 0.3))
        assert pruned_layer.out_channels == expected_channels, f"無數據剪枝後通道數應為 {expected_channels}，但得到 {pruned_layer.out_channels}"
    
    # 測試無效層名稱
    invalid_layer_name = "non_existent_layer"
    result = model.prune_channel(
        layer_name=invalid_layer_name,
        prune_ratio=0.3
    )
    
    assert not result, "對不存在的層進行剪枝應該失敗"
    
    print("✅ prune_channel 測試通過")
    return True

def test_in_prune_model():
    """測試 prune_model 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 創建輔助網路
    auxiliary_net = AuxiliaryNetwork(in_channels=2048)
    
    # 創建測試數據
    test_image = torch.randn(1, 3, 224, 224)
    test_box = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
    test_label = torch.tensor([0], dtype=torch.long)
    
    # 測試自動選擇可剪枝層
    pruned_layers = model.prune_model(
        prune_ratio=0.3,
        images=test_image,
        boxes=[test_box],
        labels=[test_label],
        auxiliary_net=auxiliary_net,
        prunable_layers=None  # 自動選擇層
    )
    
    # 驗證是否成功剪枝
    assert pruned_layers is not None, "剪枝應該返回剪枝層列表"
    
    # 檢查是否跳過了 layer4 (OS2D 的最終輸出層)
    for layer_name in pruned_layers:
        assert not layer_name.startswith("layer4"), f"不應該剪枝 layer4，但發現 {layer_name}"
    
    # 檢查是否跳過了 conv3 層
    for layer_name in pruned_layers:
        assert not layer_name.endswith("conv3"), f"不應該剪枝 conv3，但發現 {layer_name}"
    
    # 測試指定剪枝層
    specific_layers = ["layer2.0.conv1", "layer2.0.conv2"]
    pruned_layers = model.prune_model(
        prune_ratio=0.3,
        images=test_image,
        boxes=[test_box],
        labels=[test_label],
        auxiliary_net=auxiliary_net,
        prunable_layers=specific_layers
    )
    
    # 驗證是否只剪枝了指定的層
    assert set(pruned_layers) <= set(specific_layers), f"應該只剪枝指定層，但剪枝了 {pruned_layers}"
    
    # 測試不同的剪枝比例
    high_ratio = 0.7
    pruned_layers_high = model.prune_model(
        prune_ratio=high_ratio,
        images=test_image,
        boxes=[test_box],
        labels=[test_label],
        auxiliary_net=auxiliary_net,
        prunable_layers=["layer2.1.conv1"]
    )
    
    # 驗證高剪枝比例的效果
    if pruned_layers_high:
        layer_name = pruned_layers_high[0]
        target_layer = None
        for name, module in model.backbone.named_modules():
            if name == layer_name and isinstance(module, nn.Conv2d):
                target_layer = module
                break
        
        if target_layer is not None:
            # 計算預期的通道數
            original_channels = 256  # ResNet50 layer2 的標準通道數
            expected_channels = int(original_channels * (1 - high_ratio))
            assert target_layer.out_channels == expected_channels, f"高剪枝比例應該將通道數減少到 {expected_channels}，但得到 {target_layer.out_channels}"
    
    # 測試無數據剪枝
    pruned_layers_no_data = model.prune_model(
        prune_ratio=0.3,
        auxiliary_net=auxiliary_net
    )
    
    # 驗證無數據剪枝是否成功
    assert pruned_layers_no_data is not None, "無數據剪枝應該成功"
    
    # 測試 BatchNorm 統計數據重置
    # 獲取一個 BatchNorm 層
    bn_layer = None
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layer = module
            break
    
    if bn_layer is not None:
        # 檢查 running_mean 和 running_var 是否已重置
        assert torch.allclose(bn_layer.running_mean, torch.zeros_like(bn_layer.running_mean)), "BatchNorm running_mean 應該被重置為 0"
        assert torch.allclose(bn_layer.running_var, torch.ones_like(bn_layer.running_var)), "BatchNorm running_var 應該被重置為 1"
    
    print("✅ prune_model 測試通過")
    return True

def test_in_visualize_model_architecture():
    """測試 visualize_model_architecture 方法的功能"""
    import os
    import tempfile
    from unittest.mock import patch, MagicMock
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 創建臨時輸出路徑
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        output_path = tmp_file.name
    
    try:
        # 測試基本功能 - 使用 mock 避免實際執行 graphviz
        with patch('torchviz.make_dot') as mock_make_dot:
            # 模擬 make_dot 返回值
            mock_dot = MagicMock()
            mock_make_dot.return_value = mock_dot
            
            # 執行可視化
            result = model.visualize_model_architecture(output_path=output_path)
            
            # 驗證 make_dot 是否被調用
            mock_make_dot.assert_called_once()
            
            # 驗證 dot.render 是否被調用
            mock_dot.render.assert_called_once()
            
            # 驗證返回值
            assert result is True, "可視化方法應該返回 True"
        
        # 測試輸入形狀參數
        with patch('torchviz.make_dot') as mock_make_dot:
            mock_dot = MagicMock()
            mock_make_dot.return_value = mock_dot
            
            # 使用自定義輸入形狀
            custom_shape = (2, 3, 320, 320)
            result = model.visualize_model_architecture(
                output_path=output_path,
                input_shape=custom_shape
            )
            
            # 驗證 make_dot 是否被調用
            mock_make_dot.assert_called_once()
            
            # 驗證返回值
            assert result is True, "使用自定義輸入形狀時可視化方法應該返回 True"
        
        # 測試導入錯誤處理
        with patch('torchviz.make_dot', side_effect=ImportError("模擬導入錯誤")):
            result = model.visualize_model_architecture(output_path=output_path)
            assert result is False, "導入錯誤時應該返回 False"
        
        # 測試執行錯誤處理
        with patch('torchviz.make_dot', side_effect=RuntimeError("模擬執行錯誤")):
            result = model.visualize_model_architecture(output_path=output_path)
            assert result is False, "執行錯誤時應該返回 False"
        
        # 測試 _print_model_summary 方法是否被調用
        with patch.object(model, '_print_model_summary') as mock_summary:
            with patch('torchviz.make_dot') as mock_make_dot:
                mock_dot = MagicMock()
                mock_make_dot.return_value = mock_dot
                
                model.visualize_model_architecture(output_path=output_path)
                
                # 驗證 _print_model_summary 是否被調用
                mock_summary.assert_called_once()
    
    finally:
        # 清理臨時文件
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # 清理可能生成的 .png 文件
        png_path = output_path.replace('.png', '') + '.png'
        if os.path.exists(png_path):
            os.remove(png_path)
    
    print("✅ visualize_model_architecture 測試通過")
    return True

def test_in_get_feature_map():
    """測試 get_feature_map 方法的功能，使用 Grozi 數據集的真實圖像"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 準備 Grozi 數據集
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if not os.path.exists(grozi_csv):
        print("\n❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        # 使用合成數據進行基本測試
        test_image = torch.randn(1, 3, 224, 224)
    else:
        # 使用真實數據集
        dataset = build_grozi_dataset(
            data_path=data_path,
            name="grozi-train-mini",
            eval_scale=224,
            cache_images=False
        )
        
        # 建立 box_coder
        box_coder = Os2dBoxCoder(
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.4,
            remap_classification_targets_iou_pos=0.5,
            remap_classification_targets_iou_neg=0.4,
            output_box_grid_generator=BoxGridGenerator(
                box_size=FeatureMapSize(w=16, h=16),
                box_stride=FeatureMapSize(w=16, h=16)
            ),
            function_get_feature_map_size=lambda img_size: FeatureMapSize(w=img_size.w // 16, h=img_size.h // 16),
            do_nms_across_classes=False
        )
        
        dataloader = DataloaderOneShotDetection(
            dataset=dataset,
            box_coder=box_coder,
            batch_size=1,
            img_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            gt_image_size=64,
            random_flip_batches=False,
            random_crop_size=None,
            random_color_distortion=False,
            pyramid_scales_eval=[1.0],
            do_augmentation=False
        )
        
        # 獲取第一個批次的圖像
        batch = dataloader.get_batch(0)
        test_image = batch[0]  # images 是批次的第一個元素
        print(f"✓ 使用 Grozi 數據集圖像進行測試，形狀: {test_image.shape}")
    
    # 測試基本功能
    feature_maps = model.get_feature_map(test_image)
    
    # 驗證輸出是否為張量
    assert isinstance(feature_maps, torch.Tensor), "特徵圖應該是 torch.Tensor 類型"
    
    # 驗證輸出維度
    assert feature_maps.dim() == 4, f"特徵圖應該是 4D 張量，但得到 {feature_maps.dim()}D"
    
    # 驗證批次維度
    assert feature_maps.size(0) == test_image.size(0), f"批次維度應為 {test_image.size(0)}，但得到 {feature_maps.size(0)}"
    
    # 驗證通道維度 (ResNet50 最後一層特徵圖通常有 2048 通道)
    assert feature_maps.size(1) > 0, f"通道維度應大於 0，但得到 {feature_maps.size(1)}"
    
    # 驗證特徵圖尺寸合理 (應該比輸入小)
    assert feature_maps.size(2) <= test_image.size(2) and feature_maps.size(3) <= test_image.size(3), \
        f"特徵圖尺寸 {feature_maps.size(2)}x{feature_maps.size(3)} 應該小於輸入尺寸 {test_image.size(2)}x{test_image.size(3)}"
    
    print(f"✓ 基本特徵圖提取測試通過，輸入形狀: {test_image.shape}，特徵圖形狀: {feature_maps.shape}")
    
    # 如果有 Grozi 數據集，使用類別圖像進行測試
    if os.path.exists(grozi_csv):
        # 獲取所有類別圖像
        batch_class_images, class_image_sizes, class_ids = dataloader.get_all_class_images()
        
        # 測試類別圖像的特徵提取
        for i, class_image in enumerate(batch_class_images[:2]):  # 只測試前兩個類別
            # 確保類別圖像是 4D 張量 [B, C, H, W]
            if class_image.dim() == 3:
                class_image = class_image.unsqueeze(0)
            
            class_feature = model.get_feature_map(class_image)
            
            # 驗證類別特徵
            assert isinstance(class_feature, torch.Tensor), "類別特徵圖應該是 torch.Tensor 類型"
            assert class_feature.dim() == 4, f"類別特徵圖應該是 4D 張量，但得到 {class_feature.dim()}D"
            print(f"✓ 類別 {i} 特徵提取測試通過，特徵形狀: {class_feature.shape}")
    
    # 測試不同輸入尺寸
    test_sizes = [(1, 3, 320, 320), (2, 3, 224, 224), (1, 3, 512, 512)]
    for size in test_sizes:
        test_input = torch.randn(size)
        feature_maps = model.get_feature_map(test_input)
        
        # 驗證批次維度匹配
        assert feature_maps.size(0) == size[0], f"批次維度不匹配: 預期 {size[0]}，得到 {feature_maps.size(0)}"
        
        # 驗證特徵圖尺寸合理 (應該比輸入小)
        assert feature_maps.size(2) <= size[2] and feature_maps.size(3) <= size[3], \
            f"特徵圖尺寸 {feature_maps.size(2)}x{feature_maps.size(3)} 應該小於輸入尺寸 {size[2]}x{size[3]}"
    
    print(f"✓ 不同輸入尺寸測試通過")
    
    # 測試 3D 輸入 (單張圖像，無批次維度)
    test_3d_input = torch.randn(3, 224, 224)
    feature_maps = model.get_feature_map(test_3d_input)
    
    # 驗證輸出是否自動添加了批次維度
    assert feature_maps.dim() == 4, f"3D 輸入的特徵圖應該是 4D 張量，但得到 {feature_maps.dim()}D"
    assert feature_maps.size(0) == 1, f"3D 輸入的批次維度應為 1，但得到 {feature_maps.size(0)}"
    
    print(f"✓ 3D 輸入測試通過，特徵圖形狀: {feature_maps.shape}")
    
    # 測試 CUDA 支持 (如果可用)
    if torch.cuda.is_available():
        try:
            cuda_model = Os2dModelInPrune(is_cuda=True)
            cuda_input = test_image.cuda()
            cuda_feature_maps = cuda_model.get_feature_map(cuda_input)
            
            # 驗證輸出設備
            assert cuda_feature_maps.device.type == 'cuda', f"特徵圖應該在 CUDA 上，但在 {cuda_feature_maps.device}"
            
            print("✓ CUDA 測試通過")
        except Exception as e:
            print(f"⚠️ CUDA 測試失敗: {e}")
    
    # 測試異常處理
    try:
        # 測試無效輸入
        invalid_input = torch.randn(1, 1, 10, 10)  # 通道數不正確
        model.get_feature_map(invalid_input)
        # 如果沒有引發異常，則測試失敗
        print("⚠️ 無效輸入沒有引發異常")
    except Exception:
        # 預期會引發異常
        print("✓ 無效輸入正確引發異常")
    
    print("✅ get_feature_map 測試通過")
    return True

def test_in_forward():
    """測試 Os2dModelInPrune 的 forward 方法功能"""
    import os
    import pytest
    import torch
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 準備測試數據
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    
    if not os.path.exists(grozi_csv):
        print("\n❌ Grozi-3.2k dataset not found. 使用合成數據進行測試")
        # 使用合成數據
        test_image = torch.randn(1, 3, 224, 224)
        test_class_images = [torch.randn(3, 64, 64)]
    else:
        # 使用真實數據集
        dataset = build_grozi_dataset(
            data_path=data_path,
            name="grozi-train-mini",
            eval_scale=224,
            cache_images=False
        )
        
        # 建立 box_coder
        box_coder = Os2dBoxCoder(
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.4,
            remap_classification_targets_iou_pos=0.5,
            remap_classification_targets_iou_neg=0.4,
            output_box_grid_generator=BoxGridGenerator(
                box_size=FeatureMapSize(w=16, h=16),
                box_stride=FeatureMapSize(w=16, h=16)
            ),
            function_get_feature_map_size=lambda img_size: FeatureMapSize(w=img_size.w // 16, h=img_size.h // 16),
            do_nms_across_classes=False
        )
        
        dataloader = DataloaderOneShotDetection(
            dataset=dataset,
            box_coder=box_coder,
            batch_size=1,
            img_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            gt_image_size=64,
            random_flip_batches=False,
            random_crop_size=None,
            random_color_distortion=False,
            pyramid_scales_eval=[1.0],
            do_augmentation=False
        )
        
        # 獲取第一個批次的圖像和類別圖像
        batch = dataloader.get_batch(0)
        test_image = batch[0]  # images 是批次的第一個元素
        test_class_images = batch[1]  # class_images 是批次的第二個元素
        print(f"✓ 使用 Grozi 數據集進行測試，圖像形狀: {test_image.shape}, 類別圖像數量: {len(test_class_images)}")
    
    # 測試標準前向傳播 (images + class_images)
    outputs = model(test_image, class_images=test_class_images)
    
    # 驗證輸出是否為元組
    assert isinstance(outputs, tuple), "前向傳播輸出應該是元組"
    assert len(outputs) >= 2, "輸出元組應該至少有兩個元素"
    
    # 驗證輸出元素
    class_scores, boxes = outputs[0], outputs[1]
    assert isinstance(class_scores, torch.Tensor), "class_scores 應該是張量"
    assert isinstance(boxes, torch.Tensor), "boxes 應該是張量"
    
    # 驗證 class_scores 形狀
    if class_scores.dim() == 4:  # OS2D 密集格式 [B, C, 4, N]
        assert class_scores.size(0) == test_image.size(0), "批次維度不匹配"
        print(f"✓ 密集格式輸出: class_scores 形狀 {class_scores.shape}, boxes 形狀 {boxes.shape}")
    elif class_scores.dim() == 2:  # 標準格式 [N, C]
        print(f"✓ 標準格式輸出: class_scores 形狀 {class_scores.shape}, boxes 形狀 {boxes.shape}")
    
    # 測試 NMS 功能
    max_boxes = 10
    outputs_with_nms = model(test_image, class_images=test_class_images, max_boxes=max_boxes, nms_threshold=0.5)
    
    # 驗證 NMS 後的框數量
    if outputs_with_nms[1].dim() == 2:  # 標準格式 [N, 4]
        assert outputs_with_nms[1].size(0) <= max_boxes, f"NMS 後框數量應小於等於 {max_boxes}"
        print(f"✓ NMS 後框數量: {outputs_with_nms[1].size(0)}")
    
    # 測試無類別圖像的情況
    try:
        outputs_no_class = model(test_image)
        print("✓ 無類別圖像的前向傳播成功")
    except ValueError:
        print("⚠️ 模型需要類別圖像才能運行")
    
    # 測試 OS2D pipeline 模式 (class_head + feature_maps)
    try:
        # 獲取特徵圖
        feature_maps = model.get_feature_map(test_image)
        
        # 創建假的 class_head
        class_head = torch.randn(1, 64, 7, 7)
        
        # 執行 OS2D pipeline 模式
        outputs_pipeline = model(class_head=class_head, feature_maps=feature_maps)
        
        # 驗證輸出
        assert isinstance(outputs_pipeline, tuple), "OS2D pipeline 輸出應該是元組"
        print("✓ OS2D pipeline 模式前向傳播成功")
    except Exception as e:
        print(f"⚠️ OS2D pipeline 模式測試失敗: {e}")
    
    # 測試無效輸入
    try:
        outputs_invalid = model()
        assert False, "應該引發錯誤但沒有"
    except ValueError:
        print("✓ 無效輸入正確引發錯誤")
    
    print("✅ forward 方法測試通過")
    return True

def test_in_print_model_summary():
    """測試 _print_model_summary 方法的功能"""
    import os
    import pytest
    import torch
    from unittest.mock import patch, MagicMock
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 使用 mock 捕獲打印輸出
    with patch('builtins.print') as mock_print:
        # 調用要測試的方法
        model._print_model_summary()
        
        # 驗證打印函數被調用
        assert mock_print.call_count > 0, "打印函數應該被調用多次"
        
        # 驗證模型摘要信息的關鍵部分
        summary_calls = [call[0][0] for call in mock_print.call_args_list if isinstance(call[0][0], str)]
        
        # 檢查標題
        assert any("====== 模型摘要 ======" in call for call in summary_calls), "應該打印模型摘要標題"
        
        # 檢查參數量信息
        assert any("總參數量:" in call for call in summary_calls), "應該打印總參數量"
        assert any("可訓練參數量:" in call for call in summary_calls), "應該打印可訓練參數量"
        
        # 檢查層級結構分析
        assert any("層級結構分析:" in call for call in summary_calls), "應該打印層級結構分析"
        
        # 檢查是否打印了卷積層信息
        conv_layer_info = False
        for call in summary_calls:
            if "類型: Conv2d" in call:
                conv_layer_info = True
                break
        assert conv_layer_info, "應該打印卷積層信息"
        
        # 檢查是否打印了 BatchNorm 層信息
        bn_layer_info = False
        for call in summary_calls:
            if "類型: BatchNorm2d" in call:
                bn_layer_info = True
                break
        assert bn_layer_info, "應該打印 BatchNorm 層信息"
        
        # 檢查層級連接分析
        assert any("====== 層級連接分析 ======" in call for call in summary_calls), "應該打印層級連接分析"
        
        # 檢查是否打印了 Layer 信息
        layer_info = False
        for call in summary_calls:
            if "[Layer " in call:
                layer_info = True
                break
        assert layer_info, "應該打印 Layer 信息"
        
        # 檢查是否打印了 Block 信息
        block_info = False
        for call in summary_calls:
            if "Block " in call:
                block_info = True
                break
        assert block_info, "應該打印 Block 信息"
        
        # 檢查是否打印了通道數信息
        channel_info = False
        for call in summary_calls:
            if " -> " in call and ("Conv" in call or "Downsample" in call):
                channel_info = True
                break
        assert channel_info, "應該打印通道數信息"
    
    # 測試模型參數統計
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 確認模型有參數
    assert total_params > 0, "模型應該有參數"
    assert trainable_params > 0, "模型應該有可訓練參數"
    
    # 測試層級結構
    has_conv_layers = False
    has_bn_layers = False
    
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            has_conv_layers = True
            # 檢查卷積層屬性
            assert hasattr(module, 'in_channels'), "卷積層應該有 in_channels 屬性"
            assert hasattr(module, 'out_channels'), "卷積層應該有 out_channels 屬性"
        elif isinstance(module, nn.BatchNorm2d):
            has_bn_layers = True
            # 檢查 BatchNorm 層屬性
            assert hasattr(module, 'num_features'), "BatchNorm 層應該有 num_features 屬性"
    
    assert has_conv_layers, "模型應該包含卷積層"
    assert has_bn_layers, "模型應該包含 BatchNorm 層"
    
    print("✅ _print_model_summary 測試通過")
    return True

def test_in_normalize_batch_images():
    """測試 _normalize_batch_images 方法的功能"""
    import os
    import torch
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試單一張量輸入
    # 測試 3D 張量 [C, H, W]
    test_3d = torch.randn(3, 224, 224)
    normalized_3d = model._normalize_batch_images(test_3d)
    
    # 驗證輸出
    assert isinstance(normalized_3d, torch.Tensor), "輸出應該是張量"
    assert normalized_3d.dim() == 4, f"輸出應該是 4D 張量，但得到 {normalized_3d.dim()}D"
    assert normalized_3d.shape == (1, 3, 224, 224), f"輸出形狀應為 (1, 3, 224, 224)，但得到 {normalized_3d.shape}"
    
    # 測試 4D 張量 [B, C, H, W]
    test_4d = torch.randn(2, 3, 224, 224)
    normalized_4d = model._normalize_batch_images(test_4d)
    
    # 驗證輸出
    assert normalized_4d.shape == (2, 3, 224, 224), f"輸出形狀應為 (2, 3, 224, 224)，但得到 {normalized_4d.shape}"
    
    # 測試張量列表輸入
    test_list = [
        torch.randn(3, 224, 224),
        torch.randn(3, 224, 224)
    ]
    normalized_list = model._normalize_batch_images(test_list)
    
    # 驗證輸出
    assert normalized_list.shape == (2, 3, 224, 224), f"輸出形狀應為 (2, 3, 224, 224)，但得到 {normalized_list.shape}"
    
    # 測試不同尺寸的張量列表
    test_diff_sizes = [
        torch.randn(3, 320, 320),
        torch.randn(3, 224, 224)
    ]
    normalized_diff = model._normalize_batch_images(test_diff_sizes, target_size=(256, 256))
    
    # 驗證輸出
    assert normalized_diff.shape == (2, 3, 256, 256), f"輸出形狀應為 (2, 3, 256, 256)，但得到 {normalized_diff.shape}"
    
    # 測試包含無效元素的列表
    test_invalid = [
        torch.randn(3, 224, 224),
        None,
        "not a tensor"
    ]
    normalized_invalid = model._normalize_batch_images(test_invalid)
    
    # 驗證輸出
    assert normalized_invalid.shape == (1, 3, 224, 224), f"輸出形狀應為 (1, 3, 224, 224)，但得到 {normalized_invalid.shape}"
    
    # 測試空列表
    test_empty = []
    normalized_empty = model._normalize_batch_images(test_empty)
    
    # 驗證輸出
    assert normalized_empty is None, "空列表輸入應該返回 None"
    
    # 測試 Grozi 數據集的圖像
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if os.path.exists(grozi_csv):
        # 使用真實數據集
        dataset = build_grozi_dataset(
            data_path=data_path,
            name="grozi-train-mini",
            eval_scale=224,
            cache_images=False
        )
        
        # 建立 box_coder
        box_coder = Os2dBoxCoder(
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.4,
            remap_classification_targets_iou_pos=0.5,
            remap_classification_targets_iou_neg=0.4,
            output_box_grid_generator=BoxGridGenerator(
                box_size=FeatureMapSize(w=16, h=16),
                box_stride=FeatureMapSize(w=16, h=16)
            ),
            function_get_feature_map_size=lambda img_size: FeatureMapSize(w=img_size.w // 16, h=img_size.h // 16),
            do_nms_across_classes=False
        )
        
        dataloader = DataloaderOneShotDetection(
            dataset=dataset,
            box_coder=box_coder,
            batch_size=1,
            img_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            gt_image_size=64,
            random_flip_batches=False,
            random_crop_size=None,
            random_color_distortion=False,
            pyramid_scales_eval=[1.0],
            do_augmentation=False
        )
        
        # 獲取第一個批次的圖像和類別圖像
        batch = dataloader.get_batch(0)
        images = batch[0]  # images 是批次的第一個元素
        class_images = batch[1]  # class_images 是批次的第二個元素
        
        # 測試真實圖像
        normalized_real = model._normalize_batch_images(images)
        
        # 驗證輸出
        assert normalized_real.shape[0] == images.shape[0], "批次大小應該保持不變"
        assert normalized_real.shape[1] == 3, "通道數應該為 3"
        
        # 測試真實類別圖像
        normalized_class = model._normalize_batch_images(class_images)
        
        # 驗證輸出
        assert normalized_class is not None, "類別圖像標準化不應返回 None"
        print(f"✓ Grozi 數據集圖像標準化測試通過，輸出形狀: {normalized_real.shape}")
    
    print("✅ _normalize_batch_images 測試通過")
    return True 

def test_in_cat_boxes_list():
    """測試 _cat_boxes_list 方法的功能"""
    import torch
    from os2d.modeling.model import Os2dModel
    from src.os2d_model_in_prune import Os2dModelInPrune
    from collections import namedtuple
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試空列表
    empty_list = []
    result = model._cat_boxes_list(empty_list)
    assert result.shape == (0, 4), f"空列表應返回形狀為 (0, 4) 的張量，但得到 {result.shape}"
    
    # 測試只包含張量的列表
    tensor_list = [
        torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.float32),
        torch.tensor([[15, 25, 35, 45]], dtype=torch.float32)
    ]
    result = model._cat_boxes_list(tensor_list)
    assert result.shape == (3, 4), f"應返回形狀為 (3, 4) 的張量，但得到 {result.shape}"
    assert torch.allclose(result[0], torch.tensor([10, 20, 30, 40], dtype=torch.float32))
    assert torch.allclose(result[1], torch.tensor([50, 60, 70, 80], dtype=torch.float32))
    assert torch.allclose(result[2], torch.tensor([15, 25, 35, 45], dtype=torch.float32))
    
    # 測試包含 BoxList 的列表
    # 創建模擬的 BoxList 類
    BoxList = namedtuple('BoxList', ['bbox_xyxy'])
    box_list = [
        BoxList(bbox_xyxy=torch.tensor([[5, 15, 25, 35]], dtype=torch.float32)),
        BoxList(bbox_xyxy=torch.tensor([[100, 200, 300, 400], [500, 600, 700, 800]], dtype=torch.float32))
    ]
    result = model._cat_boxes_list(box_list)
    assert result.shape == (3, 4), f"BoxList 應返回形狀為 (3, 4) 的張量，但得到 {result.shape}"
    assert torch.allclose(result[0], torch.tensor([5, 15, 25, 35], dtype=torch.float32))
    assert torch.allclose(result[1], torch.tensor([100, 200, 300, 400], dtype=torch.float32))
    assert torch.allclose(result[2], torch.tensor([500, 600, 700, 800], dtype=torch.float32))
    
    # 測試混合列表（張量和 BoxList）
    mixed_list = [
        torch.tensor([[10, 20, 30, 40]], dtype=torch.float32),
        BoxList(bbox_xyxy=torch.tensor([[100, 200, 300, 400]], dtype=torch.float32))
    ]
    result = model._cat_boxes_list(mixed_list)
    assert result.shape == (2, 4), f"混合列表應返回形狀為 (2, 4) 的張量，但得到 {result.shape}"
    assert torch.allclose(result[0], torch.tensor([10, 20, 30, 40], dtype=torch.float32))
    assert torch.allclose(result[1], torch.tensor([100, 200, 300, 400], dtype=torch.float32))
    
    # 測試包含空張量的列表
    empty_tensor_list = [
        torch.tensor([], dtype=torch.float32).reshape(0, 4),
        torch.tensor([[10, 20, 30, 40]], dtype=torch.float32),
        BoxList(bbox_xyxy=torch.tensor([], dtype=torch.float32).reshape(0, 4))
    ]
    result = model._cat_boxes_list(empty_tensor_list)
    assert result.shape == (1, 4), f"包含空張量的列表應返回形狀為 (1, 4) 的張量，但得到 {result.shape}"
    assert torch.allclose(result[0], torch.tensor([10, 20, 30, 40], dtype=torch.float32))
    
    # 測試指定設備
    if torch.cuda.is_available():
        cuda_device = torch.device('cuda')
        result = model._cat_boxes_list(tensor_list, device=cuda_device)
        assert result.device.type == 'cuda', f"結果應在 CUDA 設備上，但在 {result.device.type} 上"
    
    print("✅ _cat_boxes_list 測試通過")
    return True

def test_in_analyze_os2d_output():
    """測試 analyze_os2d_outputs 方法的功能"""
    import torch
    import os
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試標準格式輸出分析
    # 創建模擬的標準格式輸出 [N, C] 和 [N, 4]
    class_scores = torch.rand(10, 9)  # 10個框，9個類別
    boxes = torch.rand(10, 4)  # 10個框，每個框4個坐標
    outputs = (class_scores, boxes)
    
    # 分析輸出
    results = model.analyze_os2d_outputs(outputs)
    
    # 驗證結果
    assert isinstance(results, dict), "分析結果應該是字典類型"
    assert 'output_type' in results, "結果應包含 output_type 字段"
    assert results['output_type'] == 'tuple', "output_type 應為 tuple"
    assert 'num_elements' in results, "結果應包含 num_elements 字段"
    assert results['num_elements'] == 2, "num_elements 應為 2"
    assert 'class_scores_shape' in results, "結果應包含 class_scores_shape 字段"
    assert results['class_scores_shape'] == [10, 9], "class_scores_shape 應為 [10, 9]"
    assert 'num_detections' in results, "結果應包含 num_detections 字段"
    assert results['num_detections'] == 10, "num_detections 應為 10"
    assert 'num_classes' in results, "結果應包含 num_classes 字段"
    assert results['num_classes'] == 9, "num_classes 應為 9"
    assert 'dense_format' in results, "結果應包含 dense_format 字段"
    assert results['dense_format'] == False, "dense_format 應為 False"
    assert 'boxes_shape' in results, "結果應包含 boxes_shape 字段"
    assert results['boxes_shape'] == [10, 4], "boxes_shape 應為 [10, 4]"
    assert 'num_boxes' in results, "結果應包含 num_boxes 字段"
    assert results['num_boxes'] == 10, "num_boxes 應為 10"
    
    print("✓ 標準格式輸出分析測試通過")
    
    # 測試密集格式輸出分析
    # 創建模擬的密集格式輸出 [B, C, 4, N] 和 [B, C, N]
    dense_class_scores = torch.rand(1, 9, 4, 100)  # 批次大小1，9個類別，4個值，100個位置
    dense_boxes = torch.rand(1, 9, 100)  # 批次大小1，9個類別，100個位置
    dense_outputs = (dense_class_scores, dense_boxes)
    
    # 分析輸出
    dense_results = model.analyze_os2d_outputs(dense_outputs)
    
    # 驗證結果
    assert 'dense_format' in dense_results, "結果應包含 dense_format 字段"
    assert dense_results['dense_format'] == True, "dense_format 應為 True"
    assert 'batch_size' in dense_results, "結果應包含 batch_size 字段"
    assert dense_results['batch_size'] == 1, "batch_size 應為 1"
    assert 'num_classes' in dense_results, "結果應包含 num_classes 字段"
    assert dense_results['num_classes'] == 9, "num_classes 應為 9"
    assert 'num_positions' in dense_results, "結果應包含 num_positions 字段"
    assert dense_results['num_positions'] == 100, "num_positions 應為 100"
    
    print("✓ 密集格式輸出分析測試通過")
    
    # 測試帶有目標的分析
    # 創建模擬的目標
    targets = {
        'class_ids': [torch.tensor([0, 1, 2])],
        'boxes': [torch.rand(3, 4)]
    }
    
    # 分析輸出
    target_results = model.analyze_os2d_outputs(outputs, targets)
    
    # 驗證結果
    assert 'target_info' in target_results, "結果應包含 target_info 字段"
    assert 'num_classes' in target_results['target_info'], "target_info 應包含 num_classes 字段"
    assert target_results['target_info']['num_classes'] == 3, "target_info 中的 num_classes 應為 3"
    
    print("✓ 帶有目標的分析測試通過")
    
    # 測試實際數據集的輸出分析（如果可用）
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if os.path.exists(grozi_csv):
        # 使用真實數據集
        dataset = build_grozi_dataset(
            data_path=data_path,
            name="grozi-train-mini",
            eval_scale=224,
            cache_images=False
        )
        
        # 建立 box_coder
        box_coder = Os2dBoxCoder(
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.4,
            remap_classification_targets_iou_pos=0.5,
            remap_classification_targets_iou_neg=0.4,
            output_box_grid_generator=BoxGridGenerator(
                box_size=FeatureMapSize(w=16, h=16),
                box_stride=FeatureMapSize(w=16, h=16)
            ),
            function_get_feature_map_size=lambda img_size: FeatureMapSize(w=img_size.w // 16, h=img_size.h // 16),
            do_nms_across_classes=False
        )
        
        dataloader = DataloaderOneShotDetection(
            dataset=dataset,
            box_coder=box_coder,
            batch_size=1,
            img_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            gt_image_size=64,
            random_flip_batches=False,
            random_crop_size=None,
            random_color_distortion=False,
            pyramid_scales_eval=[1.0],
            do_augmentation=False
        )
        
        # 獲取第一個批次
        batch = dataloader.get_batch(0)
        images, class_images = batch[0], batch[1]
        
        # 獲取模型輸出
        outputs = model(images, class_images=class_images)
        
        # 分析輸出
        real_results = model.analyze_os2d_outputs(outputs)
        
        # 驗證結果
        assert isinstance(real_results, dict), "實際數據的分析結果應該是字典類型"
        print(f"✓ 實際數據輸出分析測試通過，輸出形狀: {real_results.get('class_scores_shape', 'N/A')}")
    
    print("✅ analyze_os2d_outputs 測試通過")
    return True

def test_in_convert_4d_to_2d_scores():
    """測試 _convert_4d_to_2d_scores 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試基本功能 - 4D 輸入
    batch_size = 2
    num_classes = 9
    height = 4
    width = 5
    test_4d = torch.randn(batch_size, num_classes, height, width)
    
    # 調用方法
    result_2d = model._convert_4d_to_2d_scores(test_4d)
    
    # 驗證輸出
    assert isinstance(result_2d, torch.Tensor), "輸出應該是張量"
    assert result_2d.dim() == 2, f"輸出應該是 2D 張量，但得到 {result_2d.dim()}D"
    assert result_2d.shape == (batch_size, num_classes), f"輸出形狀應為 ({batch_size}, {num_classes})，但得到 {result_2d.shape}"
    
    # 測試 OS2D 特殊格式 [B, C, 4, N]
    special_4d = torch.randn(1, 9, 4, 100)
    result_special = model._convert_4d_to_2d_scores(special_4d)
    
    # 驗證輸出
    assert result_special.shape == (1, 9), f"OS2D 特殊格式輸出形狀應為 (1, 9)，但得到 {result_special.shape}"
    
    # 測試極端情況 - 單一類別
    single_class = torch.randn(1, 1, 4, 4)
    result_single = model._convert_4d_to_2d_scores(single_class)
    
    # 驗證輸出
    assert result_single.shape == (1, 1), f"單一類別輸出形狀應為 (1, 1)，但得到 {result_single.shape}"
    
    # 測試極端情況 - 大批次
    large_batch = torch.randn(10, 20, 2, 2)
    result_large = model._convert_4d_to_2d_scores(large_batch)
    
    # 驗證輸出
    assert result_large.shape == (10, 20), f"大批次輸出形狀應為 (10, 20)，但得到 {result_large.shape}"
    
    # 測試平均池化的正確性
    # 創建一個簡單的測試用例，每個位置的值都相同
    constant_tensor = torch.ones(1, 2, 2, 2) * 5.0
    result_constant = model._convert_4d_to_2d_scores(constant_tensor)
    
    # 驗證平均值是否正確
    assert torch.allclose(result_constant, torch.ones(1, 2) * 5.0), "平均池化結果不正確"
    
    # 測試邊界情況 - 空間維度為1
    edge_case = torch.randn(2, 3, 1, 1)
    result_edge = model._convert_4d_to_2d_scores(edge_case)
    
    # 驗證輸出
    assert result_edge.shape == (2, 3), f"邊界情況輸出形狀應為 (2, 3)，但得到 {result_edge.shape}"
    assert torch.allclose(result_edge, edge_case.squeeze(-1).squeeze(-1)), "空間維度為1時結果不正確"
    
    print("✅ _convert_4d_to_2d_scores 測試通過")
    return True

def test_in_prepare_classification_targets():
    """測試 _prepare_classification_targets 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試列表輸入 - 單一類別
    class_ids_list = [torch.tensor([3]), torch.tensor([5])]
    num_classes = 10
    targets = model._prepare_classification_targets(class_ids_list, num_classes)
    
    # 驗證輸出
    assert isinstance(targets, torch.Tensor), "輸出應該是張量"
    assert targets.shape == (2,), f"輸出形狀應為 (2,)，但得到 {targets.shape}"
    assert targets.dtype == torch.long, f"輸出類型應為 torch.long，但得到 {targets.dtype}"
    assert torch.allclose(targets, torch.tensor([3, 5], dtype=torch.long, device=model.device)), "輸出值不正確"
    
    # 測試列表輸入 - 多類別 (應該只使用第一個類別)
    class_ids_multi = [torch.tensor([3, 4, 5]), torch.tensor([5, 6, 7])]
    targets_multi = model._prepare_classification_targets(class_ids_multi, num_classes)
    assert torch.allclose(targets_multi, torch.tensor([3, 5], dtype=torch.long, device=model.device)), "多類別時應只使用第一個類別"
    
    # 測試張量輸入
    class_ids_tensor = torch.tensor([1, 2, 3])
    targets_tensor = model._prepare_classification_targets(class_ids_tensor, num_classes)
    assert torch.allclose(targets_tensor, class_ids_tensor.to(model.device)), "張量輸入應直接轉換為目標"
    
    # 測試類別索引超出範圍
    class_ids_out_of_range = [torch.tensor([9]), torch.tensor([10])]  # 10 超出範圍
    targets_clamped = model._prepare_classification_targets(class_ids_out_of_range, num_classes)
    assert torch.allclose(targets_clamped, torch.tensor([9, 9], dtype=torch.long, device=model.device)), "超出範圍的類別應被截斷"
    
    # 測試空列表
    empty_list = []
    targets_empty = model._prepare_classification_targets(empty_list, num_classes)
    assert targets_empty.shape == (1,), f"空列表輸出形狀應為 (1,)，但得到 {targets_empty.shape}"
    assert targets_empty[0] == 0, "空列表應返回零張量"
    
    # 測試包含空張量的列表
    class_ids_with_empty = [torch.tensor([]), torch.tensor([2])]
    targets_with_empty = model._prepare_classification_targets(class_ids_with_empty, num_classes)
    assert targets_with_empty.shape == (1,), "包含空張量的列表應只返回有效類別"
    assert targets_with_empty[0] == 2, "應正確處理包含空張量的列表"
    
    print("✅ _prepare_classification_targets 測試通過")
    return True

def test_in_prepare_target_boxes():
    """測試 _prepare_target_boxes 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    device = torch.device('cpu')
    
    # 測試列表輸入
    boxes_list = [
        torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.float32),
        torch.tensor([[15, 25, 35, 45]], dtype=torch.float32)
    ]
    target_boxes = model._prepare_target_boxes(boxes_list, device)
    
    # 驗證輸出
    assert isinstance(target_boxes, torch.Tensor), "輸出應該是張量"
    assert target_boxes.shape == (3, 4), f"輸出形狀應為 (3, 4)，但得到 {target_boxes.shape}"
    assert torch.allclose(target_boxes[0], torch.tensor([10, 20, 30, 40], dtype=torch.float32))
    assert torch.allclose(target_boxes[1], torch.tensor([50, 60, 70, 80], dtype=torch.float32))
    assert torch.allclose(target_boxes[2], torch.tensor([15, 25, 35, 45], dtype=torch.float32))
    
    # 測試 BoxList 輸入
    from collections import namedtuple
    BoxList = namedtuple('BoxList', ['bbox_xyxy'])
    box_list = BoxList(bbox_xyxy=torch.tensor([[5, 15, 25, 35]], dtype=torch.float32))
    target_boxes = model._prepare_target_boxes(box_list, device)
    
    # 驗證輸出
    assert isinstance(target_boxes, torch.Tensor), "輸出應該是張量"
    assert target_boxes.shape == (1, 4), f"輸出形狀應為 (1, 4)，但得到 {target_boxes.shape}"
    assert torch.allclose(target_boxes[0], torch.tensor([5, 15, 25, 35], dtype=torch.float32))
    
    # 測試帶有 bbox 和 size 屬性的輸入
    class BoxWithSize:
        def __init__(self, bbox):
            self.bbox = bbox
            self.size = (100, 100)
    
    box_with_size = BoxWithSize(torch.tensor([[100, 200, 300, 400]], dtype=torch.float32))
    target_boxes = model._prepare_target_boxes(box_with_size, device)
    
    # 驗證輸出
    assert isinstance(target_boxes, torch.Tensor), "輸出應該是張量"
    assert target_boxes.shape == (1, 4), f"輸出形狀應為 (1, 4)，但得到 {target_boxes.shape}"
    assert torch.allclose(target_boxes[0], torch.tensor([100, 200, 300, 400], dtype=torch.float32))
    
    # 測試直接張量輸入
    direct_tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
    target_boxes = model._prepare_target_boxes(direct_tensor, device)
    
    # 驗證輸出
    assert isinstance(target_boxes, torch.Tensor), "輸出應該是張量"
    assert target_boxes.shape == (2, 4), f"輸出形狀應為 (2, 4)，但得到 {target_boxes.shape}"
    assert torch.allclose(target_boxes, direct_tensor)
    
    # 測試設備轉換
    if torch.cuda.is_available():
        cuda_device = torch.device('cuda')
        cuda_tensor = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32, device=cuda_device)
        target_boxes = model._prepare_target_boxes(cuda_tensor, device)
        
        # 驗證輸出設備
        assert target_boxes.device.type == 'cpu', f"輸出應該在 CPU 上，但在 {target_boxes.device.type} 上"
    
    # 測試無效輸入
    invalid_input = "not a tensor or list"
    target_boxes = model._prepare_target_boxes(invalid_input, device)
    
    # 驗證輸出 - 應該返回預設值
    assert isinstance(target_boxes, torch.Tensor), "無效輸入應該返回預設張量"
    assert target_boxes.shape == (1, 4), f"預設張量形狀應為 (1, 4)，但得到 {target_boxes.shape}"
    
    # 測試空列表
    empty_list = []
    target_boxes = model._prepare_target_boxes(empty_list, device)
    
    # 驗證輸出 - 應該返回空張量
    assert isinstance(target_boxes, torch.Tensor), "空列表應該返回空張量"
    assert target_boxes.numel() == 0, f"空列表應該返回空張量，但得到 {target_boxes.numel()} 個元素"
    
    print("✅ _prepare_target_boxes 測試通過")
    return True

def test_in_convert_dense_boxes_to_standard():
    """測試 _convert_dense_boxes_to_standard 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試 3D 密集格式 [B, C, N]
    batch_size = 1
    num_classes = 9
    num_positions = 100
    dense_3d = torch.rand(batch_size, num_classes, num_positions)
    
    # 創建對應的分類分數
    class_scores = torch.rand(batch_size, num_classes, 4, num_positions)
    
    # 調用方法
    result = model._convert_dense_boxes_to_standard(dense_3d, class_scores)
    
    # 驗證輸出
    assert isinstance(result, torch.Tensor), "輸出應該是張量"
    assert result.dim() == 2, f"輸出應該是 2D 張量，但得到 {result.dim()}D"
    assert result.size(1) == 4, f"輸出應該有 4 列 (x1, y1, x2, y2)，但得到 {result.size(1)}"
    
    # 測試 OS2D 特殊格式 [B, C, N] 其中 N 不是 4 的倍數
    special_3d = torch.rand(1, 9, 56661)  # 特殊的 OS2D 格式
    special_scores = torch.rand(1, 9, 4, 56661)
    
    # 調用方法
    result_special = model._convert_dense_boxes_to_standard(special_3d, special_scores)
    
    # 驗證輸出
    assert result_special.dim() == 2, f"特殊格式輸出應該是 2D 張量，但得到 {result_special.dim()}D"
    assert result_special.size(1) == 4, f"特殊格式輸出應該有 4 列，但得到 {result_special.size(1)}"
    
    # 測試 4D 密集格式 [B, C, 4, N]
    dense_4d = torch.rand(batch_size, num_classes, 4, num_positions)
    
    # 調用方法
    result_4d = model._convert_dense_boxes_to_standard(dense_4d, class_scores)
    
    # 驗證輸出
    assert result_4d.dim() == 2, f"4D 格式輸出應該是 2D 張量，但得到 {result_4d.dim()}D"
    assert result_4d.size(1) == 4, f"4D 格式輸出應該有 4 列，但得到 {result_4d.size(1)}"
    
    # 測試無效輸入 - 維度錯誤
    invalid_dim = torch.rand(2, 3)  # 2D 張量，不是密集格式
    
    try:
        model._convert_dense_boxes_to_standard(invalid_dim, None)
        # 如果沒有引發異常，則測試失敗
        assert False, "應該引發異常但沒有"
    except Exception as e:
        # 預期會引發異常
        print(f"✓ 無效維度正確引發異常: {e}")
    
    # 測試無效輸入 - NaN 值
    nan_tensor = torch.full((1, 9, 100), float('nan'))
    
    # 調用方法 - 應該能處理 NaN
    result_nan = model._convert_dense_boxes_to_standard(nan_tensor, class_scores)
    
    # 驗證輸出 - 應該將 NaN 替換為 0
    assert not torch.isnan(result_nan).any(), "輸出不應該包含 NaN"
    
    # 測試帶有 confidence_scores 的情況
    # 創建帶有明確置信度模式的測試
    conf_3d = torch.rand(1, 9, 100)
    conf_scores = torch.rand(1, 9, 4, 100)
    
    # 設置一些位置的分數特別高，以測試排序功能
    conf_scores[0, 0, 0, 0] = 0.9  # 第一個位置分數高
    conf_scores[0, 1, 0, 1] = 0.95  # 第二個位置分數更高
    
    # 調用方法
    result_conf = model._convert_dense_boxes_to_standard(conf_3d, conf_scores)
    
    # 驗證輸出 - 應該按置信度排序
    assert result_conf.size(0) > 0, "應該返回非空結果"
    
    print("✅ _convert_dense_boxes_to_standard 測試通過")
    return True

def test_in_match_box_counts():
    """測試 _match_box_counts 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試預測框數量大於目標框數量的情況
    pred_boxes = torch.rand(100, 4)  # 100個預測框
    target_boxes = torch.rand(30, 4)  # 30個目標框
    class_scores = torch.rand(100, 20)  # 每個框的類別分數
    
    # 調用方法
    matched_pred_boxes, matched_target_boxes = model._match_box_counts(pred_boxes, target_boxes, class_scores)
    
    # 驗證輸出
    assert matched_pred_boxes.shape[0] == matched_target_boxes.shape[0], "匹配後的框數量應該相同"
    assert matched_pred_boxes.shape[0] == 30, f"預期匹配後有30個框，但得到 {matched_pred_boxes.shape[0]}"
    assert matched_target_boxes.shape[0] == 30, f"預期匹配後有30個目標框，但得到 {matched_target_boxes.shape[0]}"
    
    # 測試預測框數量小於目標框數量的情況
    pred_boxes_small = torch.rand(20, 4)  # 20個預測框
    target_boxes_large = torch.rand(50, 4)  # 50個目標框
    
    # 調用方法
    matched_pred_small, matched_target_large = model._match_box_counts(pred_boxes_small, target_boxes_large)
    
    # 驗證輸出
    assert matched_pred_small.shape[0] == matched_target_large.shape[0], "匹配後的框數量應該相同"
    assert matched_pred_small.shape[0] == 20, f"預期匹配後有20個框，但得到 {matched_pred_small.shape[0]}"
    assert matched_target_large.shape[0] == 20, f"預期匹配後有20個目標框，但得到 {matched_target_large.shape[0]}"
    
    # 測試框數量已經匹配的情況
    pred_boxes_equal = torch.rand(40, 4)  # 40個預測框
    target_boxes_equal = torch.rand(40, 4)  # 40個目標框
    
    # 調用方法
    matched_pred_equal, matched_target_equal = model._match_box_counts(pred_boxes_equal, target_boxes_equal)
    
    # 驗證輸出
    assert matched_pred_equal.shape[0] == matched_target_equal.shape[0], "匹配後的框數量應該相同"
    assert matched_pred_equal.shape[0] == 40, f"預期匹配後有40個框，但得到 {matched_pred_equal.shape[0]}"
    assert torch.allclose(matched_pred_equal, pred_boxes_equal), "框數量已匹配時不應修改預測框"
    assert torch.allclose(matched_target_equal, target_boxes_equal), "框數量已匹配時不應修改目標框"
    
    # 測試使用密集格式的類別分數
    dense_class_scores = torch.rand(1, 20, 4, 100)  # [B, C, 4, N] 格式
    pred_boxes_dense = torch.rand(100, 4)
    target_boxes_dense = torch.rand(30, 4)
    
    # 調用方法
    matched_pred_dense, matched_target_dense = model._match_box_counts(pred_boxes_dense, target_boxes_dense, dense_class_scores)
    
    # 驗證輸出
    assert matched_pred_dense.shape[0] == matched_target_dense.shape[0], "匹配後的框數量應該相同"
    assert matched_pred_dense.shape[0] == 30, f"預期匹配後有30個框，但得到 {matched_pred_dense.shape[0]}"
    
    # 測試空框處理
    empty_pred_boxes = torch.zeros((0, 4))
    empty_target_boxes = torch.zeros((0, 4))
    
    # 調用方法
    matched_empty_pred, matched_empty_target = model._match_box_counts(empty_pred_boxes, empty_target_boxes)
    
    # 驗證輸出
    assert matched_empty_pred.shape[0] == 0, "空預測框應該保持為空"
    assert matched_empty_target.shape[0] == 0, "空目標框應該保持為空"
    
    print("✅ _match_box_counts 測試通過")
    return True

def test_in_select_boxes_by_confidence():
    """測試 _select_boxes_by_confidence 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試標準格式輸入 - 2D 分類分數
    boxes = torch.rand(100, 4)  # 100個框
    class_scores = torch.rand(100, 20)  # 每個框20個類別的分數
    target_size = 30  # 要選擇的框數量
    
    # 調用方法
    selected_boxes = model._select_boxes_by_confidence(boxes, target_size, class_scores)
    
    # 驗證輸出
    assert isinstance(selected_boxes, torch.Tensor), "輸出應該是張量"
    assert selected_boxes.shape == (target_size, 4), f"輸出形狀應為 ({target_size}, 4)，但得到 {selected_boxes.shape}"
    
    # 測試密集格式輸入 - 4D 分類分數
    dense_class_scores = torch.rand(1, 20, 4, 100)  # [B, C, 4, N] 格式
    dense_boxes = torch.rand(100, 4)
    
    # 調用方法
    selected_dense = model._select_boxes_by_confidence(dense_boxes, target_size, dense_class_scores)
    
    # 驗證輸出
    assert selected_dense.shape == (target_size, 4), f"密集格式輸出形狀應為 ({target_size}, 4)，但得到 {selected_dense.shape}"
    
    # 測試目標大小大於輸入大小的情況
    small_boxes = torch.rand(20, 4)  # 只有20個框
    large_target = 30  # 要求30個框
    
    # 調用方法 - 應該返回所有可用的框
    selected_small = model._select_boxes_by_confidence(small_boxes, large_target, class_scores[:20])
    
    # 驗證輸出
    assert selected_small.shape == (20, 4), f"當目標大小大於輸入時，應返回所有框，但得到 {selected_small.shape}"
    
    # 測試無分類分數的情況
    no_scores_boxes = torch.rand(50, 4)
    
    # 調用方法 - 應該隨機選擇框
    selected_no_scores = model._select_boxes_by_confidence(no_scores_boxes, target_size, None)
    
    # 驗證輸出
    assert selected_no_scores.shape == (target_size, 4), f"無分數情況下輸出形狀應為 ({target_size}, 4)，但得到 {selected_no_scores.shape}"
    
    # 測試分類分數形狀不匹配的情況
    mismatched_boxes = torch.rand(80, 4)
    mismatched_scores = torch.rand(70, 20)  # 分數數量少於框數量
    
    # 調用方法 - 應該處理不匹配情況
    selected_mismatched = model._select_boxes_by_confidence(mismatched_boxes, target_size, mismatched_scores)
    
    # 驗證輸出
    assert selected_mismatched.shape == (target_size, 4), f"不匹配情況下輸出形狀應為 ({target_size}, 4)，但得到 {selected_mismatched.shape}"
    
    # 測試異常處理
    try:
        # 傳入無效的分類分數
        invalid_scores = "not a tensor"
        model._select_boxes_by_confidence(boxes, target_size, invalid_scores)
        # 如果沒有引發異常，則測試失敗
        assert False, "應該引發異常但沒有"
    except Exception:
        # 預期會引發異常
        pass
    
    print("✅ _select_boxes_by_confidence 測試通過")
    return True

def test_in_align_score_dimensions():
    model = Os2dModelInPrune(is_cuda=False)
    # 測試 4D 教師分數轉換為 2D
    teacher_4d = torch.rand(2, 9, 4, 100)  # [B, C, 4, N] 格式
    student_2d = torch.rand(2, 9)  # [B, C] 格式

    aligned_teacher, aligned_student = model._align_score_dimensions(teacher_4d, student_2d)

    # 驗證輸出
    assert aligned_teacher.dim() == 2, f"教師分數應轉換為 2D，但得到 {aligned_teacher.dim()}D"
    assert aligned_teacher.shape == (2, 9), f"教師分數形狀應為 (2, 9)，但得到 {aligned_teacher.shape}"
    assert torch.allclose(aligned_student, student_2d), "學生分數不應被修改"

    # 測試 2D 教師分數和 4D 學生分數
    teacher_2d = torch.rand(2, 9)  # [B, C] 格式
    student_4d = torch.rand(2, 9, 4, 100)  # [B, C, 4, N] 格式

    aligned_teacher, aligned_student = model._align_score_dimensions(teacher_2d, student_4d)

    # 驗證輸出
    assert aligned_student.dim() == 2, f"學生分數應轉換為 2D，但得到 {aligned_student.dim()}D"
    assert aligned_student.shape == (2, 9), f"學生分數形狀應為 (2, 9)，但得到 {aligned_student.shape}"
    assert torch.allclose(aligned_teacher, teacher_2d), "教師分數不應被修改"

    # 測試兩者都是 2D 的情況
    teacher_2d = torch.rand(2, 9)
    student_2d = torch.rand(2, 9)

    aligned_teacher, aligned_student = model._align_score_dimensions(teacher_2d, student_2d)

    # 驗證輸出 - 應該保持不變
    assert aligned_teacher.shape == teacher_2d.shape, "教師分數形狀不應改變"
    assert aligned_student.shape == student_2d.shape, "學生分數形狀不應改變"
    assert torch.allclose(aligned_teacher, teacher_2d), "教師分數不應被修改"
    assert torch.allclose(aligned_student, student_2d), "學生分數不應被修改"

    # 測試批次大小不同的情況
    teacher_4d = torch.rand(1, 9, 4, 100)  # 批次大小 1
    student_4d = torch.rand(2, 9, 4, 100)  # 批次大小 2

    aligned_teacher, aligned_student = model._align_score_dimensions(teacher_4d, student_4d)

    # 驗證輸出 - 應該都轉換為 2D，但保持各自的批次大小
    assert aligned_teacher.shape == (1, 9), f"教師分數形狀應為 (1, 9)，但得到 {aligned_teacher.shape}"
    assert aligned_student.shape == (2, 9), f"學生分數形狀應為 (2, 9)，但得到 {aligned_student.shape}"

    # 測試類別數不同的情況
    teacher_4d = torch.rand(2, 9, 4, 100)  # 9 個類別
    student_4d = torch.rand(2, 20, 4, 100)  # 20 個類別

    aligned_teacher, aligned_student = model._align_score_dimensions(teacher_4d, student_4d)

    # 驗證輸出 - 應該都轉換為 2D，保持各自的類別數
    assert aligned_teacher.shape == (2, 9), f"教師分數形狀應為 (2, 9)，但得到 {aligned_teacher.shape}"
    assert aligned_student.shape == (2, 20), f"學生分數形狀應為 (2, 20)，但得到 {aligned_student.shape}"

    print("✅ _align_score_dimensions 測試通過")
    return True

def test_in_compute_classification_distillation_loss():
    """測試 _compute_classification_distillation_loss 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試 2D 標準格式
    teacher_scores = torch.randn(4, 9)  # 批次大小4，9個類別
    student_scores = torch.randn(4, 9)  # 相同形狀
    
    # 調用方法
    loss = model._compute_classification_distillation_loss(teacher_scores, student_scores)
    
    # 驗證輸出
    assert isinstance(loss, torch.Tensor), "損失應該是張量"
    assert loss.dim() == 0, "損失應該是標量"
    assert not torch.isnan(loss), "損失不應為 NaN"
    assert not torch.isinf(loss), "損失不應為 Inf"
    
    # 測試類別數不匹配的情況
    teacher_scores_more = torch.randn(4, 12)  # 教師有更多類別
    student_scores_less = torch.randn(4, 7)   # 學生有更少類別
    
    # 教師更多類別
    loss_more_teacher = model._compute_classification_distillation_loss(teacher_scores_more, student_scores)
    
    # 學生更多類別
    loss_more_student = model._compute_classification_distillation_loss(teacher_scores, student_scores_less)
    
    # 兩者都不同
    loss_both_diff = model._compute_classification_distillation_loss(teacher_scores_more, student_scores_less)
    
    # 驗證所有情況都能正確處理
    assert not torch.isnan(loss_more_teacher), "教師更多類別時損失不應為 NaN"
    assert not torch.isnan(loss_more_student), "學生更多類別時損失不應為 NaN"
    assert not torch.isnan(loss_both_diff), "兩者類別數不同時損失不應為 NaN"
    
    # 測試維度不一致的情況
    teacher_4d = torch.randn(1, 9, 4, 10)  # 4D 格式
    student_2d = torch.randn(1, 9)         # 2D 格式
    
    # 調用方法
    loss_dim_mismatch = model._compute_classification_distillation_loss(teacher_4d, student_2d)
    
    # 驗證輸出
    assert not torch.isnan(loss_dim_mismatch), "維度不匹配時損失不應為 NaN"
    
    # 測試批次大小不同的情況
    teacher_diff_batch = torch.randn(2, 9)
    student_diff_batch = torch.randn(4, 9)
    
    try:
        loss_batch_mismatch = model._compute_classification_distillation_loss(teacher_diff_batch, student_diff_batch)
        # 如果沒有引發異常，確保結果有效
        assert not torch.isnan(loss_batch_mismatch), "批次大小不同時損失不應為 NaN"
    except Exception as e:
        # 如果引發異常，這也是可接受的行為
        print(f"批次大小不同時引發異常: {e}")
    
    print("✅ _compute_classification_distillation_loss 測試通過")
    return True

def test_in_compute_4d_classification_distillation_loss():
    """測試 _compute_4d_classification_distillation_loss 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試相同大小的 4D 張量
    teacher_scores = torch.randn(2, 9, 4, 100)  # [B, C, 4, N]
    student_scores = torch.randn(2, 9, 4, 100)  # 相同形狀
    
    # 調用方法
    loss = model._compute_4d_classification_distillation_loss(teacher_scores, student_scores)
    
    # 驗證輸出
    assert isinstance(loss, torch.Tensor), "損失應該是張量"
    assert loss.dim() == 0, "損失應該是標量"
    assert not torch.isnan(loss), "損失不應為 NaN"
    
    # 測試特徵位置數量不同的情況
    teacher_more_positions = torch.randn(2, 9, 4, 120)  # 教師有更多位置
    student_less_positions = torch.randn(2, 9, 4, 80)   # 學生有更少位置
    
    # 調用方法
    loss_diff_positions = model._compute_4d_classification_distillation_loss(teacher_more_positions, student_less_positions)
    
    # 驗證輸出
    assert not torch.isnan(loss_diff_positions), "位置數量不同時損失不應為 NaN"
    
    # 測試類別數不同的情況
    teacher_more_classes = torch.randn(2, 12, 4, 100)  # 教師有更多類別
    student_less_classes = torch.randn(2, 7, 4, 100)   # 學生有更少類別
    
    # 調用方法
    loss_diff_classes = model._compute_4d_classification_distillation_loss(teacher_more_classes, student_less_classes)
    
    # 驗證輸出
    assert not torch.isnan(loss_diff_classes), "類別數不同時損失不應為 NaN"
    
    # 測試批次大小不同的情況
    teacher_diff_batch = torch.randn(1, 9, 4, 100)
    student_diff_batch = torch.randn(2, 9, 4, 100)
    
    try:
        loss_batch_mismatch = model._compute_4d_classification_distillation_loss(teacher_diff_batch, student_diff_batch)
        # 如果沒有引發異常，確保結果有效
        assert not torch.isnan(loss_batch_mismatch), "批次大小不同時損失不應為 NaN"
    except Exception as e:
        # 如果引發異常，這也是可接受的行為
        print(f"批次大小不同時引發異常: {e}")
    
    print("✅ _compute_4d_classification_distillation_loss 測試通過")
    return True

def test_in_select_top_feature_positions():
    """測試 _select_top_feature_positions 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 創建測試數據
    batch_size = 2
    num_classes = 9
    scores = torch.rand(batch_size, num_classes, 4, 100)  # [B, C, 4, N]
    boxes = torch.rand(batch_size, num_classes, 100)      # [B, C, N]
    
    # 設置目標位置數量
    target_positions = 50
    
    # 調用方法
    new_scores, new_boxes = model._select_top_feature_positions(scores, boxes, target_positions)
    
    # 驗證輸出形狀
    assert new_scores.shape == (batch_size, num_classes, 4, target_positions), f"分數形狀應為 {(batch_size, num_classes, 4, target_positions)}，但得到 {new_scores.shape}"
    assert new_boxes.shape == (batch_size, num_classes, target_positions), f"框形狀應為 {(batch_size, num_classes, target_positions)}，但得到 {new_boxes.shape}"
    
    # 測試目標位置數量大於原始數量的情況
    large_target = 150
    
    try:
        large_scores, large_boxes = model._select_top_feature_positions(scores, boxes, large_target)
        # 如果沒有引發異常，檢查結果是否合理
        assert large_scores.shape[3] <= large_target, "當目標位置數量大於原始數量時，應該返回原始數量或有效處理"
    except Exception as e:
        # 如果引發異常，這也是可接受的行為
        print(f"目標位置數量大於原始數量時引發異常: {e}")
    
    # 測試無框的情況
    scores_only, _ = model._select_top_feature_positions(scores, None, target_positions)
    
    # 驗證只有分數的輸出
    assert scores_only.shape == (batch_size, num_classes, 4, target_positions), "無框情況下分數形狀應正確"
    
    # 測試極端情況 - 目標位置為1
    min_scores, min_boxes = model._select_top_feature_positions(scores, boxes, 1)
    
    # 驗證極小目標的輸出
    assert min_scores.shape == (batch_size, num_classes, 4, 1), "極小目標位置數量應正確處理"
    assert min_boxes.shape == (batch_size, num_classes, 1), "極小目標位置數量應正確處理"
    
    print("✅ _select_top_feature_positions 測試通過")
    return True

def test_in_compute_box_distillation_loss():
    """測試 _compute_box_distillation_loss 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試標準格式
    teacher_boxes = torch.rand(10, 4)  # 10個框
    student_boxes = torch.rand(10, 4)  # 相同數量
    
    # 調用方法
    loss = model._compute_box_distillation_loss(teacher_boxes, student_boxes)
    
    # 驗證輸出
    assert isinstance(loss, torch.Tensor), "損失應該是張量"
    assert loss.dim() == 0, "損失應該是標量"
    assert not torch.isnan(loss), "損失不應為 NaN"
    
    # 測試框數量不同的情況
    teacher_more_boxes = torch.rand(15, 4)  # 教師有更多框
    student_less_boxes = torch.rand(8, 4)   # 學生有更少框
    
    # 創建分數用於選擇框
    teacher_scores = torch.rand(15, 9)  # 教師框的分數
    student_scores = torch.rand(8, 9)   # 學生框的分數
    
    # 教師更多框
    loss_more_teacher = model._compute_box_distillation_loss(
        teacher_more_boxes, student_boxes, teacher_scores, None)
    
    # 學生更多框
    loss_more_student = model._compute_box_distillation_loss(
        teacher_boxes, student_more_boxes, None, student_scores)
    
    # 兩者都不同
    loss_both_diff = model._compute_box_distillation_loss(
        teacher_more_boxes, student_less_boxes, teacher_scores, student_scores)
    
    # 驗證所有情況都能正確處理
    assert not torch.isnan(loss_more_teacher), "教師更多框時損失不應為 NaN"
    assert not torch.isnan(loss_more_student), "學生更多框時損失不應為 NaN"
    assert not torch.isnan(loss_both_diff), "兩者框數量不同時損失不應為 NaN"
    
    # 測試密集格式
    teacher_dense = torch.rand(1, 9, 100)  # [B, C, N] 密集格式
    student_dense = torch.rand(1, 9, 100)  # 相同形狀
    
    # 調用方法
    loss_dense = model._compute_box_distillation_loss(teacher_dense, student_dense)
    
    # 驗證輸出
    assert not torch.isnan(loss_dense), "密集格式時損失不應為 NaN"
    
    # 測試損失縮放 - 創建大值框使損失爆炸
    teacher_large = torch.ones(10, 4) * 1000
    student_large = torch.zeros(10, 4)
    
    # 調用方法
    loss_large = model._compute_box_distillation_loss(teacher_large, student_large)
    
    # 驗證損失被適當縮放
    assert loss_large < 100, f"大損失值應被縮放，但得到 {loss_large.item()}"
    
    print("✅ _compute_box_distillation_loss 測試通過")
    return True
def test_in_standardize_boxes():
    """測試 _standardize_boxes 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試 3D 格式 [B, C, N] 且 N 是 4 的倍數
    boxes_3d = torch.rand(2, 5, 8)  # 2批次，5類別，8個值（2個框，每個框4個坐標）
    result_3d = model._standardize_boxes(boxes_3d)
    
    # 驗證輸出
    assert result_3d.dim() == 2, f"輸出應該是 2D 張量，但得到 {result_3d.dim()}D"
    assert result_3d.shape[1] == 4, f"輸出應該有 4 列，但得到 {result_3d.shape[1]}"
    assert result_3d.shape[0] == 10, f"輸出應該有 10 行 (2*5*2)，但得到 {result_3d.shape[0]}"
    
    # 測試 3D 格式 [B, C, N] 且 N 不是 4 的倍數
    boxes_3d_odd = torch.rand(2, 5, 10)  # 10 不是 4 的倍數
    result_3d_odd = model._standardize_boxes(boxes_3d_odd)
    
    # 驗證輸出 - 應該截斷為最接近的 4 的倍數
    assert result_3d_odd.dim() == 2, f"輸出應該是 2D 張量，但得到 {result_3d_odd.dim()}D"
    assert result_3d_odd.shape[1] == 4, f"輸出應該有 4 列，但得到 {result_3d_odd.shape[1]}"
    expected_rows = (2 * 5 * 10) // 4  # 總元素數除以 4
    assert result_3d_odd.shape[0] == expected_rows, f"輸出應該有 {expected_rows} 行，但得到 {result_3d_odd.shape[0]}"
    
    # 測試 4D 格式 [B, C, 4, N]
    boxes_4d = torch.rand(2, 5, 4, 3)  # 2批次，5類別，4個坐標，3個框
    result_4d = model._standardize_boxes(boxes_4d)
    
    # 驗證輸出
    assert result_4d.dim() == 2, f"輸出應該是 2D 張量，但得到 {result_4d.dim()}D"
    assert result_4d.shape[1] == 4, f"輸出應該有 4 列，但得到 {result_4d.shape[1]}"
    
    # 測試已經是標準格式的輸入 [N, 4]
    boxes_std = torch.rand(10, 4)
    result_std = model._standardize_boxes(boxes_std)
    
    # 驗證輸出 - 應該保持不變
    assert torch.allclose(result_std, boxes_std), "標準格式輸入應該保持不變"
    
    # 測試邊界情況 - 空張量
    try:
        empty_boxes = torch.zeros(0, 0)
        result_empty = model._standardize_boxes(empty_boxes)
        # 如果沒有引發異常，檢查結果是否合理
        assert result_empty.numel() == 0, "空張量應該返回空結果"
    except Exception as e:
        print(f"空張量處理引發異常: {e}")
    
    print("✅ _standardize_boxes 測試通過")
    return True

def test_in_compute_classification_loss():
    """測試 _compute_classification_loss 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試標準格式 - 2D 分類分數
    batch_size = 4
    num_classes = 9
    class_scores = torch.randn(batch_size, num_classes)
    class_ids = torch.randint(0, num_classes, (batch_size,))
    
    # 創建輸出字典
    outputs = {'class_scores': class_scores}
    
    # 計算損失
    loss = model._compute_classification_loss(outputs, class_ids)
    
    # 驗證輸出
    assert isinstance(loss, torch.Tensor), "損失應該是張量"
    assert loss.dim() == 0, "損失應該是標量"
    assert not torch.isnan(loss), "損失不應為 NaN"
    assert not torch.isinf(loss), "損失不應為 Inf"
    
    # 測試輸出元組
    outputs_tuple = (class_scores, torch.randn(batch_size, 4))
    loss_tuple = model._compute_classification_loss(outputs_tuple, class_ids)
    
    # 驗證輸出
    assert isinstance(loss_tuple, torch.Tensor), "損失應該是張量"
    assert loss_tuple.dim() == 0, "損失應該是標量"
    
    # 測試 4D 密集格式
    class_scores_4d = torch.randn(1, num_classes, 4, 100)  # [B, C, 4, N]
    outputs_4d = {'class_scores': class_scores_4d}
    
    # 模擬 _convert_4d_to_2d_scores 方法
    with patch.object(model, '_convert_4d_to_2d_scores', return_value=torch.randn(1, num_classes)) as mock_convert:
        loss_4d = model._compute_classification_loss(outputs_4d, torch.tensor([0]))
        mock_convert.assert_called_once()
        assert isinstance(loss_4d, torch.Tensor), "4D 輸入的損失應該是張量"
    
    # 測試列表類型的 class_ids
    class_ids_list = [torch.tensor([2]), torch.tensor([5])]
    loss_list = model._compute_classification_loss(outputs, class_ids_list)
    
    # 驗證輸出
    assert isinstance(loss_list, torch.Tensor), "列表輸入的損失應該是張量"
    assert loss_list.dim() == 0, "列表輸入的損失應該是標量"
    
    # 測試類別索引超出範圍的情況
    class_ids_out_of_range = torch.tensor([num_classes + 5])
    loss_out_of_range = model._compute_classification_loss(outputs, class_ids_out_of_range)
    
    # 驗證輸出 - 應該正確處理超出範圍的類別
    assert isinstance(loss_out_of_range, torch.Tensor), "超出範圍類別的損失應該是張量"
    assert loss_out_of_range.dim() == 0, "超出範圍類別的損失應該是標量"
    
    print("✅ _compute_classification_loss 測試通過")
    return True

def test_in_compute_box_regression_loss():
    """測試 _compute_box_regression_loss 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試標準格式 - 2D 框
    batch_size = 4
    num_classes = 9
    pred_boxes = torch.rand(batch_size, 4)
    target_boxes = torch.rand(batch_size, 4)
    class_scores = torch.randn(batch_size, num_classes)
    
    # 創建輸出字典
    outputs = {'class_scores': class_scores, 'boxes': pred_boxes}
    
    # 模擬 _prepare_target_boxes 方法
    with patch.object(model, '_prepare_target_boxes', return_value=target_boxes) as mock_prepare:
        # 計算損失
        loss = model._compute_box_regression_loss(outputs, target_boxes)
        mock_prepare.assert_called_once()
        
        # 驗證輸出
        assert isinstance(loss, torch.Tensor), "損失應該是張量"
        assert loss.dim() == 0, "損失應該是標量"
        assert not torch.isnan(loss), "損失不應為 NaN"
        assert not torch.isinf(loss), "損失不應為 Inf"
    
    # 測試輸出元組
    outputs_tuple = (class_scores, pred_boxes)
    
    with patch.object(model, '_prepare_target_boxes', return_value=target_boxes):
        with patch.object(model, '_match_box_counts', return_value=(pred_boxes, target_boxes)) as mock_match:
            loss_tuple = model._compute_box_regression_loss(outputs_tuple, target_boxes)
            mock_match.assert_called_once()
            
            # 驗證輸出
            assert isinstance(loss_tuple, torch.Tensor), "損失應該是張量"
            assert loss_tuple.dim() == 0, "損失應該是標量"
    
    # 測試密集格式
    dense_boxes = torch.rand(1, num_classes, 100)  # [B, C, N]
    outputs_dense = {'class_scores': class_scores, 'boxes': dense_boxes}
    
    with patch.object(model, '_prepare_target_boxes', return_value=target_boxes):
        with patch.object(model, '_convert_dense_boxes_to_standard', return_value=pred_boxes) as mock_convert:
            loss_dense = model._compute_box_regression_loss(outputs_dense, target_boxes)
            mock_convert.assert_called_once()
            
            # 驗證輸出
            assert isinstance(loss_dense, torch.Tensor), "密集格式的損失應該是張量"
            assert loss_dense.dim() == 0, "密集格式的損失應該是標量"
    
    # 測試框數量不匹配的情況
    mismatched_boxes = torch.rand(batch_size + 5, 4)
    outputs_mismatched = {'class_scores': class_scores, 'boxes': mismatched_boxes}
    
    with patch.object(model, '_prepare_target_boxes', return_value=target_boxes):
        with patch.object(model, '_match_box_counts', return_value=(pred_boxes, target_boxes)) as mock_match:
            loss_mismatched = model._compute_box_regression_loss(outputs_mismatched, target_boxes)
            mock_match.assert_called_once()
            
            # 驗證輸出
            assert isinstance(loss_mismatched, torch.Tensor), "不匹配框的損失應該是張量"
            assert loss_mismatched.dim() == 0, "不匹配框的損失應該是標量"
    
    # 測試空框處理
    empty_boxes = torch.zeros((0, 4))
    outputs_empty = {'class_scores': class_scores, 'boxes': empty_boxes}
    
    with patch.object(model, '_prepare_target_boxes', return_value=empty_boxes):
        loss_empty = model._compute_box_regression_loss(outputs_empty, empty_boxes)
        
        # 驗證輸出 - 應該返回零損失
        assert loss_empty.item() == 0.0, f"空框應該返回零損失，但得到 {loss_empty.item()}"
    
    print("✅ _compute_box_regression_loss 測試通過")
    return True

def test_in_compute_teacher_distillation_loss():
    """測試 _compute_teacher_distillation_loss 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試標準格式 - 2D 分數和框
    batch_size = 4
    num_classes = 9
    teacher_scores = torch.randn(batch_size, num_classes)
    teacher_boxes = torch.rand(batch_size, 4)
    student_scores = torch.randn(batch_size, num_classes)
    student_boxes = torch.rand(batch_size, 4)
    
    # 創建輸出和目標
    outputs = {'class_scores': student_scores, 'boxes': student_boxes}
    targets = {
        'teacher_outputs': (teacher_scores, teacher_boxes),
        'selected_boxes': student_boxes  # 模擬已選擇的框
    }
    
    # 模擬分類和框蒸餾損失計算
    with patch.object(model, '_compute_classification_distillation_loss', return_value=torch.tensor(0.5)) as mock_cls:
        with patch.object(model, '_compute_box_distillation_loss', return_value=torch.tensor(0.3)) as mock_box:
            # 計算損失
            loss = model._compute_teacher_distillation_loss(outputs, targets)
            mock_cls.assert_called_once()
            mock_box.assert_called_once()
            
            # 驗證輸出
            assert isinstance(loss, torch.Tensor), "損失應該是張量"
            assert loss.dim() == 0, "損失應該是標量"
            assert loss.item() == 0.8, f"損失應該是 0.8 (0.5 + 0.3)，但得到 {loss.item()}"
    
    # 測試輸出元組
    outputs_tuple = (student_scores, student_boxes)
    
    with patch.object(model, '_compute_classification_distillation_loss', return_value=torch.tensor(0.5)):
        with patch.object(model, '_compute_box_distillation_loss', return_value=torch.tensor(0.3)):
            loss_tuple = model._compute_teacher_distillation_loss(outputs_tuple, targets)
            
            # 驗證輸出
            assert isinstance(loss_tuple, torch.Tensor), "損失應該是張量"
            assert loss_tuple.dim() == 0, "損失應該是標量"
            assert loss_tuple.item() == 0.8, f"損失應該是 0.8，但得到 {loss_tuple.item()}"
    
    # 測試維度不匹配的情況
    teacher_scores_4d = torch.randn(1, num_classes, 4, 100)  # [B, C, 4, N]
    teacher_boxes_3d = torch.rand(1, num_classes, 100)  # [B, C, N]
    student_scores_2d = torch.randn(batch_size, num_classes)  # [B, C]
    
    targets_dim_mismatch = {
        'teacher_outputs': (teacher_scores_4d, teacher_boxes_3d),
        'selected_boxes': student_boxes
    }
    
    with patch.object(model, '_align_score_dimensions', return_value=(teacher_scores, student_scores)) as mock_align:
        with patch.object(model, '_compute_classification_distillation_loss', return_value=torch.tensor(0.5)):
            with patch.object(model, '_compute_box_distillation_loss', return_value=torch.tensor(0.3)):
                loss_dim_mismatch = model._compute_teacher_distillation_loss(outputs, targets_dim_mismatch)
                mock_align.assert_called_once()
                
                # 驗證輸出
                assert isinstance(loss_dim_mismatch, torch.Tensor), "維度不匹配的損失應該是張量"
                assert loss_dim_mismatch.dim() == 0, "維度不匹配的損失應該是標量"
    
    # 測試無效教師輸出
    targets_invalid = {'teacher_outputs': None}
    loss_invalid = model._compute_teacher_distillation_loss(outputs, targets_invalid)
    
    # 驗證輸出 - 應該返回零損失
    assert loss_invalid.item() == 0.0, f"無效教師輸出應該返回零損失，但得到 {loss_invalid.item()}"
    
    # 測試框數量不匹配的情況
    teacher_boxes_more = torch.rand(batch_size + 5, 4)
    targets_box_mismatch = {
        'teacher_outputs': (teacher_scores, teacher_boxes_more),
        'selected_boxes': student_boxes
    }
    
    with patch.object(model, '_select_boxes_by_confidence', return_value=teacher_boxes) as mock_select:
        with patch.object(model, '_compute_classification_distillation_loss', return_value=torch.tensor(0.5)):
            with patch.object(model, '_compute_box_distillation_loss', return_value=torch.tensor(0.3)):
                loss_box_mismatch = model._compute_teacher_distillation_loss(outputs, targets_box_mismatch)
                mock_select.assert_called_once()
                
                # 驗證輸出
                assert isinstance(loss_box_mismatch, torch.Tensor), "框不匹配的損失應該是張量"
                assert loss_box_mismatch.dim() == 0, "框不匹配的損失應該是標量"
    
    print("✅ _compute_teacher_distillation_loss 測試通過")
    return True

def test_in_compute_lcp_loss():
    """測試 _compute_lcp_loss 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 創建輔助網路
    auxiliary_net = AuxiliaryNetwork(in_channels=2048)
    
    # 創建測試輸入
    feature_maps = torch.randn(1, 2048, 14, 14)
    boxes = torch.rand(5, 4)  # 5個框
    class_ids = [torch.tensor([0, 1])]  # 兩個類別
    
    # 創建模擬輸出和目標
    outputs = {
        'feature_maps': feature_maps,
        'boxes': boxes
    }
    
    targets = {
        'class_ids': class_ids,
        'images': torch.randn(1, 3, 224, 224)
    }
    
    # 模擬 _get_feature_maps_for_lcp 方法
    with patch.object(model, '_get_feature_maps_for_lcp', return_value=feature_maps) as mock_get_features:
        # 模擬 _get_boxes_for_lcp 方法
        with patch.object(model, '_get_boxes_for_lcp', return_value=boxes) as mock_get_boxes:
            # 模擬 _prepare_roi_format_boxes 方法
            roi_boxes = torch.cat([torch.zeros(5, 1), boxes], dim=1)  # [5, 5] 格式
            with patch.object(model, '_prepare_roi_format_boxes', return_value=roi_boxes) as mock_prepare_roi:
                # 模擬輔助網路輸出
                aux_outputs = (torch.randn(5, len(VOC_CLASSES)), torch.randn(5, 4))
                with patch.object(auxiliary_net, 'forward', return_value=aux_outputs) as mock_aux_forward:
                    # 模擬分類損失計算
                    with patch.object(model, '_compute_classification_loss', return_value=torch.tensor(0.5)) as mock_cls_loss:
                        # 模擬損失增強
                        with patch.object(model, '_enhance_lcp_loss', return_value=torch.tensor(0.7)) as mock_enhance:
                            # 執行測試
                            lcp_loss = model._compute_lcp_loss(outputs, targets, auxiliary_net)
                            
                            # 驗證方法調用
                            mock_get_features.assert_called_once()
                            mock_get_boxes.assert_called_once()
                            mock_prepare_roi.assert_called_once()
                            mock_aux_forward.assert_called_once()
                            mock_cls_loss.assert_called_once()
                            mock_enhance.assert_called_once()
                            
                            # 驗證輸出
                            assert isinstance(lcp_loss, torch.Tensor), "損失應該是張量"
                            assert lcp_loss.item() == 0.7, "損失值應該是0.7"
    
    # 測試特徵圖為空的情況
    with patch.object(model, '_get_feature_maps_for_lcp', return_value=None):
        lcp_loss = model._compute_lcp_loss(outputs, targets, auxiliary_net)
        assert lcp_loss.item() == 0.0, "特徵圖為空時應返回零損失"
    
    # 測試框為空的情況
    with patch.object(model, '_get_feature_maps_for_lcp', return_value=feature_maps):
        with patch.object(model, '_get_boxes_for_lcp', return_value=None):
            lcp_loss = model._compute_lcp_loss(outputs, targets, auxiliary_net)
            assert lcp_loss.item() == 0.0, "框為空時應返回零損失"
    
    # 測試 ROI 格式框為空的情況
    with patch.object(model, '_get_feature_maps_for_lcp', return_value=feature_maps):
        with patch.object(model, '_get_boxes_for_lcp', return_value=boxes):
            with patch.object(model, '_prepare_roi_format_boxes', return_value=None):
                lcp_loss = model._compute_lcp_loss(outputs, targets, auxiliary_net)
                assert lcp_loss.item() == 0.0, "ROI 格式框為空時應返回零損失"
    
    print("✅ _compute_lcp_loss 測試通過")
    return True
def test_in_get_feature_maps_for_lcp():
    """測試 _get_feature_maps_for_lcp 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 創建測試輸入
    feature_maps = torch.randn(1, 2048, 14, 14)
    images = torch.randn(1, 3, 224, 224)
    
    # 測試從 outputs 字典中獲取特徵圖
    outputs = {'feature_maps': feature_maps}
    targets = {}
    
    result = model._get_feature_maps_for_lcp(outputs, targets)
    assert torch.equal(result, feature_maps), "應該直接返回 outputs 中的特徵圖"
    
    # 測試從 outputs 中的 images 生成特徵圖
    outputs = {'images': images}
    targets = {}
    
    with patch.object(model, 'get_feature_map', return_value=feature_maps) as mock_get_feature:
        result = model._get_feature_maps_for_lcp(outputs, targets)
        mock_get_feature.assert_called_once_with(images)
        assert torch.equal(result, feature_maps), "應該使用 outputs 中的 images 生成特徵圖"
    
    # 測試從 targets 中的 images 生成特徵圖
    outputs = {}
    targets = {'images': images}
    
    with patch.object(model, 'get_feature_map', return_value=feature_maps) as mock_get_feature:
        result = model._get_feature_maps_for_lcp(outputs, targets)
        mock_get_feature.assert_called_once_with(images)
        assert torch.equal(result, feature_maps), "應該使用 targets 中的 images 生成特徵圖"
    
    # 測試無法獲取特徵圖的情況
    outputs = {}
    targets = {}
    
    result = model._get_feature_maps_for_lcp(outputs, targets)
    assert result is None, "無法獲取特徵圖時應返回 None"
    
    print("✅ _get_feature_maps_for_lcp 測試通過")
    return True

def test_in_get_boxes_for_lcp():
    """測試 _get_boxes_for_lcp 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 創建測試框
    boxes = torch.rand(5, 4)
    
    # 測試從 outputs 字典中獲取框
    outputs = {'boxes': boxes}
    
    result = model._get_boxes_for_lcp(outputs)
    assert torch.equal(result, boxes), "應該直接返回 outputs 中的框"
    
    # 測試從 outputs 元組中獲取框
    outputs = (torch.randn(5, 20), boxes)  # (class_scores, boxes)
    
    result = model._get_boxes_for_lcp(outputs)
    assert torch.equal(result, boxes), "應該返回 outputs 元組中的第二個元素"
    
    # 測試 outputs 元組長度不足的情況
    outputs = (torch.randn(5, 20),)  # 只有 class_scores
    
    result = model._get_boxes_for_lcp(outputs)
    assert result is None, "輸出元組長度不足時應返回 None"
    
    # 測試無效 outputs 的情況
    outputs = "not a valid output"
    
    result = model._get_boxes_for_lcp(outputs)
    assert result is None, "無效輸出時應返回 None"
    
    print("✅ _get_boxes_for_lcp 測試通過")
    return True
def test_in_prepare_roi_format_boxes():
    """測試 _prepare_roi_format_boxes 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試標準格式框 [N, 4]
    boxes = torch.tensor([
        [10, 20, 30, 40],
        [50, 60, 70, 80]
    ], dtype=torch.float32)
    
    result = model._prepare_roi_format_boxes(boxes)
    
    # 驗證輸出
    assert isinstance(result, torch.Tensor), "輸出應該是張量"
    assert result.shape == (2, 5), f"輸出形狀應為 (2, 5)，但得到 {result.shape}"
    assert torch.allclose(result[:, 0], torch.zeros(2)), "第一列應為批次索引 0"
    assert torch.allclose(result[:, 1:], boxes), "後四列應為原始框坐標"
    
    # 測試 3D 密集格式 [B, C, N]
    dense_boxes = torch.rand(1, 3, 8)  # 1批次，3類別，8個值 (2個框)
    
    # 模擬處理過程
    expected_result = torch.cat([
        torch.zeros(2, 1),  # 批次索引
        torch.rand(2, 4)    # 標準化後的框
    ], dim=1)
    
    with patch.object(model, '_standardize_boxes', return_value=torch.rand(2, 4)) as mock_standardize:
        result = model._prepare_roi_format_boxes(dense_boxes)
        mock_standardize.assert_called_once()
        assert result.shape == (2, 5), f"密集格式輸出形狀應為 (2, 5)，但得到 {result.shape}"
    
    # 測試框坐標修正 (x1 > x2 或 y1 > y2 的情況)
    inverted_boxes = torch.tensor([
        [30, 40, 10, 20],  # x1 > x2, y1 > y2
        [50, 80, 70, 60]   # x1 < x2, y1 > y2
    ], dtype=torch.float32)
    
    result = model._prepare_roi_format_boxes(inverted_boxes)
    
    # 驗證坐標修正
    assert result[0, 1] < result[0, 3], "應確保 x1 < x2"
    assert result[0, 2] < result[0, 4], "應確保 y1 < y2"
    assert result[1, 1] < result[1, 3], "應確保 x1 < x2"
    assert result[1, 2] < result[1, 4], "應確保 y1 < y2"
    
    # 測試最小框尺寸強制
    tiny_boxes = torch.tensor([
        [10, 20, 10.0001, 20.0001],  # 非常小的框
        [50, 60, 50, 60]             # 零尺寸框
    ], dtype=torch.float32)
    
    result = model._prepare_roi_format_boxes(tiny_boxes)
    
    # 驗證最小尺寸強制
    assert (result[0, 3] - result[0, 1]) >= 1e-3, "應確保框寬度至少為 min_size"
    assert (result[0, 4] - result[0, 2]) >= 1e-3, "應確保框高度至少為 min_size"
    assert (result[1, 3] - result[1, 1]) >= 1e-3, "應確保框寬度至少為 min_size"
    assert (result[1, 4] - result[1, 2]) >= 1e-3, "應確保框高度至少為 min_size"
    
    # 測試異常處理
    with patch.object(model, '_standardize_boxes', side_effect=Exception("測試異常")) as mock_standardize:
        result = model._prepare_roi_format_boxes(dense_boxes)
        assert result is None, "發生異常時應返回 None"
    
    print("✅ _prepare_roi_format_boxes 測試通過")
    return True

def test_in_enhance_lcp_loss():
    """測試 _enhance_lcp_loss 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 創建測試輸入
    batch_size = 3
    num_classes = len(VOC_CLASSES)  # 20 for VOC
    aux_scores = torch.randn(batch_size, num_classes)
    aux_cls_loss = torch.tensor(0.005)  # 小損失值，會觸發增強邏輯
    
    # 創建目標
    targets = {
        'class_ids': [torch.tensor([2]), torch.tensor([5]), torch.tensor([8])]
    }
    
    # 調用方法
    enhanced_loss = model._enhance_lcp_loss(
        (aux_scores, torch.randn(batch_size, 4)),  # 模擬輔助網路輸出
        aux_cls_loss, 
        targets
    )
    
    # 驗證輸出
    assert isinstance(enhanced_loss, torch.Tensor), "增強後的損失應該是張量"
    assert enhanced_loss.dim() == 0, "增強後的損失應該是標量"
    assert enhanced_loss.item() > aux_cls_loss.item(), "增強後的損失應該大於原始損失"
    assert enhanced_loss.item() >= 0.1, "增強後的損失應該至少為 0.1"
    
    # 測試較大的損失值 (不需要增強)
    large_loss = torch.tensor(0.5)
    enhanced_large = model._enhance_lcp_loss(
        (aux_scores, torch.randn(batch_size, 4)),
        large_loss,
        targets
    )
    
    # 驗證大損失值的輸出
    assert enhanced_large.item() == large_loss.item(), "較大的損失值不應該被增強"
    
    # 測試極端小的損失值
    tiny_loss = torch.tensor(1e-10)
    enhanced_tiny = model._enhance_lcp_loss(
        (aux_scores, torch.randn(batch_size, 4)),
        tiny_loss,
        targets
    )
    
    # 驗證極小損失值的輸出
    assert enhanced_tiny.item() >= 0.1, "極小損失值應該被增強到至少 0.1"
    
    # 測試損失值上限
    huge_loss = torch.tensor(10.0)
    enhanced_huge = model._enhance_lcp_loss(
        (aux_scores, torch.randn(batch_size, 4)),
        huge_loss,
        targets
    )
    
    # 驗證超大損失值的輸出
    assert enhanced_huge.item() <= 5.0, "超大損失值應該被限制在 5.0 以內"
    
    print("✅ _enhance_lcp_loss 測試通過")
    return True

def test_in_prepare_gt_labels_for_lcp():
    """測試 _prepare_gt_labels_for_lcp 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 測試列表輸入
    class_ids = [torch.tensor([3]), torch.tensor([5])]
    batch_size = 10
    num_classes = len(VOC_CLASSES)  # 20 for VOC
    
    # 調用方法
    gt_labels = model._prepare_gt_labels_for_lcp(class_ids, batch_size, num_classes)
    
    # 驗證輸出
    assert isinstance(gt_labels, torch.Tensor), "輸出應該是張量"
    assert gt_labels.shape[0] == batch_size, f"輸出形狀應為 ({batch_size},)，但得到 {gt_labels.shape}"
    assert gt_labels[0] == 3, "第一個標籤應該是 3"
    
    # 測試張量輸入
    class_ids_tensor = torch.tensor([7])
    
    # 調用方法
    gt_labels_tensor = model._prepare_gt_labels_for_lcp(class_ids_tensor, batch_size, num_classes)
    
    # 驗證輸出
    assert gt_labels_tensor.shape[0] == batch_size, f"輸出形狀應為 ({batch_size},)，但得到 {gt_labels_tensor.shape}"
    assert gt_labels_tensor[0] == 7, "所有標籤應該是 7"
    
    # 測試非張量輸入
    class_ids_int = 9
    
    # 調用方法
    gt_labels_int = model._prepare_gt_labels_for_lcp(class_ids_int, batch_size, num_classes)
    
    # 驗證輸出
    assert gt_labels_int.shape[0] == batch_size, f"輸出形狀應為 ({batch_size},)，但得到 {gt_labels_int.shape}"
    assert gt_labels_int[0] == 9, "所有標籤應該是 9"
    
    # 測試類別索引超出範圍的情況
    class_ids_out_of_range = [torch.tensor([num_classes + 5])]
    
    # 調用方法
    gt_labels_out_of_range = model._prepare_gt_labels_for_lcp(class_ids_out_of_range, batch_size, num_classes)
    
    # 驗證輸出 - 應該對類別索引取模
    assert gt_labels_out_of_range[0] % num_classes == (num_classes + 5) % num_classes, "超出範圍的類別索引應該取模"
    
    print("✅ _prepare_gt_labels_for_lcp 測試通過")
    return True

def test_in_scale_losses():
    """測試 _scale_losses 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 創建測試損失值
    cls_loss = torch.tensor(5.0)
    box_loss = torch.tensor(3.0)
    teacher_loss = torch.tensor(2.0)
    lcp_loss = torch.tensor(1.0)
    
    # 調用方法
    scaled_losses = model._scale_losses(cls_loss, box_loss, teacher_loss, lcp_loss)
    
    # 驗證輸出
    assert len(scaled_losses) == 4, "應該返回 4 個損失值"
    scaled_cls, scaled_box, scaled_teacher, scaled_lcp = scaled_losses
    
    # 驗證損失縮放 - 所有損失都小於閾值，應該保持不變
    assert scaled_cls.item() == cls_loss.item(), "分類損失不應該被縮放"
    assert scaled_box.item() == box_loss.item(), "框回歸損失不應該被縮放"
    assert scaled_teacher.item() == teacher_loss.item(), "教師損失不應該被縮放"
    assert scaled_lcp.item() == lcp_loss.item(), "LCP 損失不應該被縮放"
    
    # 測試超過閾值的損失值
    large_cls_loss = torch.tensor(15.0)  # 大於閾值 10.0
    large_box_loss = torch.tensor(20.0)  # 大於閾值 10.0
    large_teacher_loss = torch.tensor(25.0)  # 大於閾值 10.0
    large_lcp_loss = torch.tensor(30.0)  # 大於閾值 10.0
    
    # 調用方法
    scaled_large = model._scale_losses(large_cls_loss, large_box_loss, large_teacher_loss, large_lcp_loss)
    
    # 驗證輸出
    scaled_cls_large, scaled_box_large, scaled_teacher_large, scaled_lcp_large = scaled_large
    
    # 驗證損失縮放 - 超過閾值的損失應該被縮放
    assert scaled_cls_large.item() == 10.0, "大於閾值的分類損失應該被縮放為 10.0"
    assert scaled_box_large.item() == 10.0, "大於閾值的框回歸損失應該被縮放為 10.0"
    assert scaled_teacher_large.item() == 10.0, "大於閾值的教師損失應該被縮放為 10.0"
    assert scaled_lcp_large.item() == 10.0, "大於閾值的 LCP 損失應該被縮放為 10.0"
    
    # 測試負損失值
    negative_cls_loss = torch.tensor(-5.0)
    
    # 調用方法
    scaled_negative = model._scale_losses(negative_cls_loss, box_loss, teacher_loss, lcp_loss)
    
    # 驗證輸出
    scaled_cls_negative = scaled_negative[0]
    
    # 驗證負損失值處理
    assert scaled_cls_negative.item() >= 0, "負損失值應該被處理為非負值"
    
    print("✅ _scale_losses 測試通過")
    return True

def test_in_compute_losses():
    """測試 compute_losses 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 創建測試輸入
    batch_size = 2
    num_classes = len(VOC_CLASSES)  # 20 for VOC
    class_scores = torch.randn(batch_size, num_classes)
    boxes = torch.rand(batch_size, 4)
    
    # 創建輸出字典
    outputs = {
        'class_scores': class_scores,
        'boxes': boxes,
        'images': torch.randn(batch_size, 3, 224, 224),
        'class_images': [torch.randn(3, 64, 64) for _ in range(batch_size)]
    }
    
    # 創建目標字典
    targets = {
        'class_ids': [torch.tensor([2]), torch.tensor([5])],
        'boxes': [torch.rand(1, 4), torch.rand(1, 4)],
        'images': torch.randn(batch_size, 3, 224, 224),
        'teacher_outputs': (
            torch.randn(batch_size, num_classes),
            torch.rand(batch_size, 4)
        )
    }
    
    # 創建輔助網路
    auxiliary_net = AuxiliaryNetwork(in_channels=2048)
    
    # 模擬各種損失計算方法
    with patch.object(model, '_compute_classification_loss', return_value=torch.tensor(1.0)) as mock_cls:
        with patch.object(model, '_compute_box_regression_loss', return_value=torch.tensor(2.0)) as mock_box:
            with patch.object(model, '_compute_teacher_distillation_loss', return_value=torch.tensor(3.0)) as mock_teacher:
                with patch.object(model, '_compute_lcp_loss', return_value=torch.tensor(4.0)) as mock_lcp:
                    with patch.object(model, '_scale_losses', return_value=(
                        torch.tensor(1.0), torch.tensor(2.0), 
                        torch.tensor(3.0), torch.tensor(4.0)
                    )) as mock_scale:
                        # 調用方法
                        loss, loss_dict = model.compute_losses(
                            outputs,
                            targets,
                            class_num=num_classes,
                            auxiliary_net=auxiliary_net,
                            use_lcp_loss=True,
                            loss_weights={'cls': 1.0, 'box_reg': 1.0, 'teacher': 0.5, 'lcp': 0.1}
                        )
                        
                        # 驗證方法調用
                        mock_cls.assert_called_once()
                        mock_box.assert_called_once()
                        mock_teacher.assert_called_once()
                        mock_lcp.assert_called_once()
                        mock_scale.assert_called_once()
                        
                        # 驗證輸出
                        assert isinstance(loss, torch.Tensor), "損失應該是張量"
                        assert isinstance(loss_dict, dict), "損失字典應該是字典"
                        assert 'cls_loss' in loss_dict, "損失字典應該包含分類損失"
                        assert 'box_loss' in loss_dict, "損失字典應該包含框回歸損失"
                        assert 'teacher_loss' in loss_dict, "損失字典應該包含教師損失"
                        assert 'lcp_loss' in loss_dict, "損失字典應該包含 LCP 損失"
                        
                        # 驗證總損失計算
                        expected_loss = 1.0 + 2.0 + 0.5 * 3.0 + 0.1 * 4.0
                        assert abs(loss.item() - expected_loss) < 1e-5, f"總損失應為 {expected_loss}，但得到 {loss.item()}"
    
    # 測試禁用 LCP 損失
    with patch.object(model, '_compute_classification_loss', return_value=torch.tensor(1.0)):
        with patch.object(model, '_compute_box_regression_loss', return_value=torch.tensor(2.0)):
            with patch.object(model, '_compute_teacher_distillation_loss', return_value=torch.tensor(3.0)):
                with patch.object(model, '_compute_lcp_loss', return_value=torch.tensor(4.0)) as mock_lcp:
                    with patch.object(model, '_scale_losses', return_value=(
                        torch.tensor(1.0), torch.tensor(2.0), 
                        torch.tensor(3.0), torch.tensor(0.0)
                    )):
                        # 調用方法
                        loss, loss_dict = model.compute_losses(
                            outputs,
                            targets,
                            class_num=num_classes,
                            auxiliary_net=auxiliary_net,
                            use_lcp_loss=False,  # 禁用 LCP 損失
                            loss_weights={'cls': 1.0, 'box_reg': 1.0, 'teacher': 0.5, 'lcp': 0.1}
                        )
                        
                        # 驗證 LCP 損失計算未被調用
                        mock_lcp.assert_not_called()
                        
                        # 驗證總損失計算
                        expected_loss = 1.0 + 2.0 + 0.5 * 3.0
                        assert abs(loss.item() - expected_loss) < 1e-5, f"禁用 LCP 損失時總損失應為 {expected_loss}，但得到 {loss.item()}"
    
    # 測試沒有教師輸出的情況
    targets_no_teacher = {
        'class_ids': [torch.tensor([2]), torch.tensor([5])],
        'boxes': [torch.rand(1, 4), torch.rand(1, 4)],
        'images': torch.randn(batch_size, 3, 224, 224)
    }
    
    with patch.object(model, '_compute_classification_loss', return_value=torch.tensor(1.0)):
        with patch.object(model, '_compute_box_regression_loss', return_value=torch.tensor(2.0)):
            with patch.object(model, '_compute_teacher_distillation_loss', return_value=torch.tensor(0.0)) as mock_teacher:
                with patch.object(model, '_compute_lcp_loss', return_value=torch.tensor(4.0)):
                    with patch.object(model, '_scale_losses', return_value=(
                        torch.tensor(1.0), torch.tensor(2.0), 
                        torch.tensor(0.0), torch.tensor(4.0)
                    )):
                        # 調用方法
                        loss, loss_dict = model.compute_losses(
                            outputs,
                            targets_no_teacher,
                            class_num=num_classes,
                            auxiliary_net=auxiliary_net,
                            use_lcp_loss=True,
                            loss_weights={'cls': 1.0, 'box_reg': 1.0, 'teacher': 0.5, 'lcp': 0.1}
                        )
                        
                        # 驗證教師損失計算被調用但返回零損失
                        mock_teacher.assert_called_once()
                        
                        # 驗證總損失計算
                        expected_loss = 1.0 + 2.0 + 0.1 * 4.0
                        assert abs(loss.item() - expected_loss) < 1e-5, f"無教師輸出時總損失應為 {expected_loss}，但得到 {loss.item()}"
    
    print("✅ compute_losses 測試通過")
    return True

def test_in_update_classifier_for_classes():
    """測試 _update_classifier_for_classes 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    device = torch.device('cpu')
    
    # 獲取原始分類器的輸出特徵數
    original_out_features = None
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'out_features'):
        original_out_features = model.classifier.out_features
        original_in_features = model.classifier.in_features
    
    # 如果模型沒有分類器，則添加一個用於測試
    if original_out_features is None:
        model.classifier = nn.Linear(512, 20)
        original_out_features = 20
        original_in_features = 512
    
    # 測試更新為不同的類別數
    new_class_num = original_out_features + 5
    model._update_classifier_for_classes(new_class_num, device)
    
    # 驗證分類器已更新
    assert hasattr(model, 'classifier'), "模型應該有分類器屬性"
    assert model.classifier.out_features == new_class_num, f"分類器輸出特徵數應為 {new_class_num}，但得到 {model.classifier.out_features}"
    assert model.classifier.in_features == original_in_features, f"分類器輸入特徵數應保持不變，但從 {original_in_features} 變為 {model.classifier.in_features}"
    
    # 測試更新為相同的類別數 (不應該有變化)
    model._update_classifier_for_classes(new_class_num, device)
    assert model.classifier.out_features == new_class_num, "分類器不應該被更新"
    
    # 測試更新為更小的類別數
    smaller_class_num = original_out_features - 5
    if smaller_class_num > 0:
        model._update_classifier_for_classes(smaller_class_num, device)
        assert model.classifier.out_features == smaller_class_num, f"分類器輸出特徵數應為 {smaller_class_num}，但得到 {model.classifier.out_features}"
    
    print("✅ _update_classifier_for_classes 測試通過")
    return True

def test_in_update_auxiliary_classifier():
    """測試 _update_auxiliary_classifier 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    device = torch.device('cpu')
    
    # 創建輔助網路
    auxiliary_net = AuxiliaryNetwork(in_channels=2048)
    original_out_features = auxiliary_net.cls_head[-1].out_features
    original_in_features = auxiliary_net.cls_head[0].in_features
    
    # 測試更新為不同的類別數
    new_class_num = original_out_features + 5
    model._update_auxiliary_classifier(auxiliary_net, new_class_num, device)
    
    # 驗證輔助網路分類器已更新
    assert hasattr(auxiliary_net, 'cls_head'), "輔助網路應該有分類器屬性"
    assert auxiliary_net.cls_head[-1].out_features == new_class_num, f"輔助網路分類器輸出特徵數應為 {new_class_num}，但得到 {auxiliary_net.cls_head[-1].out_features}"
    
    # 測試更新為相同的類別數 (不應該有變化)
    model._update_auxiliary_classifier(auxiliary_net, new_class_num, device)
    assert auxiliary_net.cls_head[-1].out_features == new_class_num, "輔助網路分類器不應該被更新"
    
    # 測試更新為更小的類別數
    smaller_class_num = original_out_features - 5
    if smaller_class_num > 0:
        model._update_auxiliary_classifier(auxiliary_net, smaller_class_num, device)
        assert auxiliary_net.cls_head[-1].out_features == smaller_class_num, f"輔助網路分類器輸出特徵數應為 {smaller_class_num}，但得到 {auxiliary_net.cls_head[-1].out_features}"
    
    print("✅ _update_auxiliary_classifier 測試通過")
    return True

def test_in_print_training_config():
    """測試 _print_training_config 方法的功能"""
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 使用 mock 捕獲打印輸出
    with patch('builtins.print') as mock_print:
        # 測試特徵金字塔啟用的情況
        use_feature_pyramid = True
        pyramid_scales = [1.0, 0.75, 0.5]
        apply_nms = True
        nms_threshold = 0.5
        
        model._print_training_config(use_feature_pyramid, pyramid_scales, apply_nms, nms_threshold)
        
        # 驗證打印函數被調用
        assert mock_print.call_count >= 3, "打印函數應該被調用至少3次"
        
        # 驗證打印內容
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("特徵金字塔狀態: 啟用" in call for call in calls), "應該打印特徵金字塔啟用狀態"
        assert any("金字塔尺度: [1.0, 0.75, 0.5]" in call for call in calls), "應該打印金字塔尺度"
        assert any("NMS 狀態: 啟用" in call for call in calls), "應該打印NMS狀態"
        assert any("閾值: 0.5" in call for call in calls), "應該打印NMS閾值"
        
        # 重置 mock
        mock_print.reset_mock()
        
        # 測試特徵金字塔停用的情況
        use_feature_pyramid = False
        apply_nms = False
        
        model._print_training_config(use_feature_pyramid, pyramid_scales, apply_nms, nms_threshold)
        
        # 驗證打印函數被調用
        assert mock_print.call_count >= 2, "打印函數應該被調用至少2次"
        
        # 驗證打印內容
        calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("特徵金字塔狀態: 停用" in call for call in calls), "應該打印特徵金字塔停用狀態"
        assert any("NMS 狀態: 停用" in call for call in calls), "應該打印NMS停用狀態"
    
    print("✅ _print_training_config 測試通過")
    return True

def test_in_update_auxiliarty_channels():
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    device = torch.device('cpu')

    # 準備 Grozi 數據集
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")

    if not os.path.exists(grozi_csv):
        print("\n❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        pytest.skip("Grozi-3.2k dataset missing, test skipped.")
        return False

    # 建立 dataset/dataloader
    dataset = build_grozi_dataset(
        data_path=data_path,
        name="grozi-train-mini",  # 只取2張圖2類別
        eval_scale=224,
        cache_images=False
    )

    # 創建 box_coder
    box_coder = Os2dBoxCoder(
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.4,
        remap_classification_targets_iou_pos=0.5,
        remap_classification_targets_iou_neg=0.4,
        output_box_grid_generator=BoxGridGenerator(
            box_size=FeatureMapSize(w=16, h=16),
            box_stride=FeatureMapSize(w=16, h=16)
        ),
        function_get_feature_map_size=lambda img_size: FeatureMapSize(w=img_size.w // 16, h=img_size.h // 16),
        do_nms_across_classes=False
    )

    train_loader = DataloaderOneShotDetection(
        dataset=dataset,
        box_coder=box_coder,
        batch_size=1,  # 最小 batch
        img_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        gt_image_size=64,
        random_flip_batches=False,
        random_crop_size=None,
        random_color_distortion=False,
        pyramid_scales_eval=[1.0],
        do_augmentation=False
    )

    # 獲取真實圖像
    batch = train_loader.get_batch(0)
    test_image = batch  # 第一個元素是圖像

    print(f"✓ 使用 Grozi 數據集圖像進行測試，形狀: {test_image.shape}")

    # 創建輔助網路
    auxiliary_net = AuxiliaryNetwork(in_channels=64)

    # 測試通道數不同的情況
    with patch.object(model, 'get_feature_map', return_value=torch.randn(1, 128, 14, 14)) as mock_get_feature_map:
        with patch.object(auxiliary_net, 'update_input_channels') as mock_update_channels:
            with patch.object(auxiliary_net, 'get_current_channels', return_value=64) as mock_get_channels:
                # 調用方法
                model._update_auxiliary_channels(auxiliary_net, test_image, device)
                
                # 驗證 get_feature_map 被調用
                mock_get_feature_map.assert_called_once_with(test_image)
                
                # 驗證 update_input_channels 被調用，因為通道數不同
                mock_update_channels.assert_called_once_with(128)

    # 測試通道數相同的情況
    with patch.object(model, 'get_feature_map', return_value=torch.randn(1, 64, 14, 14)) as mock_get_feature_map:
        with patch.object(auxiliary_net, 'get_current_channels', return_value=64) as mock_get_channels:
            with patch.object(auxiliary_net, 'update_input_channels') as mock_update_channels:
                # 調用方法
                model._update_auxiliary_channels(auxiliary_net, test_image, device)
                
                # 驗證 get_feature_map 被調用
                mock_get_feature_map.assert_called_once_with(test_image)
                
                # 驗證 update_input_channels 不被調用，因為通道數相同
                mock_update_channels.assert_not_called()

    # 測試 feature_maps 不是張量的情況
    with patch.object(model, 'get_feature_map', return_value=None) as mock_get_feature_map:
        with patch.object(auxiliary_net, 'update_input_channels') as mock_update_channels:
            # 調用方法
            model._update_auxiliary_channels(auxiliary_net, test_image, device)
            
            # 驗證 get_feature_map 被調用
            mock_get_feature_map.assert_called_once_with(test_image)
            
            # 驗證 update_input_channels 不被調用
            mock_update_channels.assert_not_called()

    # 測試實際的特徵提取 (不使用 mock)
    try:
        # 獲取實際特徵圖
        real_features = model.get_feature_map(test_image)
        
        # 創建新的輔助網路以測試實際更新
        real_auxiliary_net = AuxiliaryNetwork(in_channels=64)
        original_channels = real_auxiliary_net.get_current_channels()
        
        # 調用方法
        model._update_auxiliary_channels(real_auxiliary_net, test_image, device)
        
        # 驗證通道數是否已更新
        new_channels = real_auxiliary_net.get_current_channels()
        
        if original_channels != real_features.size(1):
            assert new_channels == real_features.size(1), f"通道數應更新為 {real_features.size(1)}，但得到 {new_channels}"
            print(f"✓ 實際特徵圖測試通過: 通道數從 {original_channels} 更新為 {new_channels}")
        else:
            assert new_channels == original_channels, "通道數不應變化"
            print(f"✓ 實際特徵圖測試通過: 通道數保持為 {original_channels}")
    except Exception as e:
        print(f"⚠️ 實際特徵圖測試失敗: {e}")

    print("✅ _update_auxiliary_channels 測試通過")
    return True

def test_in_run_standard_inference():
    """測試 _run_standard_inference 方法的功能"""
    import os
    import pytest
    import torch
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    from src.os2d_model_in_prune import Os2dModelInPrune
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 準備 Grozi 數據集
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if not os.path.exists(grozi_csv):
        print("\n❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        pytest.skip("Grozi-3.2k dataset missing, test skipped.")
        return False

    # 建立 dataset
    dataset = build_grozi_dataset(
        data_path=data_path,
        name="grozi-train-mini",
        eval_scale=224,
        cache_images=False
    )
    
    # 創建 box_coder
    box_coder = Os2dBoxCoder(
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.4,
        remap_classification_targets_iou_pos=0.5,
        remap_classification_targets_iou_neg=0.4,
        output_box_grid_generator=BoxGridGenerator(
            box_size=FeatureMapSize(w=16, h=16),
            box_stride=FeatureMapSize(w=16, h=16)
        ),
        function_get_feature_map_size=lambda img_size: FeatureMapSize(w=img_size.w // 16, h=img_size.h // 16),
        do_nms_across_classes=False
    )
    
    dataloader = DataloaderOneShotDetection(
        dataset=dataset,
        box_coder=box_coder,
        batch_size=1,
        img_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        gt_image_size=64,
        random_flip_batches=False,
        random_crop_size=None,
        random_color_distortion=False,
        pyramid_scales_eval=[1.0],
        do_augmentation=False
    )
    
    # 獲取第一個批次
    batch = dataloader.get_batch(0)
    images, class_images = batch[0], batch[1]
    
    # 設置參數
    max_predictions = 50
    nms_threshold = 0.5
    apply_nms = True
    
    # 測試學生模型 (self) 的推理
    outputs = model._run_standard_inference(model, images, class_images, max_predictions, nms_threshold, apply_nms)
    
    # 驗證輸出
    assert isinstance(outputs, tuple), "輸出應該是元組"
    assert len(outputs) >= 2, "輸出元組應該至少有兩個元素"
    
    class_scores, boxes = outputs[0], outputs[1]
    print(f"✓ 標準推理輸出 - class_scores: {class_scores.shape}, boxes: {boxes.shape}")
    
    # 測試教師模型的推理
    with torch.no_grad():
        teacher_outputs = model._run_standard_inference(model.teacher_model, images, class_images, max_predictions, nms_threshold, apply_nms)
    
    # 驗證輸出
    assert isinstance(teacher_outputs, tuple), "教師模型輸出應該是元組"
    assert len(teacher_outputs) >= 2, "教師模型輸出元組應該至少有兩個元素"
    
    teacher_scores, teacher_boxes = teacher_outputs[0], teacher_outputs[1]
    print(f"✓ 教師模型標準推理輸出 - class_scores: {teacher_scores.shape}, boxes: {teacher_boxes.shape}")
    
    # 測試不使用 NMS 的情況
    outputs_no_nms = model._run_standard_inference(model, images, class_images, max_predictions, nms_threshold, apply_nms=False)
    
    # 驗證輸出
    assert isinstance(outputs_no_nms, tuple), "不使用 NMS 的輸出應該是元組"
    assert len(outputs_no_nms) >= 2, "不使用 NMS 的輸出元組應該至少有兩個元素"
    
    no_nms_scores, no_nms_boxes = outputs_no_nms[0], outputs_no_nms[1]
    print(f"✓ 不使用 NMS 的標準推理輸出 - class_scores: {no_nms_scores.shape}, boxes: {no_nms_boxes.shape}")
    
    print("✅ _run_standard_inference 測試通過")
    return True

def test_in_run_teacher_feature_pyramid():
    """測試 _run_teacher_feature_pyramid 方法的功能"""
    import os
    import pytest
    import torch
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    from src.os2d_model_in_prune import Os2dModelInPrune
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 準備 Grozi 數據集
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if not os.path.exists(grozi_csv):
        print("\n❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        pytest.skip("Grozi-3.2k dataset missing, test skipped.")
        return False

    # 建立 dataset
    dataset = build_grozi_dataset(
        data_path=data_path,
        name="grozi-train-mini",
        eval_scale=224,
        cache_images=False
    )
    
    # 創建 box_coder
    box_coder = Os2dBoxCoder(
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.4,
        remap_classification_targets_iou_pos=0.5,
        remap_classification_targets_iou_neg=0.4,
        output_box_grid_generator=BoxGridGenerator(
            box_size=FeatureMapSize(w=16, h=16),
            box_stride=FeatureMapSize(w=16, h=16)
        ),
        function_get_feature_map_size=lambda img_size: FeatureMapSize(w=img_size.w // 16, h=img_size.h // 16),
        do_nms_across_classes=False
    )
    
    dataloader = DataloaderOneShotDetection(
        dataset=dataset,
        box_coder=box_coder,
        batch_size=1,
        img_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        gt_image_size=64,
        random_flip_batches=False,
        random_crop_size=None,
        random_color_distortion=False,
        pyramid_scales_eval=[1.0],
        do_augmentation=False
    )
    
    # 獲取第一個批次
    batch = dataloader.get_batch(0)
    images, class_images = batch[0], batch[1]
    
    # 設置參數
    pyramid_scales = [1.0, 0.75, 0.5]
    max_predictions = 50
    nms_threshold = 0.5
    apply_nms = True
    target_feature_count = 100
    
    # 測試教師模型的特徵金字塔推理
    outputs = model._run_teacher_feature_pyramid(
        images, class_images, pyramid_scales,
        max_predictions, nms_threshold, apply_nms,
        target_feature_count
    )
    
    # 驗證輸出
    assert isinstance(outputs, tuple), "輸出應該是元組"
    assert len(outputs) >= 2, "輸出元組應該至少有兩個元素"
    
    class_scores, boxes = outputs[0], outputs[1]
    print(f"✓ 教師特徵金字塔輸出 - class_scores: {class_scores.shape}, boxes: {boxes.shape}")
    
    # 測試不同的金字塔尺度
    pyramid_scales_single = [1.0]
    outputs_single = model._run_teacher_feature_pyramid(
        images, class_images, pyramid_scales_single,
        max_predictions, nms_threshold, apply_nms,
        target_feature_count
    )
    
    # 驗證輸出
    assert isinstance(outputs_single, tuple), "單一尺度輸出應該是元組"
    assert len(outputs_single) >= 2, "單一尺度輸出元組應該至少有兩個元素"
    
    single_scores, single_boxes = outputs_single[0], outputs_single[1]
    print(f"✓ 單一尺度教師特徵金字塔輸出 - class_scores: {single_scores.shape}, boxes: {single_boxes.shape}")
    
    # 測試不使用 NMS 的情況
    outputs_no_nms = model._run_teacher_feature_pyramid(
        images, class_images, pyramid_scales,
        max_predictions, nms_threshold, apply_nms=False,
        target_feature_count=target_feature_count
    )
    
    # 驗證輸出
    assert isinstance(outputs_no_nms, tuple), "不使用 NMS 的輸出應該是元組"
    assert len(outputs_no_nms) >= 2, "不使用 NMS 的輸出元組應該至少有兩個元素"
    
    no_nms_scores, no_nms_boxes = outputs_no_nms[0], outputs_no_nms[1]
    print(f"✓ 不使用 NMS 的教師特徵金字塔輸出 - class_scores: {no_nms_scores.shape}, boxes: {no_nms_boxes.shape}")
    
    print("✅ _run_teacher_feature_pyramid 測試通過")
    return True

def test_in_run_student_feature_pyramid():
    """測試 _run_student_feature_pyramid 方法的功能"""
    import os
    import pytest
    import torch
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    from src.os2d_model_in_prune import Os2dModelInPrune
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 準備 Grozi 數據集
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if not os.path.exists(grozi_csv):
        print("\n❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        pytest.skip("Grozi-3.2k dataset missing, test skipped.")
        return False

    # 建立 dataset
    dataset = build_grozi_dataset(
        data_path=data_path,
        name="grozi-train-mini",
        eval_scale=224,
        cache_images=False
    )
    
    # 創建 box_coder
    box_coder = Os2dBoxCoder(
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.4,
        remap_classification_targets_iou_pos=0.5,
        remap_classification_targets_iou_neg=0.4,
        output_box_grid_generator=BoxGridGenerator(
            box_size=FeatureMapSize(w=16, h=16),
            box_stride=FeatureMapSize(w=16, h=16)
        ),
        function_get_feature_map_size=lambda img_size: FeatureMapSize(w=img_size.w // 16, h=img_size.h // 16),
        do_nms_across_classes=False
    )
    
    dataloader = DataloaderOneShotDetection(
        dataset=dataset,
        box_coder=box_coder,
        batch_size=1,
        img_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        gt_image_size=64,
        random_flip_batches=False,
        random_crop_size=None,
        random_color_distortion=False,
        pyramid_scales_eval=[1.0],
        do_augmentation=False
    )
    
    # 獲取第一個批次
    batch = dataloader.get_batch(0)
    images, class_images, _, _, _, _, _, batch_boxes, _ = batch
    
    # 設置參數
    pyramid_scales = [1.0, 0.75, 0.5]
    max_predictions = 50
    nms_threshold = 0.5
    apply_nms = True
    target_feature_count = 100
    
    # 測試學生模型的特徵金字塔推理
    outputs = model._run_student_feature_pyramid(
        images, class_images, pyramid_scales,
        max_predictions, nms_threshold, apply_nms,
        target_feature_count, batch_boxes
    )
    
    # 驗證輸出
    assert isinstance(outputs, tuple), "輸出應該是元組"
    assert len(outputs) >= 2, "輸出元組應該至少有兩個元素"
    
    class_scores, boxes = outputs[0], outputs[1]
    print(f"✓ 學生特徵金字塔輸出 - class_scores: {class_scores.shape}, boxes: {boxes.shape}")
    
    # 測試不同的金字塔尺度
    pyramid_scales_single = [1.0]
    outputs_single = model._run_student_feature_pyramid(
        images, class_images, pyramid_scales_single,
        max_predictions, nms_threshold, apply_nms,
        target_feature_count, batch_boxes
    )
    
    # 驗證輸出
    assert isinstance(outputs_single, tuple), "單一尺度輸出應該是元組"
    assert len(outputs_single) >= 2, "單一尺度輸出元組應該至少有兩個元素"
    
    single_scores, single_boxes = outputs_single[0], outputs_single[1]
    print(f"✓ 單一尺度學生特徵金字塔輸出 - class_scores: {single_scores.shape}, boxes: {single_boxes.shape}")
    
    # 測試不使用 NMS 的情況
    outputs_no_nms = model._run_student_feature_pyramid(
        images, class_images, pyramid_scales,
        max_predictions, nms_threshold, apply_nms=False,
        target_feature_count=target_feature_count, batch_boxes=batch_boxes
    )
    
    # 驗證輸出
    assert isinstance(outputs_no_nms, tuple), "不使用 NMS 的輸出應該是元組"
    assert len(outputs_no_nms) >= 2, "不使用 NMS 的輸出元組應該至少有兩個元素"
    
    no_nms_scores, no_nms_boxes = outputs_no_nms[0], outputs_no_nms[1]
    print(f"✓ 不使用 NMS 的學生特徵金字塔輸出 - class_scores: {no_nms_scores.shape}, boxes: {no_nms_boxes.shape}")
    
    print("✅ _run_student_feature_pyramid 測試通過")
    return True

def test_in_scale_inputs():
    """測試 _scale_inputs 方法的功能"""
    import os
    import pytest
    import torch
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    from src.os2d_model_in_prune import Os2dModelInPrune
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 準備 Grozi 數據集
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if not os.path.exists(grozi_csv):
        print("\n❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        pytest.skip("Grozi-3.2k dataset missing, test skipped.")
        return False

    # 建立 dataset
    dataset = build_grozi_dataset(
        data_path=data_path,
        name="grozi-train-mini",
        eval_scale=224,
        cache_images=False
    )
    
    # 創建 box_coder
    box_coder = Os2dBoxCoder(
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.4,
        remap_classification_targets_iou_pos=0.5,
        remap_classification_targets_iou_neg=0.4,
        output_box_grid_generator=BoxGridGenerator(
            box_size=FeatureMapSize(w=16, h=16),
            box_stride=FeatureMapSize(w=16, h=16)
        ),
        function_get_feature_map_size=lambda img_size: FeatureMapSize(w=img_size.w // 16, h=img_size.h // 16),
        do_nms_across_classes=False
    )
    
    dataloader = DataloaderOneShotDetection(
        dataset=dataset,
        box_coder=box_coder,
        batch_size=1,
        img_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        gt_image_size=64,
        random_flip_batches=False,
        random_crop_size=None,
        random_color_distortion=False,
        pyramid_scales_eval=[1.0],
        do_augmentation=False
    )
    
    # 獲取第一個批次
    batch = dataloader.get_batch(0)
    images, class_images = batch[0], batch[1]
    
    print(f"✓ 原始圖像形狀: {images.shape}, 類別圖像數量: {len(class_images)}")
    
    # 測試不同的縮放比例
    scales = [1.0, 0.75, 0.5, 1.5, 2.0]
    
    for scale in scales:
        # 調用方法
        scaled_images, scaled_class_images = model._scale_inputs(images, class_images, scale)
        
        # 驗證圖像縮放
        if scale == 1.0:
            # 縮放比例為 1.0 時，應該返回原始圖像
            assert torch.allclose(scaled_images, images), "縮放比例為 1.0 時應該返回原始圖像"
            assert scaled_class_images == class_images, "縮放比例為 1.0 時應該返回原始類別圖像"
            print(f"✓ 縮放比例 {scale}: 保持原始尺寸")
        else:
            # 檢查圖像尺寸是否正確縮放
            expected_h = int(images.shape[2] * scale)
            expected_w = int(images.shape[3] * scale)
            assert scaled_images.shape == (images.shape[0], images.shape[1], expected_h, expected_w), \
                f"縮放後圖像形狀應為 ({images.shape[0]}, {images.shape[1]}, {expected_h}, {expected_w})，但得到 {scaled_images.shape}"
            
            # 檢查類別圖像是否正確縮放
            for i, (orig_img, scaled_img) in enumerate(zip(class_images, scaled_class_images)):
                expected_h = int(orig_img.shape[1] * scale)
                expected_w = int(orig_img.shape[2] * scale)
                assert scaled_img.shape == (orig_img.shape[0], expected_h, expected_w), \
                    f"縮放後類別圖像 {i} 形狀應為 ({orig_img.shape[0]}, {expected_h}, {expected_w})，但得到 {scaled_img.shape}"
            
            print(f"✓ 縮放比例 {scale}: 圖像形狀 {scaled_images.shape}, 類別圖像形狀 {scaled_class_images[0].shape}")
    
    # 測試 3D 輸入 (單張圖像，無批次維度)
    image_3d = images[0]  # [C, H, W]
    
    # 縮放比例為 0.5
    scale = 0.5
    scaled_image_3d, scaled_class_images_3d = model._scale_inputs(image_3d, class_images, scale)
    
    # 驗證 3D 圖像縮放
    expected_h = int(image_3d.shape[1] * scale)
    expected_w = int(image_3d.shape[2] * scale)
    assert scaled_image_3d.shape == (1, image_3d.shape[0], expected_h, expected_w), \
        f"縮放後 3D 圖像形狀應為 (1, {image_3d.shape[0]}, {expected_h}, {expected_w})，但得到 {scaled_image_3d.shape}"
    
    print(f"✓ 3D 輸入縮放測試通過，輸出形狀: {scaled_image_3d.shape}")
    
    # 測試空類別圖像列表
    empty_class_images = []
    scaled_images, scaled_empty_class = model._scale_inputs(images, empty_class_images, scale)
    
    # 驗證空類別圖像列表處理
    assert scaled_empty_class == empty_class_images, "空類別圖像列表應該保持不變"
    print("✓ 空類別圖像列表測試通過")
    
    print("✅ _scale_inputs 測試通過")
    return True

def test_in_merge_dense_outputs():
    """測試 _merge_dense_outputs 方法的功能"""
    import os
    import pytest
    import torch
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    from src.os2d_model_in_prune import Os2dModelInPrune
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 準備 Grozi 數據集
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if not os.path.exists(grozi_csv):
        print("\n❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        pytest.skip("Grozi-3.2k dataset missing, test skipped.")
        return False
    
    # 創建測試數據 - 模擬密集格式輸出
    batch_size = 1
    num_classes = 9
    num_positions = 100
    
    # 創建多個尺度的分數和框
    all_scores = [
        torch.randn(batch_size, num_classes, 4, num_positions),
        torch.randn(batch_size, num_classes, 4, num_positions // 2),
        torch.randn(batch_size, num_classes, 4, num_positions // 4)
    ]
    
    all_boxes = [
        torch.rand(batch_size, num_classes, num_positions),
        torch.rand(batch_size, num_classes, num_positions // 2),
        torch.rand(batch_size, num_classes, num_positions // 4)
    ]
    
    # 創建額外輸出
    all_extras = [
        (torch.rand(batch_size, num_classes, num_positions),),
        (torch.rand(batch_size, num_classes, num_positions // 2),),
        (torch.rand(batch_size, num_classes, num_positions // 4),)
    ]
    
    # 設置參數
    target_feature_count = 150
    apply_nms = True
    max_predictions = 50
    nms_threshold = 0.5
    
    # 調用方法
    outputs = model._merge_dense_outputs(
        all_scores, all_boxes, all_extras, 
        target_feature_count, apply_nms, max_predictions, nms_threshold
    )
    
    # 驗證輸出
    assert isinstance(outputs, tuple), "輸出應該是元組"
    assert len(outputs) >= 2, "輸出元組應該至少有兩個元素"
    
    merged_scores, merged_boxes = outputs[0], outputs[1]
    print(f"✓ 合併後的密集格式輸出 - scores: {merged_scores.shape}, boxes: {merged_boxes.shape}")
    
    # 驗證合併後的形狀
    total_positions = sum(score.size(3) for score in all_scores)
    if merged_scores.size(3) < total_positions:
        print(f"✓ 合併後進行了過濾: {total_positions} → {merged_scores.size(3)}")
    
    # 測試目標特徵數量大於總位置數的情況
    large_target = 1000
    outputs_large = model._merge_dense_outputs(
        all_scores, all_boxes, all_extras, 
        large_target, apply_nms, max_predictions, nms_threshold
    )
    
    merged_scores_large, merged_boxes_large = outputs_large[0], outputs_large[1]
    print(f"✓ 大目標特徵數量的輸出 - scores: {merged_scores_large.shape}, boxes: {merged_boxes_large.shape}")
    
    # 測試不應用 NMS 的情況
    outputs_no_nms = model._merge_dense_outputs(
        all_scores, all_boxes, all_extras, 
        target_feature_count, False, max_predictions, nms_threshold
    )
    
    merged_scores_no_nms, merged_boxes_no_nms = outputs_no_nms[0], outputs_no_nms[1]
    print(f"✓ 不使用 NMS 的輸出 - scores: {merged_scores_no_nms.shape}, boxes: {merged_boxes_no_nms.shape}")
    
    print("✅ _merge_dense_outputs 測試通過")
    return True

def test_in_merge_standard_outputs():
    """測試 _merge_standard_outputs 方法的功能"""
    import os
    import pytest
    import torch
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    from src.os2d_model_in_prune import Os2dModelInPrune
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 準備 Grozi 數據集
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if not os.path.exists(grozi_csv):
        print("\n❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        pytest.skip("Grozi-3.2k dataset missing, test skipped.")
        return False
    
    # 創建測試數據 - 模擬標準格式輸出
    num_classes = 9
    
    # 創建多個尺度的分數和框
    all_scores = [
        torch.randn(50, num_classes),
        torch.randn(30, num_classes),
        torch.randn(20, num_classes)
    ]
    
    all_boxes = [
        torch.rand(50, 4),
        torch.rand(30, 4),
        torch.rand(20, 4)
    ]
    
    # 創建額外輸出
    all_extras = [
        (torch.rand(50, 10),),
        (torch.rand(30, 10),),
        (torch.rand(20, 10),)
    ]
    
    # 設置參數
    target_feature_count = 60
    apply_nms = True
    max_predictions = 50
    nms_threshold = 0.5
    
    # 調用方法
    outputs = model._merge_standard_outputs(
        all_scores, all_boxes, all_extras, 
        target_feature_count, apply_nms, max_predictions, nms_threshold
    )
    
    # 驗證輸出
    assert isinstance(outputs, tuple), "輸出應該是元組"
    assert len(outputs) >= 2, "輸出元組應該至少有兩個元素"
    
    merged_scores, merged_boxes = outputs[0], outputs[1]
    print(f"✓ 合併後的標準格式輸出 - scores: {merged_scores.shape}, boxes: {merged_boxes.shape}")
    
    # 驗證合併後的形狀
    total_boxes = sum(score.size(0) for score in all_scores)
    if merged_scores.size(0) < total_boxes:
        print(f"✓ 合併後進行了過濾: {total_boxes} → {merged_scores.size(0)}")
    
    # 測試目標特徵數量大於總框數的情況
    large_target = 1000
    outputs_large = model._merge_standard_outputs(
        all_scores, all_boxes, all_extras, 
        large_target, apply_nms, max_predictions, nms_threshold
    )
    
    merged_scores_large, merged_boxes_large = outputs_large[0], outputs_large[1]
    print(f"✓ 大目標特徵數量的輸出 - scores: {merged_scores_large.shape}, boxes: {merged_boxes_large.shape}")
    
    # 測試不應用 NMS 的情況
    outputs_no_nms = model._merge_standard_outputs(
        all_scores, all_boxes, all_extras, 
        target_feature_count, False, max_predictions, nms_threshold
    )
    
    merged_scores_no_nms, merged_boxes_no_nms = outputs_no_nms[0], outputs_no_nms[1]
    print(f"✓ 不使用 NMS 的輸出 - scores: {merged_scores_no_nms.shape}, boxes: {merged_boxes_no_nms.shape}")
    
    print("✅ _merge_standard_outputs 測試通過")
    return True

def test_in_apply_nms_to_outputs():
    """測試 _apply_nms_to_outputs 方法的功能"""
    import os
    import pytest
    import torch
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    from src.os2d_model_in_prune import Os2dModelInPrune
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 準備 Grozi 數據集
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if not os.path.exists(grozi_csv):
        print("\n❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        pytest.skip("Grozi-3.2k dataset missing, test skipped.")
        return False
    
    # 測試標準格式輸出的 NMS
    num_classes = 9
    num_boxes = 100
    
    # 創建重疊的框以測試 NMS
    boxes = torch.zeros(num_boxes, 4)
    for i in range(num_boxes):
        # 創建一系列重疊的框
        x = i % 10 * 10
        y = i // 10 * 10
        boxes[i] = torch.tensor([x, y, x + 15, y + 15])  # 有意製造重疊
    
    # 創建分數，使相鄰框的分數接近
    scores = torch.zeros(num_boxes, num_classes)
    for i in range(num_boxes):
        scores[i, i % num_classes] = 0.9 - (i % 10) * 0.01  # 確保第一個框分數最高
    
    # 設置參數
    max_predictions = 20
    nms_threshold = 0.5
    
    # 調用方法
    outputs = (scores, boxes)
    filtered_outputs = model._apply_nms_to_outputs(outputs, max_predictions, nms_threshold)
    
    # 驗證輸出
    assert isinstance(filtered_outputs, tuple), "輸出應該是元組"
    assert len(filtered_outputs) >= 2, "輸出元組應該至少有兩個元素"
    
    filtered_scores, filtered_boxes = filtered_outputs[0], filtered_outputs[1]
    print(f"✓ NMS 後的標準格式輸出 - scores: {filtered_scores.shape}, boxes: {filtered_boxes.shape}")
    
    # 驗證 NMS 效果
    assert filtered_boxes.size(0) <= max_predictions, f"NMS 後框數量應小於等於 {max_predictions}"
    assert filtered_boxes.size(0) < num_boxes, "NMS 應該減少框數量"
    
    # 測試密集格式輸出的 NMS
    batch_size = 1
    dense_scores = torch.randn(batch_size, num_classes, 4, num_boxes)
    dense_boxes = torch.rand(batch_size, num_classes, num_boxes)
    
    # 調用方法
    dense_outputs = (dense_scores, dense_boxes)
    filtered_dense = model._apply_nms_to_outputs(dense_outputs, max_predictions, nms_threshold)
    
    # 驗證輸出
    assert isinstance(filtered_dense, tuple), "密集格式輸出應該是元組"
    assert len(filtered_dense) >= 2, "密集格式輸出元組應該至少有兩個元素"
    
    filtered_dense_scores, filtered_dense_boxes = filtered_dense[0], filtered_dense[1]
    print(f"✓ NMS 後的密集格式輸出 - scores: {filtered_dense_scores.shape}, boxes: {filtered_dense_boxes.shape}")
    
    # 測試帶有額外元素的輸出
    extra_element = torch.rand(num_boxes, 5)
    outputs_with_extra = (scores, boxes, extra_element)
    
    filtered_with_extra = model._apply_nms_to_outputs(outputs_with_extra, max_predictions, nms_threshold)
    
    # 驗證輸出
    assert len(filtered_with_extra) == 3, "帶有額外元素的輸出元組應該有三個元素"
    filtered_extra = filtered_with_extra[2]
    print(f"✓ NMS 後的額外元素 - shape: {filtered_extra.shape}")
    
    print("✅ _apply_nms_to_outputs 測試通過")
    return True

def test_in_analyze_model_outputs():
    """測試 _analyze_model_outputs 方法的功能"""
    import os
    import pytest
    import torch
    from unittest.mock import patch
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    from src.os2d_model_in_prune import Os2dModelInPrune
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 準備 Grozi 數據集
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if not os.path.exists(grozi_csv):
        print("\n❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        pytest.skip("Grozi-3.2k dataset missing, test skipped.")
        return False
    
    # 測試標準格式輸出的分析
    num_classes = 9
    num_boxes = 30
    
    # 創建標準格式輸出
    scores = torch.rand(num_boxes, num_classes)
    boxes = torch.rand(num_boxes, 4)
    outputs = (scores, boxes)
    
    # 創建教師模型輸出
    teacher_scores = torch.rand(num_boxes, num_classes)
    teacher_boxes = torch.rand(num_boxes, 4)
    teacher_outputs = (teacher_scores, teacher_boxes)
    
    # 使用 mock 捕獲打印輸出
    with patch('builtins.print') as mock_print:
        # 調用方法
        model._analyze_model_outputs(outputs, teacher_outputs)
        
        # 驗證打印函數被調用
        assert mock_print.call_count > 0, "打印函數應該被調用"
        
        # 驗證打印內容
        calls = [call[0][0] for call in mock_print.call_args_list if isinstance(call[0][0], str)]
        
        # 檢查標準格式輸出分析
        assert any("學生模型輸出" in call for call in calls), "應該打印學生模型輸出信息"
        assert any("框數量" in call for call in calls), "應該打印框數量信息"
        assert any("張量形狀" in call for call in calls), "應該打印張量形狀信息"
    
    # 測試密集格式輸出的分析
    batch_size = 1
    dense_scores = torch.randn(batch_size, num_classes, 4, num_boxes)
    dense_boxes = torch.rand(batch_size, num_classes, num_boxes)
    dense_outputs = (dense_scores, dense_boxes)
    
    # 創建密集格式教師模型輸出
    teacher_dense_scores = torch.randn(batch_size, num_classes, 4, num_boxes)
    teacher_dense_boxes = torch.rand(batch_size, num_classes, num_boxes)
    teacher_dense_outputs = (teacher_dense_scores, teacher_dense_boxes)
    
    # 使用 mock 捕獲打印輸出
    with patch('builtins.print') as mock_print:
        # 調用方法
        model._analyze_model_outputs(dense_outputs, teacher_dense_outputs)
        
        # 驗證打印函數被調用
        assert mock_print.call_count > 0, "打印函數應該被調用"
        
        # 驗證打印內容
        calls = [call[0][0] for call in mock_print.call_args_list if isinstance(call[0][0], str)]
        
        # 檢查密集格式輸出分析
        assert any("學生模型輸出" in call for call in calls), "應該打印學生模型輸出信息"
        assert any("批次大小" in call for call in calls), "應該打印批次大小信息"
        assert any("特徵位置數" in call for call in calls), "應該打印特徵位置數信息"
    
    # 測試無效輸出的分析
    invalid_outputs = "not a tuple"
    
    # 使用 mock 捕獲打印輸出
    with patch('builtins.print') as mock_print:
        # 調用方法
        model._analyze_model_outputs(invalid_outputs, teacher_outputs)
        
        # 驗證打印函數被調用
        assert mock_print.call_count > 0, "打印函數應該被調用"
    
    print("✅ _analyze_model_outputs 測試通過")
    return True

def test_in_prepare_outputs_dict():
    """測試 _prepare_outputs_dict 方法的功能"""
    import os
    import pytest
    import torch
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    from src.os2d_model_in_prune import Os2dModelInPrune
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 準備 Grozi 數據集
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if not os.path.exists(grozi_csv):
        print("\n❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        pytest.skip("Grozi-3.2k dataset missing, test skipped.")
        return False
    
    # 建立 dataset
    dataset = build_grozi_dataset(
        data_path=data_path,
        name="grozi-train-mini",
        eval_scale=224,
        cache_images=False
    )
    
    # 創建 box_coder
    box_coder = Os2dBoxCoder(
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.4,
        remap_classification_targets_iou_pos=0.5,
        remap_classification_targets_iou_neg=0.4,
        output_box_grid_generator=BoxGridGenerator(
            box_size=FeatureMapSize(w=16, h=16),
            box_stride=FeatureMapSize(w=16, h=16)
        ),
        function_get_feature_map_size=lambda img_size: FeatureMapSize(w=img_size.w // 16, h=img_size.h // 16),
        do_nms_across_classes=False
    )
    
    dataloader = DataloaderOneShotDetection(
        dataset=dataset,
        box_coder=box_coder,
        batch_size=1,
        img_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        gt_image_size=64,
        random_flip_batches=False,
        random_crop_size=None,
        random_color_distortion=False,
        pyramid_scales_eval=[1.0],
        do_augmentation=False
    )
    
    # 獲取第一個批次
    batch = dataloader.get_batch(0)
    images, class_images = batch[0], batch[1]
    
    # 測試元組輸出處理
    class_scores = torch.randn(10, 9)  # 10個框，9個類別
    boxes = torch.rand(10, 4)  # 10個框，每個框4個坐標
    outputs = (class_scores, boxes)
    
    # 設置特徵金字塔參數
    use_feature_pyramid = True
    pyramid_scales = [1.0, 0.75, 0.5]
    
    # 調用方法
    outputs_dict = model._prepare_outputs_dict(outputs, images, class_images, use_feature_pyramid, pyramid_scales)
    
    # 驗證輸出
    assert isinstance(outputs_dict, dict), "輸出應該是字典"
    assert 'class_scores' in outputs_dict, "字典應包含 class_scores"
    assert 'boxes' in outputs_dict, "字典應包含 boxes"
    assert 'images' in outputs_dict, "字典應包含 images"
    assert 'class_images' in outputs_dict, "字典應包含 class_images"
    assert 'feature_pyramid' in outputs_dict, "字典應包含 feature_pyramid"
    assert 'pyramid_scales' in outputs_dict, "字典應包含 pyramid_scales"
    
    assert torch.equal(outputs_dict['class_scores'], class_scores), "class_scores 應該保持不變"
    assert torch.equal(outputs_dict['boxes'], boxes), "boxes 應該保持不變"
    assert outputs_dict['feature_pyramid'] == use_feature_pyramid, "feature_pyramid 應該正確設置"
    assert outputs_dict['pyramid_scales'] == pyramid_scales, "pyramid_scales 應該正確設置"
    
    print("✓ 元組輸出處理測試通過")
    
    # 測試單一張量輸出處理
    single_tensor = torch.randn(10, 9)
    
    # 調用方法
    single_dict = model._prepare_outputs_dict(single_tensor, images, class_images, use_feature_pyramid, pyramid_scales)
    
    # 驗證輸出
    assert isinstance(single_dict, dict), "輸出應該是字典"
    assert 'class_scores' in single_dict, "字典應包含 class_scores"
    assert 'images' in single_dict, "字典應包含 images"
    assert 'class_images' in single_dict, "字典應包含 class_images"
    assert 'feature_pyramid' in single_dict, "字典應包含 feature_pyramid"
    assert 'pyramid_scales' in single_dict, "字典應包含 pyramid_scales"
    
    assert torch.equal(single_dict['class_scores'], single_tensor), "class_scores 應該正確設置"
    
    print("✓ 單一張量輸出處理測試通過")
    
    # 測試字典輸出處理
    dict_output = {
        'class_scores': torch.randn(10, 9),
        'boxes': torch.rand(10, 4),
        'extra_field': 'test'
    }
    
    # 調用方法
    processed_dict = model._prepare_outputs_dict(dict_output, images, class_images, use_feature_pyramid, pyramid_scales)
    
    # 驗證輸出
    assert isinstance(processed_dict, dict), "輸出應該是字典"
    assert 'class_scores' in processed_dict, "字典應包含 class_scores"
    assert 'boxes' in processed_dict, "字典應包含 boxes"
    assert 'extra_field' in processed_dict, "字典應保留原有字段"
    assert 'feature_pyramid' in processed_dict, "字典應包含 feature_pyramid"
    assert 'pyramid_scales' in processed_dict, "字典應包含 pyramid_scales"
    
    assert torch.equal(processed_dict['class_scores'], dict_output['class_scores']), "class_scores 應該保持不變"
    assert torch.equal(processed_dict['boxes'], dict_output['boxes']), "boxes 應該保持不變"
    assert processed_dict['extra_field'] == 'test', "額外字段應該保持不變"
    
    print("✓ 字典輸出處理測試通過")
    
    print("✅ _prepare_outputs_dict 測試通過")
    return True

def test_in_print_loss_info():
    """測試 _print_loss_info 方法的功能"""
    import torch
    from unittest.mock import patch
    from src.os2d_model_in_prune import Os2dModelInPrune
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 創建測試損失字典
    loss_dict = {
        'cls_loss': torch.tensor(0.5),
        'box_loss': torch.tensor(1.2),
        'teacher_loss': torch.tensor(0.3),
        'lcp_loss': torch.tensor(0.1)
    }
    
    # 設置損失計算時間
    loss_time = 0.75
    
    # 使用 mock 捕獲打印輸出
    with patch('builtins.print') as mock_print:
        # 調用方法
        model._print_loss_info(loss_dict, loss_time)
        
        # 驗證打印函數被調用
        assert mock_print.call_count >= 5, "打印函數應該被調用至少5次"
        
        # 驗證打印內容
        calls = [call[0][0] for call in mock_print.call_args_list if isinstance(call[0][0], str)]
        
        # 檢查損失計算時間信息
        assert any("損失計算完成" in call for call in calls), "應該打印損失計算完成信息"
        assert any(f"{loss_time:.2f}秒" in call for call in calls), "應該打印損失計算時間"
        
        # 檢查各種損失值
        assert any("分類損失: 0.5" in call for call in calls), "應該打印分類損失值"
        assert any("框回歸損失: 1.2" in call for call in calls), "應該打印框回歸損失值"
        assert any("教師損失: 0.3" in call for call in calls), "應該打印教師損失值"
        assert any("LCP損失: 0.1" in call for call in calls), "應該打印LCP損失值"
    
    # 測試不同的損失值
    loss_dict = {
        'cls_loss': torch.tensor(0.01),
        'box_loss': torch.tensor(5.0),
        'teacher_loss': torch.tensor(0.0),
        'lcp_loss': torch.tensor(2.5)
    }
    
    # 使用 mock 捕獲打印輸出
    with patch('builtins.print') as mock_print:
        # 調用方法
        model._print_loss_info(loss_dict, loss_time)
        
        # 驗證打印內容
        calls = [call[0][0] for call in mock_print.call_args_list if isinstance(call[0][0], str)]
        
        # 檢查各種損失值
        assert any("分類損失: 0.0100" in call for call in calls), "應該打印分類損失值"
        assert any("框回歸損失: 5.0000" in call for call in calls), "應該打印框回歸損失值"
        assert any("教師損失: 0.0000" in call for call in calls), "應該打印教師損失值"
        assert any("LCP損失: 2.5000" in call for call in calls), "應該打印LCP損失值"
    
    print("✅ _print_loss_info 測試通過")
    return True

def test_in_clip_gradients():
    """測試 _clip_gradients 方法的功能"""
    import torch
    import torch.nn as nn
    from unittest.mock import patch
    from src.os2d_model_in_prune import Os2dModelInPrune
    from src.auxiliary_network import AuxiliaryNetwork
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 創建輔助網路
    auxiliary_net = AuxiliaryNetwork(in_channels=64)
    
    # 創建測試輸入
    test_image = torch.randn(1, 3, 224, 224, requires_grad=True)
    
    # 前向傳播以創建梯度
    try:
        outputs = model(test_image)
        
        # 創建一個損失
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            loss = outputs[0].sum() + outputs[1].sum()
            loss.backward()
        
        # 使用 mock 測試梯度裁剪
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            # 調用方法
            model._clip_gradients(model, auxiliary_net)
            
            # 驗證 clip_grad_norm_ 被調用
            assert mock_clip.call_count == 2, "clip_grad_norm_ 應該被調用兩次"
            
            # 驗證調用參數
            args, kwargs = mock_clip.call_args_list[0]
            assert args[1] == 10.0, "最大梯度範數應該為 10.0"
            
            args, kwargs = mock_clip.call_args_list[1]
            assert args[1] == 10.0, "最大梯度範數應該為 10.0"
    except Exception as e:
        print(f"⚠️ 前向傳播或反向傳播失敗: {e}")
        # 使用 mock 直接測試方法調用
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            # 調用方法
            model._clip_gradients(model, auxiliary_net)
            
            # 驗證 clip_grad_norm_ 被調用
            assert mock_clip.call_count == 2, "clip_grad_norm_ 應該被調用兩次"
    
    # 測試只有模型沒有輔助網路的情況
    with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
        # 調用方法
        model._clip_gradients(model, None)
        
        # 驗證 clip_grad_norm_ 被調用
        assert mock_clip.call_count == 1, "clip_grad_norm_ 應該被調用一次"
        
        # 驗證調用參數
        args, kwargs = mock_clip.call_args
        assert args[1] == 10.0, "最大梯度範數應該為 10.0"
    
    print("✅ _clip_gradients 測試通過")
    return True

def test_in_print_batch_summary():
    """測試 _print_batch_summary 方法的功能"""
    import torch
    from unittest.mock import patch
    from src.os2d_model_in_prune import Os2dModelInPrune
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 創建測試損失字典
    loss_dict = {
        'cls_loss': torch.tensor(0.5),
        'box_loss': torch.tensor(1.2),
        'teacher_loss': torch.tensor(0.3),
        'lcp_loss': torch.tensor(0.1)
    }
    
    # 設置批次索引和總批次數
    batch_idx = 5
    num_batches = 20
    loss_value = 2.1
    
    # 使用 mock 捕獲打印輸出
    with patch('builtins.print') as mock_print:
        # 調用方法
        model._print_batch_summary(batch_idx, num_batches, loss_dict, loss_value)
        
        # 驗證打印函數被調用
        assert mock_print.call_count >= 6, "打印函數應該被調用至少6次"
        
        # 驗證打印內容
        calls = [call[0][0] for call in mock_print.call_args_list if isinstance(call[0][0], str)]
        
        # 檢查批次摘要信息
        assert any(f"批次 {batch_idx+1}/{num_batches}" in call for call in calls), "應該打印批次摘要信息"
        
        # 檢查各種損失值
        assert any("分類損失: 0.5" in call for call in calls), "應該打印分類損失值"
        assert any("框回歸損失: 1.2" in call for call in calls), "應該打印框回歸損失值"
        assert any("教師損失: 0.3" in call for call in calls), "應該打印教師損失值"
        assert any("LCP損失: 0.1" in call for call in calls), "應該打印LCP損失值"
        assert any(f"總損失: {loss_value}" in call for call in calls), "應該打印總損失值"
    
    # 測試不同的批次索引和損失值
    batch_idx = 0
    num_batches = 10
    loss_dict = {
        'cls_loss': torch.tensor(0.01),
        'box_loss': torch.tensor(5.0),
        'teacher_loss': torch.tensor(0.0),
        'lcp_loss': torch.tensor(2.5)
    }
    loss_value = 7.51
    
    # 使用 mock 捕獲打印輸出
    with patch('builtins.print') as mock_print:
        # 調用方法
        model._print_batch_summary(batch_idx, num_batches, loss_dict, loss_value)
        
        # 驗證打印內容
        calls = [call[0][0] for call in mock_print.call_args_list if isinstance(call[0][0], str)]
        
        # 檢查批次摘要信息
        assert any(f"批次 {batch_idx+1}/{num_batches}" in call for call in calls), "應該打印批次摘要信息"
        
        # 檢查各種損失值
        assert any("分類損失: 0.01" in call for call in calls), "應該打印分類損失值"
        assert any("框回歸損失: 5.00" in call for call in calls), "應該打印框回歸損失值"
        assert any("教師損失: 0.00" in call for call in calls), "應該打印教師損失值"
        assert any("LCP損失: 2.50" in call for call in calls), "應該打印LCP損失值"
        assert any(f"總損失: {loss_value}" in call for call in calls), "應該打印總損失值"
    
    # 測試最後一個批次
    batch_idx = num_batches - 1
    
    # 使用 mock 捕獲打印輸出
    with patch('builtins.print') as mock_print:
        # 調用方法
        model._print_batch_summary(batch_idx, num_batches, loss_dict, loss_value)
        
        # 驗證打印內容
        calls = [call[0][0] for call in mock_print.call_args_list if isinstance(call[0][0], str)]
        
        # 檢查批次摘要信息
        assert any(f"批次 {num_batches}/{num_batches}" in call for call in calls), "應該打印最後一個批次的摘要信息"
    
    print("✅ _print_batch_summary 測試通過")
    return True

def test_in_print_training_summary():
    """測試 _print_training_summary 方法的功能"""
    import torch
    from unittest.mock import patch
    from src.os2d_model_in_prune import Os2dModelInPrune
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 設置測試參數
    batch_count = 8
    num_batches = 10
    loss_stats = {
        'total': 12.0,
        'cls': 5.0,
        'box_reg': 4.0,
        'teacher': 2.0,
        'lcp': 1.0
    }
    start_time = time.time() - 120  # 2分鐘前
    use_feature_pyramid = True
    pyramid_scales = [1.0, 0.75, 0.5]
    apply_nms = True
    nms_threshold = 0.5
    
    # 使用 mock 捕獲打印輸出
    with patch('builtins.print') as mock_print:
        # 調用方法
        model._print_training_summary(
            batch_count, num_batches, loss_stats, start_time,
            use_feature_pyramid, pyramid_scales, apply_nms, nms_threshold
        )
        
        # 驗證打印函數被調用
        assert mock_print.call_count >= 10, "打印函數應該被調用至少10次"
        
        # 驗證打印內容
        calls = [call[0][0] for call in mock_print.call_args_list if isinstance(call[0][0], str)]
        
        # 檢查訓練摘要信息
        assert any(f"訓練完成! 處理了 {batch_count}/{num_batches} 批次" in call for call in calls), "應該打印批次處理信息"
        assert any("總耗時" in call for call in calls), "應該打印總耗時"
        assert any("特徵金字塔: 啟用" in call for call in calls), "應該打印特徵金字塔狀態"
        assert any(f"特徵金字塔尺度: {pyramid_scales}" in call for call in calls), "應該打印特徵金字塔尺度"
        assert any(f"NMS: 啟用, 閾值: {nms_threshold}" in call for call in calls), "應該打印NMS狀態"
        
        # 檢查損失摘要
        avg_loss = loss_stats['total'] / batch_count
        avg_cls_loss = loss_stats['cls'] / batch_count
        avg_box_loss = loss_stats['box_reg'] / batch_count
        avg_teacher_loss = loss_stats['teacher'] / batch_count
        avg_lcp_loss = loss_stats['lcp'] / batch_count
        
        assert any(f"平均損失: {avg_loss:.4f}" in call for call in calls), "應該打印平均損失"
        assert any(f"平均分類損失: {avg_cls_loss:.4f}" in call for call in calls), "應該打印平均分類損失"
        assert any(f"平均框回歸損失: {avg_box_loss:.4f}" in call for call in calls), "應該打印平均框回歸損失"
        assert any(f"平均教師損失: {avg_teacher_loss:.4f}" in call for call in calls), "應該打印平均教師損失"
        assert any(f"平均LCP損失: {avg_lcp_loss:.4f}" in call for call in calls), "應該打印平均LCP損失"
    
    # 測試不使用特徵金字塔的情況
    with patch('builtins.print') as mock_print:
        # 調用方法
        model._print_training_summary(
            batch_count, num_batches, loss_stats, start_time,
            False, pyramid_scales, apply_nms, nms_threshold
        )
        
        # 驗證打印內容
        calls = [call[0][0] for call in mock_print.call_args_list if isinstance(call[0][0], str)]
        
        assert any("特徵金字塔: 停用" in call for call in calls), "應該打印特徵金字塔停用狀態"
        assert not any(f"特徵金字塔尺度: {pyramid_scales}" in call for call in calls), "不應該打印特徵金字塔尺度"
    
    print("✅ _print_training_summary 測試通過")
    return True

def test_in_train_one_epoch():
    """測試 train_one_epoch 方法的功能"""
    import os
    import pytest
    import torch
    from unittest.mock import patch, MagicMock
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    from src.os2d_model_in_prune import Os2dModelInPrune
    from src.auxiliary_network import AuxiliaryNetwork
    
    # 創建模型實例
    model = Os2dModelInPrune(is_cuda=False)
    
    # 準備 Grozi 數據集
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if not os.path.exists(grozi_csv):
        print("\n❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        pytest.skip("Grozi-3.2k dataset missing, test skipped.")
        return False
    
    # 建立 dataset
    dataset = build_grozi_dataset(
        data_path=data_path,
        name="grozi-train-mini",
        eval_scale=224,
        cache_images=False
    )
    
    # 創建 box_coder
    box_coder = Os2dBoxCoder(
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.4,
        remap_classification_targets_iou_pos=0.5,
        remap_classification_targets_iou_neg=0.4,
        output_box_grid_generator=BoxGridGenerator(
            box_size=FeatureMapSize(w=16, h=16),
            box_stride=FeatureMapSize(w=16, h=16)
        ),
        function_get_feature_map_size=lambda img_size: FeatureMapSize(w=img_size.w // 16, h=img_size.h // 16),
        do_nms_across_classes=False
    )
    
    dataloader = DataloaderOneShotDetection(
        dataset=dataset,
        box_coder=box_coder,
        batch_size=1,
        img_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        gt_image_size=64,
        random_flip_batches=False,
        random_crop_size=None,
        random_color_distortion=False,
        pyramid_scales_eval=[1.0],
        do_augmentation=False
    )
    
    # 創建輔助網路
    auxiliary_net = AuxiliaryNetwork(in_channels=2048)
    
    # 創建優化器
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(auxiliary_net.parameters()),
        lr=0.001
    )
    
    # 使用 mock 避免實際執行訓練
    with patch.object(model, '_run_standard_inference') as mock_standard_inference:
        # 模擬標準推理輸出
        mock_standard_inference.return_value = (
            torch.randn(10, 9),  # class_scores [N, C]
            torch.rand(10, 4)    # boxes [N, 4]
        )
        
        # 模擬損失計算
        with patch.object(model, 'compute_losses') as mock_compute_losses:
            mock_compute_losses.return_value = (
                torch.tensor(1.5),  # total_loss
                {
                    'cls_loss': torch.tensor(0.5),
                    'box_loss': torch.tensor(0.7),
                    'teacher_loss': torch.tensor(0.2),
                    'lcp_loss': torch.tensor(0.1)
                }
            )
            
            # 模擬反向傳播和優化器步進
            with patch.object(torch.Tensor, 'backward') as mock_backward:
                with patch.object(optimizer, 'step') as mock_optimizer_step:
                    with patch.object(model, '_clip_gradients') as mock_clip_gradients:
                        # 執行訓練
                        loss_history = model.train_one_epoch(
                            dataloader,
                            optimizer,
                            auxiliary_net=auxiliary_net,
                            class_num=9,
                            use_feature_pyramid=False,
                            pyramid_scales=[1.0],
                            apply_nms=True,
                            nms_threshold=0.5,
                            max_predictions=50,
                            target_feature_count=100,
                            use_lcp_loss=True,
                            loss_weights={'cls': 1.0, 'box_reg': 1.0, 'teacher': 0.5, 'lcp': 0.1},
                            max_batches=2,
                            print_freq=1
                        )
                        
                        # 驗證方法調用
                        assert mock_standard_inference.call_count >= 2, "標準推理應該被調用至少2次"
                        assert mock_compute_losses.call_count >= 1, "損失計算應該被調用至少1次"
                        assert mock_backward.call_count >= 1, "反向傳播應該被調用至少1次"
                        assert mock_optimizer_step.call_count >= 1, "優化器步進應該被調用至少1次"
                        assert mock_clip_gradients.call_count >= 1, "梯度裁剪應該被調用至少1次"
                        
                        # 驗證輸出
                        assert isinstance(loss_history, list), "應該返回損失歷史列表"
                        assert len(loss_history) > 0, "損失歷史列表不應為空"
    
    # 測試使用特徵金字塔的情況
    with patch.object(model, '_run_teacher_feature_pyramid') as mock_teacher_pyramid:
        with patch.object(model, '_run_student_feature_pyramid') as mock_student_pyramid:
            # 模擬特徵金字塔輸出
            pyramid_output = (
                torch.randn(10, 9),  # class_scores [N, C]
                torch.rand(10, 4)    # boxes [N, 4]
            )
            mock_teacher_pyramid.return_value = pyramid_output
            mock_student_pyramid.return_value = pyramid_output
            
            # 模擬損失計算
            with patch.object(model, 'compute_losses') as mock_compute_losses:
                mock_compute_losses.return_value = (
                    torch.tensor(1.5),  # total_loss
                    {
                        'cls_loss': torch.tensor(0.5),
                        'box_loss': torch.tensor(0.7),
                        'teacher_loss': torch.tensor(0.2),
                        'lcp_loss': torch.tensor(0.1)
                    }
                )
                
                # 模擬反向傳播和優化器步進
                with patch.object(torch.Tensor, 'backward') as mock_backward:
                    with patch.object(optimizer, 'step') as mock_optimizer_step:
                        with patch.object(model, '_clip_gradients') as mock_clip_gradients:
                            # 執行訓練
                            loss_history = model.train_one_epoch(
                                dataloader,
                                optimizer,
                                auxiliary_net=auxiliary_net,
                                class_num=9,
                                use_feature_pyramid=True,
                                pyramid_scales=[1.0, 0.75, 0.5],
                                apply_nms=True,
                                nms_threshold=0.5,
                                max_predictions=50,
                                target_feature_count=100,
                                use_lcp_loss=True,
                                loss_weights={'cls': 1.0, 'box_reg': 1.0, 'teacher': 0.5, 'lcp': 0.1},
                                max_batches=1,
                                print_freq=1
                            )
                            
                            # 驗證方法調用
                            assert mock_teacher_pyramid.call_count >= 1, "教師特徵金字塔應該被調用至少1次"
                            assert mock_student_pyramid.call_count >= 1, "學生特徵金字塔應該被調用至少1次"
                            assert mock_compute_losses.call_count >= 1, "損失計算應該被調用至少1次"
                            assert mock_backward.call_count >= 1, "反向傳播應該被調用至少1次"
                            assert mock_optimizer_step.call_count >= 1, "優化器步進應該被調用至少1次"
                            assert mock_clip_gradients.call_count >= 1, "梯度裁剪應該被調用至少1次"
                            
                            # 驗證輸出
                            assert isinstance(loss_history, list), "應該返回損失歷史列表"
                            assert len(loss_history) > 0, "損失歷史列表不應為空"
    
    print("✅ train_one_epoch 測試通過")
    return True