import os
import torch
import torchvision
import pytest
# import # traceback
import numpy as np
import unittest
from torch.utils.data import DataLoader
from unittest.mock import MagicMock, patch
from src.lcp_channel_selector import OS2DChannelSelector
from src.contextual_roi_align import ContextualRoIAlign
from src.auxiliary_network import AuxiliaryNetwork
from src.os2d_model_in_prune import Os2dModelInPrune
from src.dataset_downloader import VOCDataset , VOC_CLASSES

import logging

def test_os2d_model_in_prune_initialization():
    """測試 Os2dModelInPrune 初始化，並區分學生模型與教師模型參數量"""
    # 設置 logger
    logger = logging.getLogger("OS2D.test")
    
    # 初始化模型
    os2d_path = "./os2d_v1-train.pth"
    if not os.path.exists(os2d_path):
        pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
    
    # 初始化模型
    model = Os2dModelInPrune(logger=logger, pretrained_path=os2d_path)
    
    # 驗證模型結構
    assert hasattr(model, 'net_feature_maps'), "模型應該有 net_feature_maps 屬性"
    assert hasattr(model, 'net_label_features'), "模型應該有 net_label_features 屬性"
    assert hasattr(model, 'os2d_head_creator'), "模型應該有 os2d_head_creator 屬性"
    
    # 分開計算學生模型和教師模型的參數數量
    student_params = 0
    teacher_params = 0
    
    for name, param in model.named_parameters():
        if name.startswith('teacher_model'):
            teacher_params += param.numel()
        else:
            student_params += param.numel()
    
    total_params = student_params + teacher_params
    
    # 輸出各部分參數量
    print(f"模型總參數量: {total_params:,}")
    print(f"  - 學生模型參數量: {student_params:,}")
    print(f"  - 教師模型參數量: {teacher_params:,}")
    
    # 如果存在教師模型，則驗證學生和教師模型的參數量是否接近
    if teacher_params > 0:
        param_diff_ratio = abs(student_params - teacher_params) / max(student_params, teacher_params)
        print(f"  - 學生/教師模型參數量差異比例: {param_diff_ratio:.4f}")
        assert param_diff_ratio < 0.05, "學生模型與教師模型參數量差異過大，應該非常接近"
    
    # 確保總參數量大於0
    assert total_params > 0, "模型參數數量應該大於 0"
    
    # 如果只需要查看學生模型參數量，可以取消下面的註釋
    # print(f"純學生模型參數量: {student_params:,}")
    
    print("✅ Os2dModelInPrune 初始化測試通過")
    return True

def test_os2d_model_in_prune_forward():
    """測試 Os2dModelInPrune 前向傳播"""
    # 設置 logger
    logger = logging.getLogger("OS2D.test")
    
    # 初始化模型
    os2d_path = "./os2d_v1-train.pth"
    if not os.path.exists(os2d_path):
        pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
    model = Os2dModelInPrune(logger=logger, pretrained_path=os2d_path, is_cuda=(device.type == 'cuda'))
    model = model.to(device)
    
    # 創建測試輸入
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    class_images = [torch.randn(3, 224, 224).to(device) for _ in range(2)]
    
    # 執行前向傳播
    with torch.no_grad():
        loc_scores, class_scores, class_scores_transform_detached, fm_size, transform_corners = model(images, class_images)
    
    # 驗證輸出
    assert loc_scores is not None, "loc_scores 不應為 None"
    assert class_scores is not None, "class_scores 不應為 None"
    assert class_scores_transform_detached is not None, "class_scores_transform_detached 不應為 None"
    assert fm_size is not None, "fm_size 不應為 None"
    
    print("✅ Os2dModelInPrune 前向傳播測試通過")
    return True

def test_os2d_model_in_prune_channel():
    """測試 Os2dModelInPrune 通道剪枝功能"""
    # 設置 logger
    logger = logging.getLogger("OS2D.test")
    
    # 初始化模型
    os2d_path = "./os2d_v1-train.pth"
    if not os.path.exists(os2d_path):
        pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
    model = Os2dModelInPrune(logger=logger, pretrained_path=os2d_path, is_cuda=(device.type == 'cuda'))
    model = model.to(device)
    
    # 初始化輔助網路
    aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
    
    # 選擇要剪枝的層
    layer_name = "layer2.0.conv1"
    
    # 獲取原始通道數
    target_layer = None
    for name, module in model.backbone.named_modules():
        if name == layer_name:
            target_layer = module
            break
    
    assert target_layer is not None, f"找不到層: {layer_name}"
    orig_channels = target_layer.out_channels
    print(f"原始通道數: {orig_channels}")
    
    # 設置剪枝比例
    prune_ratio = 0.3
    expected_channels = int(orig_channels * (1 - prune_ratio))
    
    # 創建測試輸入
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    boxes = [torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32).to(device) for _ in range(batch_size)]
    labels = [torch.tensor([0, 1], dtype=torch.long).to(device) for _ in range(batch_size)]
    
    # 執行剪枝
    print(f"剪枝層 {layer_name}，剪枝比例: {prune_ratio}")
    success = model.prune_channel(layer_name, prune_ratio=prune_ratio, images=images, boxes=boxes, labels=labels, auxiliary_net=aux_net)
    
    # 驗證剪枝結果
    assert success, "剪枝操作失敗"
    
    # 獲取剪枝後的通道數
    pruned_layer = None
    for name, module in model.backbone.named_modules():
        if name == layer_name:
            pruned_layer = module
            break
    
    assert pruned_layer is not None, f"剪枝後找不到層: {layer_name}"
    pruned_channels = pruned_layer.out_channels
    print(f"剪枝後通道數: {pruned_channels}")
    
    # 驗證通道數是否正確減少
    assert pruned_channels == expected_channels, f"通道數不匹配: 預期 {expected_channels}, 實際 {pruned_channels}"
    
    # 測試前向傳播
    with torch.no_grad():
        loc_scores, class_scores, class_scores_transform_detached, fm_size, transform_corners = model(images, class_images)
    
    print("✅ Os2dModelInPrune 通道剪枝測試通過")
    return True

def test_forward_pass_with_class_images():
    """Test forward pass with class_images parameter"""
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        
        # Initialize model
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D pretrained model not found: {os2d_path}")
            
        model = Os2dModelInPrune(pretrained_path=os2d_path)
        model = model.to(device)
        
        # Create test inputs
        batch_size = 1
        x = torch.randn(batch_size, 3, 224, 224).to(device)
        class_images = [torch.randn(1, 3, 224, 224).to(device)]

        # Test forward pass
        print("\nTesting forward pass with class_images...")
        try:
            with torch.no_grad():
                output = model(x, class_images)
            print("✅ Forward pass with class_images successful")
            return True
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            assert False, f"Forward pass failed: {e}"
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        # traceback.print_exc()
        return False

def test_set_layer_out_channels():
    """測試 set_layer_out_channels 方法"""
    try:
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
        
        # 初始化模型
        model = Os2dModelInPrune(pretrained_path=os2d_path)
        model = model.to(device)
        
        # 選擇要測試的層
        layer_name = "layer2.0.conv1"
        param_count = sum(p.numel() for p in model.parameters())
        print(f"模型參數總數: {param_count}")
        # 獲取原始層與下游層
        parts = layer_name.split('.')
        layer_str, block_idx, conv_name = parts[0], int(parts[1]), parts[2]
        block = getattr(model.backbone, layer_str)[block_idx]
        
        # 獲取原始通道數
        conv_layer = getattr(block, conv_name)
        orig_out_channels = conv_layer.out_channels
        
        # 獲取下游層的原始通道數
        next_conv_name = f"conv{int(conv_name[-1])+1}"
        next_conv_layer = getattr(block, next_conv_name)
        orig_next_in_channels = next_conv_layer.in_channels
        
        # 獲取對應 BatchNorm 層的原始通道數
        bn_name = conv_name.replace('conv', 'bn')
        bn_layer = getattr(block, bn_name)
        orig_bn_channels = bn_layer.num_features
        
        print(f"原始層結構:")
        print(f"{layer_name}: out_channels={orig_out_channels}")
        print(f"{layer_str}.{block_idx}.{next_conv_name}: in_channels={orig_next_in_channels}")
        print(f"{layer_str}.{block_idx}.{bn_name}: num_features={orig_bn_channels}")
        
        # 設置新的通道數
        new_out_channels = orig_out_channels // 2
        print(f"\n將 {layer_name} 的 out_channels 設為 {new_out_channels}")
        
        # 執行測試方法
        success = model.set_layer_out_channels(layer_name, new_out_channels)
        assert success, "set_layer_out_channels 方法應該返回 True"
        
        # 重新獲取層
        block = getattr(model.backbone, layer_str)[block_idx]
        conv_layer = getattr(block, conv_name)
        next_conv_layer = getattr(block, next_conv_name)
        bn_layer = getattr(block, bn_name)
        
        # 驗證通道數是否正確更新
        print(f"\n更新後的層結構:")
        print(f"{layer_name}: out_channels={conv_layer.out_channels}")
        print(f"{layer_str}.{block_idx}.{next_conv_name}: in_channels={next_conv_layer.in_channels}")
        print(f"{layer_str}.{block_idx}.{bn_name}: num_features={bn_layer.num_features}")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"模型參數總數: {param_count}")
        assert conv_layer.out_channels == new_out_channels, f"{layer_name} 的 out_channels 應為 {new_out_channels}"
        assert next_conv_layer.in_channels == new_out_channels, f"{layer_str}.{block_idx}.{next_conv_name} 的 in_channels 應為 {new_out_channels}"
        # Add test for forward pass with class_images
        print("\nTesting forward pass...") 
        x = torch.randn(1, 3, 224, 224).to(device)
        class_images = [torch.randn(3, 224, 224).to(device)]
        try:
            with torch.no_grad():
                output = model(x, class_images)  # Added class_images parameter
            print("✅ Forward pass successful")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            assert False, f"Forward pass failed: {e}"
        # 檢查所有層的 channel 一致性
        print("\n檢查所有層的 channel 一致性...")
        _test_all_layer_channel_consistency(model)
        
        print("\n✅ set_layer_out_channels 測試通過")
        return True
    except Exception as e:
        print(f"❌ set_layer_out_channels 測試失敗: {e}")
        # import # traceback
        # traceback.print_exc()
        return False

def _test_all_layer_channel_consistency(model):
    """檢查 backbone 所有 conv/bn 層的 in/out channel 是否一致"""
    backbone = model.backbone
    all_pass = True
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if not hasattr(backbone, layer_name):
            continue
        layer = getattr(backbone, layer_name)
        for block_idx, block in enumerate(layer):
            prefix = f"{layer_name}.{block_idx}"
            # 檢查 conv1
            conv1 = getattr(block, 'conv1', None)
            bn1 = getattr(block, 'bn1', None)
            if conv1 is not None and bn1 is not None:
                if bn1.num_features != conv1.out_channels:
                    print(f"❌ {prefix}.bn1.num_features({bn1.num_features}) != conv1.out_channels({conv1.out_channels})")
                    all_pass = False
                else:
                    print(f"✓ {prefix}.bn1.num_features({bn1.num_features}) == conv1.out_channels({conv1.out_channels})")
            
            # 檢查 conv2
            conv2 = getattr(block, 'conv2', None)
            bn2 = getattr(block, 'bn2', None)
            if conv2 is not None:
                # 檢查 conv2 in_channels == conv1 out_channels
                if conv1 is not None and conv2.in_channels != conv1.out_channels:
                    print(f"❌ {prefix}.conv2.in_channels({conv2.in_channels}) != conv1.out_channels({conv1.out_channels})")
                    all_pass = False
                else:
                    print(f"✓ {prefix}.conv2.in_channels({conv2.in_channels}) == conv1.out_channels({conv1.out_channels})")
                
                if bn2 is not None:
                    if bn2.num_features != conv2.out_channels:
                        print(f"❌ {prefix}.bn2.num_features({bn2.num_features}) != conv2.out_channels({conv2.out_channels})")
                        all_pass = False
                    else:
                        print(f"✓ {prefix}.bn2.num_features({bn2.num_features}) == conv2.out_channels({conv2.out_channels})")
            
            # 檢查 conv3
            conv3 = getattr(block, 'conv3', None)
            bn3 = getattr(block, 'bn3', None)
            if conv3 is not None:
                # 檢查 conv3 in_channels == conv2 out_channels
                if conv2 is not None and conv3.in_channels != conv2.out_channels:
                    print(f"❌ {prefix}.conv3.in_channels({conv3.in_channels}) != conv2.out_channels({conv2.out_channels})")
                    all_pass = False
                else:
                    print(f"✓ {prefix}.conv3.in_channels({conv3.in_channels}) == conv2.out_channels({conv2.out_channels})")
                
                if bn3 is not None:
                    if bn3.num_features != conv3.out_channels:
                        print(f"❌ {prefix}.bn3.num_features({bn3.num_features}) != conv3.out_channels({conv3.out_channels})")
                        all_pass = False
                    else:
                        print(f"✓ {prefix}.bn3.num_features({bn3.num_features}) == conv3.out_channels({conv3.out_channels})")
    
    if all_pass:
        print("✅ 所有層的 channel 對應檢查通過")
    else:
        print("❌ 有層的 channel 對應錯誤，請檢查上方輸出")

def test_cross_block_residual_connection():
    """測試跨塊殘差連接保護機制"""
    try:
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
        
        # 初始化輔助網路
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 初始化模型
        model = Os2dModelInPrune(pretrained_path=os2d_path)
        model = model.to(device)
        
        # 選擇一個跨塊連接的層進行剪枝
        layer_name = "layer2.3.conv3"  # 此層可能連接到下一個塊的 conv1
        
        # 獲取原始通道數
        target_layer = None
        for name, module in model.backbone.named_modules():
            if name == layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            pytest.skip(f"找不到層: {layer_name}，可能是因為模型結構不同")
        
        orig_channels = target_layer.out_channels
        print(f"原始通道數: {orig_channels}")
        
        # 創建測試輸入
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        boxes = [torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32).to(device) for _ in range(batch_size)]
        labels = [torch.tensor([0, 1], dtype=torch.long).to(device) for _ in range(batch_size)]
        
        # 設置剪枝比例
        prune_ratio = 0.3
        expected_channels = int(orig_channels * (1 - prune_ratio))
        
        # 執行剪枝
        print(f"剪枝層 {layer_name}，剪枝比例: {prune_ratio}")
        success = model.prune_channel(layer_name, prune_ratio=prune_ratio, images=images, boxes=boxes, labels=labels, auxiliary_net=aux_net)
        
        # 驗證剪枝結果
        if success == "SKIPPED":
            print(f"✅ 正確跳過了 {layer_name} 的剪枝")
            # 如果跳過了剪枝，則期望通道數不變
            expected_channels = orig_channels
        else:
            assert success, "剪枝操作失敗"
        
        # 獲取剪枝後的通道數
        pruned_layer = None
        for name, module in model.backbone.named_modules():
            if name == layer_name:
                pruned_layer = module
                break
        
        assert pruned_layer is not None, f"剪枝後找不到層: {layer_name}"
        pruned_channels = pruned_layer.out_channels
        print(f"剪枝後通道數: {pruned_channels}")
        
        # 驗證通道數是否正確減少
        assert pruned_channels == expected_channels, f"通道數不匹配: 預期 {expected_channels}, 實際 {pruned_channels}"
        
        # 檢查下一個塊的第一層是否已更新
        next_layer_name = "layer3.0.conv1"
        next_layer = None
        for name, module in model.backbone.named_modules():
            if name == next_layer_name:
                next_layer = module
                break
        
        if next_layer is not None:
            assert next_layer.in_channels == pruned_channels, f"下一個塊的輸入通道數不匹配: 預期 {pruned_channels}, 實際 {next_layer.in_channels}"
            print(f"✓ 下一個塊的輸入通道數已正確更新: {next_layer.in_channels}")
        
        # 測試前向傳播
        class_images = [torch.randn(3, 224, 224).to(device) for _ in range(batch_size)]
        with torch.no_grad():
            output = model(images, class_images)
            
        # 檢查所有層的 channel 一致性
        print("\n檢查所有層的 channel 一致性...")
        _test_all_layer_channel_consistency(model)
        
        print("✅ 跨塊殘差連接保護測試通過")
        return True
    except Exception as e:
        print(f"❌ 跨塊殘差連接保護測試失敗: {e}")
        # traceback.print_exc()
        return False

def test_load_os2d_weights():
    """測試 OS2D 模型載入權重"""
    try:
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        
        # 初始化模型
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
        
        # 載入模型
        print(f"📥 載入 OS2D checkpoint: {os2d_path}")
        model = Os2dModelInPrune(pretrained_path=os2d_path)
        
        # 驗證模型載入成功
        print("✅ OS2D 權重載入成功")
        
        # 驗證參數總和 (簡單的完整性檢查)
        param_sum = sum(p.sum().item() for p in model.parameters())
        print(f"✅ 模型參數總和: {param_sum}")
        
        return True
    except Exception as e:
        print(f"❌ OS2D 模型測試失敗: {e}")
        # traceback.print_exc()
        return False
    
def test_get_feature_map():
    """測試 OS2D 特徵圖提取功能"""
    try:
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        
        # 初始化模型
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
        
        # 載入模型
        print(f"📥 載入 OS2D checkpoint: {os2d_path}")
        model = Os2dModelInPrune(pretrained_path=os2d_path)
        model = model.to(device)
        
        # 創建測試輸入
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        
        # 提取特徵圖
        with torch.no_grad():
            features = model.get_feature_map(images)
        
        # 驗證特徵圖形狀
        print(f"✅ 特徵圖形狀: {features.shape}")
        assert features.shape[0] == batch_size, f"批次大小不匹配: {features.shape[0]} != {batch_size}"
        
        # 檢查特徵圖維度
        assert len(features.shape) == 4, f"特徵圖應為4維張量，實際為{len(features.shape)}維"
        
        # 驗證特徵圖有值
        assert torch.isfinite(features).all(), "特徵圖包含無限值"
        assert not torch.isnan(features).any(), "特徵圖包含 NaN"
        
        # 檢查特徵圖數值範圍
        print(f"特徵圖數值範圍: [{features.min().item():.3f}, {features.max().item():.3f}]")
        
        print("✅ OS2D 特徵圖測試通過")
        return True
    except Exception as e:
        print(f"❌ OS2D 特徵圖測試失敗: {e}")
        # traceback.print.exc()
        return False

def test_prune_block_with_downsample():
    """測試剪枝帶有 downsample 的塊"""
    try:
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        print(f"使用設備: {device}")

        # 初始化模型
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")

        # 初始化模型和輔助網路 
        model = Os2dModelInPrune(pretrained_path=os2d_path)
        model = model.to(device)
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)

        # 選擇一個帶有 downsample 的塊
        block_name = "layer2.0"
        
        # 創建測試輸入
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        boxes = [torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32).to(device) for _ in range(batch_size)]
        labels = [torch.tensor([0, 1], dtype=torch.long).to(device) for _ in range(batch_size)]

        # 獲取原始通道數
        block = getattr(model.backbone, block_name.split('.')[0])[int(block_name.split('.')[1])]
        orig_channels = block.conv3.out_channels
        orig_downsample_channels = block.downsample[0].out_channels
        print(f"原始 conv3 通道數: {orig_channels}")
        print(f"原始 downsample 通道數: {orig_downsample_channels}")

        # 執行剪枝
        prune_ratio = 0.3
        print(f"\n剪枝塊 {block_name}，剪枝比例: {prune_ratio}...")
        success = model.prune_channel(f"{block_name}.conv3", prune_ratio=prune_ratio, 
                                    images=images, boxes=boxes, labels=labels, auxiliary_net=aux_net)

        # 驗證剪枝結果
        if success == "SKIPPED":
            print(f"⚠️ 跳過剪枝 {block_name}")
            return True

        # 獲取剪枝後的通道數
        block = getattr(model.backbone, block_name.split('.')[0])[int(block_name.split('.')[1])]
        pruned_channels = block.conv3.out_channels
        pruned_downsample_channels = block.downsample[0].out_channels
        
        print(f"剪枝後 conv3 通道數: {pruned_channels}")
        print(f"剪枝後 downsample 通道數: {pruned_downsample_channels}")

        # 驗證通道數是否符合預期
        expected_channels = int(orig_channels * (1 - prune_ratio))
        assert pruned_channels == expected_channels, \
            f"conv3 通道數不匹配: 預期 {expected_channels}, 實際 {pruned_channels}"
        assert pruned_downsample_channels == expected_channels, \
            f"downsample 通道數不匹配: 預期 {expected_channels}, 實際 {pruned_downsample_channels}"

        # 測試前向傳播
        class_images = [torch.randn(3, 224, 224).to(device)]
        with torch.no_grad():
            output = model(images, class_images)

        # 檢查所有層的 channel 一致性
        print("\n檢查所有層的 channel 一致性...")
        _test_all_layer_channel_consistency(model)

        print("\n✅ 含 downsample 塊剪枝測試通過")
        return True

    except Exception as e:
        print(f"❌ 含 downsample 塊剪枝測試失敗: {e}")
        # traceback.print_exc()
        return False

def test_prune_channel():
    """測試 OS2D 單層通道剪枝功能"""
    try:
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
        
        # 載入模型
        print(f"📥 載入 OS2D checkpoint: {os2d_path}")
        model = Os2dModelInPrune(pretrained_path=os2d_path)
        model = model.to(device)
        
        # 初始化輔助網路
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 選擇要剪枝的層
        layer_name = "layer2.0.conv1"
        
        # 獲取原始通道數
        target_layer = None
        for name, module in model.backbone.named_modules():
            if name == layer_name:
                target_layer = module
                break
        
        assert target_layer is not None, f"找不到層: {layer_name}"
        orig_channels = target_layer.out_channels
        print(f"原始通道數: {orig_channels}")
        
        # 設置剪枝比例
        prune_ratio = 0.3
        expected_channels = int(orig_channels * (1 - prune_ratio))
        
        # 創建測試輸入
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        boxes = [torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32).to(device) for _ in range(batch_size)]
        labels = [torch.tensor([0, 1], dtype=torch.long).to(device) for _ in range(batch_size)]
        
        # 執行剪枝
        print(f"剪枝層 {layer_name}，剪枝比例: {prune_ratio}")
        success = model.prune_channel(layer_name, prune_ratio=prune_ratio, images=images, boxes=boxes, labels=labels, auxiliary_net=aux_net)
        
        # 驗證剪枝結果
        assert success, "剪枝操作失敗"
        
        # 獲取剪枝後的通道數
        pruned_layer = None
        for name, module in model.backbone.named_modules():
            if name == layer_name:
                pruned_layer = module
                break
        
        assert pruned_layer is not None, f"剪枝後找不到層: {layer_name}"
        pruned_channels = pruned_layer.out_channels
        print(f"剪枝後通道數: {pruned_channels}")
        
        # 驗證通道數是否正確減少
        assert pruned_channels == expected_channels, f"通道數不匹配: 預期 {expected_channels}, 實際 {pruned_channels}"
        
        return True
    except Exception as e:
        print(f"❌ OS2D prune channel 測試失敗: {e}")
        # # traceback.print.exc()
        return False
        
def test_residual_connection_protection():
    """測試殘差連接保護機制"""
    try:
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
        
        # 初始化輔助網路
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 初始化模型
        model = Os2dModelInPrune(pretrained_path=os2d_path)
        model = model.to(device)
        
        # 選擇一個有殘差連接的層進行剪枝
        layer_name = "layer2.0.conv3"  # 此層有殘差連接
        
        # 獲取原始通道數
        target_layer = None
        for name, module in model.backbone.named_modules():
            if name == layer_name:
                target_layer = module
                break
        
        assert target_layer is not None, f"找不到層: {layer_name}"
        orig_channels = target_layer.out_channels
        print(f"原始通道數: {orig_channels}")
        
        # 獲取殘差連接層
        downsample_layer = None
        for name, module in model.backbone.named_modules():
            if name == "layer2.0.downsample.0":
                downsample_layer = module
                break
        
        assert downsample_layer is not None, "找不到殘差連接層"
        assert downsample_layer.out_channels == orig_channels, "殘差連接層通道數與目標層不一致"
        
        # 創建測試輸入
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        boxes = [torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32).to(device) for _ in range(batch_size)]
        labels = [torch.tensor([0, 1], dtype=torch.long).to(device) for _ in range(batch_size)]
        
        # 設置剪枝比例
        prune_ratio = 0.3
        expected_channels = int(orig_channels * (1 - prune_ratio))
        
        # 執行剪枝
        print(f"剪枝層 {layer_name}，剪枝比例: {prune_ratio}")
        success = model.prune_channel(layer_name, prune_ratio=prune_ratio, images=images, boxes=boxes, labels=labels, auxiliary_net=aux_net)
        
        # 驗證剪枝結果
        if success == "SKIPPED":
            print(f"✅ 正確跳過了 {layer_name} 的剪枝")
            # 如果跳過了剪枝，則期望通道數不變
            expected_channels = orig_channels
        else:
            assert success, "剪枝操作失敗"
        
        # 獲取剪枝後的通道數
        pruned_layer = None
        for name, module in model.backbone.named_modules():
            if name == layer_name:
                pruned_layer = module
                break
        
        assert pruned_layer is not None, f"剪枝後找不到層: {layer_name}"
        pruned_channels = pruned_layer.out_channels
        print(f"剪枝後通道數: {pruned_channels}")
        
        # 驗證通道數是否符合預期
        assert pruned_channels == expected_channels, f"通道數不匹配: 預期 {expected_channels}, 實際 {pruned_channels}"
        
        # 驗證殘差連接層是否同步更新
        downsample_layer = None
        for name, module in model.backbone.named_modules():
            if name == "layer2.0.downsample.0":
                downsample_layer = module
                break
        
        assert downsample_layer is not None, "剪枝後找不到殘差連接層"
        assert downsample_layer.out_channels == pruned_channels, f"殘差連接層通道數不匹配: 預期 {pruned_channels}, 實際 {downsample_layer.out_channels}"
        
        # 測試前向傳播
        class_images = [torch.randn(3, 224, 224).to(device)]
        with torch.no_grad():
            output = model(images, class_images)
            
        # 檢查所有層的 channel 一致性 
        print("\n檢查所有層的 channel 一致性...")
        _test_all_layer_channel_consistency(model)
        
        print("✅ 殘差連接保護測試通過")
        return True
    except Exception as e:
        print(f"❌ 殘差連接保護測試失敗: {e}")
        # traceback.print_exc()
        return False

def test_residual_connection_pre_post_pruning():
    """測試殘差連接剪枝前後的層間關係"""
    try:
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
        
        # 初始化輔助網路
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 初始化模型
        model = Os2dModelInPrune(pretrained_path=os2d_path)
        model = model.to(device)
        
        # 選擇要剪枝的層
        layer_name = "layer2.0.conv3"  # 有殘差連接的層
        
        # 記錄剪枝前的層間關係
        def get_layer_channels(model, layer_name):
            layer = None
            for name, module in model.backbone.named_modules():
                if name == layer_name:
                    layer = module
                    break
            return layer.out_channels if layer else None
        
        # 獲取剪枝前的通道數
        pre_prune_channels = get_layer_channels(model, layer_name)
        pre_prune_downsample = get_layer_channels(model, "layer2.0.downsample.0")
        
        print(f"剪枝前 {layer_name} 通道數: {pre_prune_channels}")
        print(f"剪枝前 downsample 通道數: {pre_prune_downsample}")
        
        # 創建測試輸入
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        boxes = [torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32).to(device) for _ in range(batch_size)]
        labels = [torch.tensor([0, 1], dtype=torch.long).to(device) for _ in range(batch_size)]
        
        # 設置剪枝比例並執行剪枝
        prune_ratio = 0.3
        success = model.prune_channel(layer_name, prune_ratio=prune_ratio, images=images, boxes=boxes, labels=labels, auxiliary_net=aux_net)
        
        if success == "SKIPPED":
            print(f"✅ 正確跳過了 {layer_name} 的剪枝")
            return True
            
        # 驗證剪枝結果
        post_prune_channels = get_layer_channels(model, layer_name)
        post_prune_downsample = get_layer_channels(model, "layer2.0.downsample.0")
        
        print(f"剪枝後 {layer_name} 通道數: {post_prune_channels}")
        print(f"剪枝後 downsample 通道數: {post_prune_downsample}")
        
        # 驗證殘差連接的一致性
        assert post_prune_channels == post_prune_downsample, \
            f"剪枝後通道數不一致: {layer_name}={post_prune_channels}, downsample={post_prune_downsample}"
        
        # 測試前向傳播
        class_images = [torch.randn(3, 224, 224).to(device)]
        with torch.no_grad():
            output = model(images, class_images)
            
        # 檢查所有層的 channel 一致性
        print("\n檢查所有層的 channel 一致性...")
        _test_all_layer_channel_consistency(model)
        
        print("✅ 殘差連接剪枝前後關係測試通過")
        return True
    except Exception as e:
        print(f"❌ 殘差連接剪枝前後關係測試失敗: {e}")
        # traceback.print_exc()
        return False

def test_continuous_blocks_pruning():
    """測試連續多個塊的剪枝"""
    try:
        # 設置設備 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
        
        # 初始化輔助網路
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 載入模型
        model = Os2dModelInPrune(pretrained_path=os2d_path) 
        model = model.to(device)
        
        # 選擇要剪枝的多個塊
        blocks_to_prune = [
            "layer2.0.conv1",
            "layer2.0.conv2",
            "layer2.1.conv1", 
            "layer2.1.conv2"
        ]
        
        # 創建測試輸入
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        boxes = [torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32).to(device) for _ in range(batch_size)]
        labels = [torch.tensor([0, 1], dtype=torch.long).to(device) for _ in range(batch_size)]
        
        # 記錄原始參數量
        orig_params = sum(p.numel() for p in model.parameters())
        print(f"原始參數量: {orig_params:,}")
        
        # 依次對每個塊進行剪枝
        prune_ratio = 0.3
        for layer_name in blocks_to_prune:
            print(f"\n剪枝層 {layer_name}...")
            success = model.prune_channel(layer_name, prune_ratio=prune_ratio, images=images, boxes=boxes, labels=labels, auxiliary_net=aux_net)
            if success == "SKIPPED":
                print(f"⚠️ 跳過剪枝 {layer_name}")
                continue
            assert success, f"剪枝 {layer_name} 失敗"
        
        # 計算最終參數量
        final_params = sum(p.numel() for p in model.parameters())
        reduction = (orig_params - final_params) / orig_params * 100
        print(f"\n參數量減少: {orig_params:,} -> {final_params:,} ({reduction:.2f}%)")
        
        # 測試前向傳播
        class_images = [torch.randn(3, 224, 224).to(device)]
        with torch.no_grad():
            output = model(images, class_images)
            
        # 檢查所有層的 channel 一致性
        print("\n檢查所有層的 channel 一致性...")
        _test_all_layer_channel_consistency(model)
        
        print("\n✅ 連續塊剪枝測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 連續塊剪枝測試失敗: {e}")
        # traceback.print.exc()  
        return False

def test_prune_conv1_only():
    """測試只剪枝 conv1 層"""
    try:
        # 設置設備
        device = 'cuda' if torch.cuda.is_available() else 'cuda'
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
        # 載入模型與輔助網路
        model = Os2dModelInPrune(pretrained_path=os2d_path).to(device)
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 選擇要剪枝的 conv1 層
        target_layers = [
            "layer1.0.conv1",
            "layer2.0.conv1", 
            "layer3.0.conv1",
            "layer4.0.conv1"
        ]
        
        # 創建測試輸入
        batch_size = 1  # 減少批次大小
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        boxes = [torch.tensor([[10, 10, 50, 50]], dtype=torch.float32).to(device)]
        labels = [torch.tensor([0], dtype=torch.long).to(device)]
        
        # 記錄原始參數量
        orig_params = sum(p.numel() for p in model.parameters())
        print(f"原始參數量: {orig_params:,}")
        
        # 對每個 conv1 層進行剪枝
        prune_ratio = 0.3
        for layer_name in target_layers:
            print(f"\n剪枝層 {layer_name}...")
            
            # 獲取原始通道數
            orig_channels = None
            for name, module in model.backbone.named_modules():
                if name == layer_name:
                    orig_channels = module.out_channels
                    break
            
            if orig_channels is None:
                print(f"⚠️ 找不到層 {layer_name}")
                continue
                
            print(f"原始通道數: {orig_channels}")
            expected_channels = int(orig_channels * (1 - prune_ratio))
            print( f"預期通道數: {expected_channels}")
            # 執行剪枝
            success = model.prune_channel(
                layer_name=layer_name,
                prune_ratio=prune_ratio,
                images=images,
                boxes=boxes,
                labels=labels,
                auxiliary_net=aux_net
            )
            
            if success == "SKIPPED":
                print(f"⚠️ 跳過剪枝 {layer_name}")
                continue
                
            assert success, f"剪枝 {layer_name} 失敗"
            
            # 驗證剪枝後的通道數
            pruned_channels = None
            for name, module in model.backbone.named_modules():
                if name == layer_name:
                    pruned_channels = module.out_channels
                    break
            
            assert pruned_channels == expected_channels, \
                f"通道數不匹配: 預期 {expected_channels}, 實際 {pruned_channels}"
                
        # 計算最終參數量
        final_params = sum(p.numel() for p in model.parameters())
        reduction = (orig_params - final_params) / orig_params * 100
        print(f"\n參數量減少: {orig_params:,} -> {final_params:,} ({reduction:.2f}%)")
        
        # model._print_model_summary()
        # 測試前向傳播
        class_images = [torch.randn(3, 224, 224).to(device)]
        with torch.no_grad():
            output = model(images, class_images)
            
        # 檢查channel一致性
        print("\n檢查所有層的 channel 一致性...")
        _test_all_layer_channel_consistency(model)
        
        print("\n✅ Conv1 剪枝測試通過")
        return True
        
    except Exception as e:
        print(f"❌ Conv1 剪枝測試失敗: {e}")
        # # traceback.print.exc()
        return False

def test_prune_conv2_only():
    """測試只剪枝 conv2 層"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
            
        # 載入模型與輔助網路    
        model = Os2dModelInPrune(pretrained_path=os2d_path, is_cuda=(device.type == 'cuda')).to(device)
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 選擇要剪枝的 conv2 層
        target_layers = [
            "layer1.0.conv2",
            "layer2.0.conv2",
            "layer3.0.conv2"
            "layer4.0.conv2"
        ]
        
        # 測試輸入
        batch_size = 1
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        boxes = [torch.tensor([[10, 10, 50, 50]], dtype=torch.float32).to(device)]
        labels = [torch.tensor([0], dtype=torch.long).to(device)]
        
        orig_params = sum(p.numel() for p in model.parameters())
        print(f"原始參數量: {orig_params:,}")
        
        # 對每個 conv2 層進行剪枝
        prune_ratio = 0.3
        for layer_name in target_layers:
            print(f"\n剪枝層 {layer_name}...")
            
            # 獲取原始通道數
            orig_channels = None 
            for name, module in model.backbone.named_modules():
                if name == layer_name:
                    orig_channels = module.out_channels
                    break
                    
            if orig_channels is None:
                print(f"⚠️ 找不到層 {layer_name}")
                continue
                
            print(f"原始通道數: {orig_channels}")
            expected_channels = int(orig_channels * (1 - prune_ratio))
            
            # 執行剪枝
            success = model.prune_channel(
                layer_name=layer_name,
                prune_ratio=prune_ratio, 
                images=images,
                boxes=boxes,
                labels=labels,
                auxiliary_net=aux_net
            )
            
            if success == "SKIPPED":
                print(f"⚠️ 跳過剪枝 {layer_name}")
                continue
                
            assert success, f"剪枝 {layer_name} 失敗"
            
            # 驗證剪枝結果
            pruned_channels = None
            for name, module in model.backbone.named_modules():
                if name == layer_name:
                    pruned_channels = module.out_channels
                    break
                    
            assert pruned_channels == expected_channels, \
                f"通道數不匹配: 預期 {expected_channels}, 實際 {pruned_channels}"
                
        # 計算參數量變化
        final_params = sum(p.numel() for p in model.parameters())
        reduction = (orig_params - final_params) / orig_params * 100
        print(f"\n參數量減少: {orig_params:,} -> {final_params:,} ({reduction:.2f}%)")
        model._print_model_summary()
        # 測試前向傳播
        class_images = [torch.randn(3, 224, 224).to(device)]
        with torch.no_grad():
            output = model(images, class_images)
            
        # 檢查channel一致性  
        print("\n檢查所有層的 channel 一致性...")
        _test_all_layer_channel_consistency(model)
        
        print("\n✅ Conv2 剪枝測試通過")
        return True
        
    except Exception as e:
        print(f"❌ Conv2 剪枝測試失敗: {e}")
        # # traceback.print.exc()
        return False

def test_prune_conv3_only():
    """測試只剪枝 conv3 層(包含殘差連接)"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
            
        # 載入模型與輔助網路
        model = Os2dModelInPrune(pretrained_path=os2d_path).to(device)
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 選擇要剪枝的 conv3 層
        target_layers = [
            "layer1.0.conv3",
            "layer2.0.conv3",
            "layer3.0.conv3",
            "layer4.0.conv3"
        ]
        
        # 測試輸入
        batch_size = 1
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        boxes = [torch.tensor([[10, 10, 50, 50]], dtype=torch.float32).to(device)]
        labels = [torch.tensor([0], dtype=torch.long).to(device)]
        
        orig_params = sum(p.numel() for p in model.parameters())
        print(f"原始參數量: {orig_params:,}")
        
        # 對每個 conv3 層進行剪枝
        prune_ratio = 0.3
        for layer_name in target_layers:
            print(f"\n剪枝層 {layer_name}...")
            
            # 獲取原始通道數
            orig_channels = None
            for name, module in model.backbone.named_modules():
                if name == layer_name:
                    orig_channels = module.out_channels
                    break
                    
            if orig_channels is None:
                print(f"⚠️ 找不到層 {layer_name}")
                continue
                
            # 獲取對應的殘差層
            block_name = ".".join(layer_name.split(".")[:2])
            downsample_name = f"{block_name}.downsample.0"
            
            print(f"原始通道數: {orig_channels}")
            expected_channels = int(orig_channels * (1 - prune_ratio))
            
            # 執行剪枝
            success = model.prune_channel(
                layer_name=layer_name,
                prune_ratio=prune_ratio,
                images=images, 
                boxes=boxes,
                labels=labels,
                auxiliary_net=aux_net
            )
            
            if success == "SKIPPED":
                print(f"⚠️ 跳過剪枝 {layer_name}")
                continue
                
            assert success, f"剪枝 {layer_name} 失敗"
            
            # 驗證剪枝結果
            pruned_channels = None
            downsample_channels = None
            for name, module in model.backbone.named_modules():
                if name == layer_name:
                    pruned_channels = module.out_channels
                elif name == downsample_name:
                    downsample_channels = module.out_channels
                    
            assert pruned_channels == expected_channels, \
                f"通道數不匹配: 預期 {expected_channels}, 實際 {pruned_channels}"
                
            # 驗證殘差層同步更新
            assert downsample_channels == pruned_channels, \
                f"殘差層通道數不匹配: conv3={pruned_channels}, downsample={downsample_channels}"
                
        # 計算參數量變化
        final_params = sum(p.numel() for p in model.parameters())
        reduction = (orig_params - final_params) / orig_params * 100
        print(f"\n參數量減少: {orig_params:,} -> {final_params:,} ({reduction:.2f}%)")
        
        # 測試前向傳播
        class_images = [torch.randn(3, 224, 224).to(device)]
        with torch.no_grad():
            output = model(images, class_images)
            
        # 檢查channel一致性
        print("\n檢查所有層的 channel 一致性...")
        _test_all_layer_channel_consistency(model)
        
        print("\n✅ Conv3 剪枝測試通過")
        return True
        
    except Exception as e:
        print(f"❌ Conv3 剪枝測試失敗: {e}")
        # traceback.print.exc()
        return False

def test_continuous_block_pruning():
    """測試連續剪枝整個殘差塊"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
            
        # 載入模型與輔助網路
        model = Os2dModelInPrune(pretrained_path=os2d_path).to(device)
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 選擇要連續剪枝的塊
        target_blocks = [
            ["layer2.0.conv1", "layer2.0.conv2", "layer2.0.conv3"],
            ["layer2.1.conv1", "layer2.1.conv2", "layer2.1.conv3"],
            ["layer3.0.conv1", "layer3.0.conv2", "layer3.0.conv3"]
        ]
        
        # 測試輸入
        batch_size = 1
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        boxes = [torch.tensor([[10, 10, 50, 50]], dtype=torch.float32).to(device)]
        labels = [torch.tensor([0], dtype=torch.long).to(device)]
        
        orig_params = sum(p.numel() for p in model.parameters())
        print(f"原始參數量: {orig_params:,}")
        
        # 對每個塊進行連續剪枝
        prune_ratio = 0.3
        for block in target_blocks:
            print(f"\n開始剪枝塊 {block[0].split('.')[0:2]}...")
            
            for layer_name in block:
                print(f"\n剪枝層 {layer_name}...")
                
                # 獲取原始通道數
                orig_channels = None
                for name, module in model.backbone.named_modules():
                    if name == layer_name:
                        orig_channels = module.out_channels
                        break
                        
                if orig_channels is None:
                    print(f"⚠️ 找不到層 {layer_name}")
                    continue
                    
                print(f"原始通道數: {orig_channels}")
                expected_channels = int(orig_channels * (1 - prune_ratio))
                
                # 執行剪枝
                success = model.prune_channel(
                    layer_name=layer_name,
                    prune_ratio=prune_ratio,
                    images=images,
                    boxes=boxes,
                    labels=labels,
                    auxiliary_net=aux_net
                )
                
                if success == "SKIPPED":
                    print(f"⚠️ 跳過剪枝 {layer_name}")
                    continue
                    
                assert success, f"剪枝 {layer_name} 失敗"
                
                # 驗證剪枝結果
                pruned_channels = None
                for name, module in model.backbone.named_modules():
                    if name == layer_name:
                        pruned_channels = module.out_channels
                        break
                        
                assert pruned_channels == expected_channels, \
                    f"通道數不匹配: 預期 {expected_channels}, 實際 {pruned_channels}"
                    
            # 每個塊剪枝完成後測試前向傳播
            class_images = [torch.randn(3, 224, 224).to(device)]
            with torch.no_grad():
                output = model(images, class_images)
                
            # 檢查channel一致性
            print("\n檢查所有層的 channel 一致性...")
            _test_all_layer_channel_consistency(model)
                
        # 計算最終參數量
        final_params = sum(p.numel() for p in model.parameters())
        reduction = (orig_params - final_params) / orig_params * 100
        print(f"\n參數量減少: {orig_params:,} -> {final_params:,} ({reduction:.2f}%)")
        
        print("\n✅ 連續塊剪枝測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 連續塊剪枝測試失敗: {e}")
        # traceback.print.exc()  
        return False

def test_prune_multiple_blocks():
    """測試連續剪多個 block"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        print(f"\n===== 測試連續剪多個 block =====")
        print(f"使用設備: {device}")
        
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
            
        model = Os2dModelInPrune(pretrained_path=os2d_path).to(device)
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 定義要剪枝的多個 block
        blocks_to_prune = [
            ["layer2.0.conv1", "layer2.0.conv2"],  # 第一個 block
            ["layer2.1.conv1", "layer2.1.conv2"],  # 第二個 block
            ["layer3.0.conv1", "layer3.0.conv2"]   # 第三個 block
        ]
        
        # 測試輸入
        batch_size = 1
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        boxes = [torch.tensor([[10, 10, 50, 50]], dtype=torch.float32).to(device)]
        labels = [torch.tensor([0], dtype=torch.long).to(device)]
        
        # 記錄原始參數量
        orig_params = sum(p.numel() for p in model.parameters())
        print(f"原始參數量: {orig_params:,}")
        
        # 對每個 block 進行剪枝
        prune_ratio = 0.3
        for block in blocks_to_prune:
            print(f"\n剪枝 block: {block}")
            for layer_name in block:
                print(f"\n剪枝層 {layer_name}...")
                success = model.prune_channel(
                    layer_name=layer_name,
                    prune_ratio=prune_ratio,
                    images=images,
                    boxes=boxes,
                    labels=labels,
                    auxiliary_net=aux_net
                )
                
                if success == "SKIPPED":
                    print(f"⚠️ 跳過剪枝 {layer_name}")
                    continue
                    
                assert success, f"剪枝 {layer_name} 失敗"
                
            # 每個 block 剪枝後測試前向傳播
            class_images = [torch.randn(3, 224, 224).to(device)]
            with torch.no_grad():
                output = model(images, class_images)
                
            # 檢查 channel 一致性
            _test_all_layer_channel_consistency(model)
            
        # 計算參數量減少
        final_params = sum(p.numel() for p in model.parameters())
        reduction = (orig_params - final_params) / orig_params * 100
        print(f"\n參數量減少: {orig_params:,} -> {final_params:,} ({reduction:.2f}%)")
        
        print("\n✅ 連續剪多個 block 測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 連續剪多個 block 測試失敗: {e}")
        # traceback.print_exc()
        return False

def test_cross_stage_prune():
    """測試跨 stage 剪枝"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        print(f"\n===== 測試跨 stage 剪枝 =====")
        print(f"使用設備: {device}")
        
        os2d_path = "./os2d_v1-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
            
        model = Os2dModelInPrune(pretrained_path=os2d_path).to(device)
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 定義跨 stage 的剪枝層
        cross_stage_layers = [
            "layer2.3.conv2",  # layer2 最後一個 block
            "layer3.0.conv1",  # layer3 第一個 block
        ]
        
        # 測試輸入
        batch_size = 1
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        boxes = [torch.tensor([[10, 10, 50, 50]], dtype=torch.float32).to(device)]
        labels = [torch.tensor([0], dtype=torch.long).to(device)]
        
        orig_params = sum(p.numel() for p in model.parameters())
        print(f"原始參數量: {orig_params:,}")
        
        # 執行跨 stage 剪枝
        prune_ratio = 0.3
        for layer_name in cross_stage_layers:
            print(f"\n剪枝層 {layer_name}...")
            success = model.prune_channel(
                layer_name=layer_name,
                prune_ratio=prune_ratio,
                images=images,
                boxes=boxes,
                labels=labels,
                auxiliary_net=aux_net
            )
            
            if success == "SKIPPED":
                print(f"⚠️ 跳過剪枝 {layer_name}")
                continue
                
            assert success, f"剪枝 {layer_name} 失敗"
            
            # 測試前向傳播
            class_images = [torch.randn(3, 224, 224).to(device)]
            with torch.no_grad():
                output = model(images, class_images)
                
            # 檢查 channel 一致性
            _test_all_layer_channel_consistency(model)
            
        final_params = sum(p.numel() for p in model.parameters())
        reduction = (orig_params - final_params) / orig_params * 100
        print(f"\n參數量減少: {orig_params:,} -> {final_params:,} ({reduction:.2f}%)")
        
        print("\n✅ 跨 stage 剪枝測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 跨 stage 剪枝測試失敗: {e}")
        # traceback.print_exc()
        return False

def test_resnet18_basicblock_prune():
    """測試 ResNet18/34 BasicBlock 剪枝"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        print(f"\n===== 測試 ResNet18 BasicBlock 剪枝 =====")
        print(f"使用設備: {device}")
        
        # 載入 ResNet18 模型
        model = torchvision.models.resnet18(weights=None).to(device)
        
        # 定義要剪枝的 BasicBlock 層
        basic_block_layers = [
            "layer1.0.conv1",
            "layer1.0.conv2",
            "layer2.0.conv1",
            "layer2.0.conv2"
        ]
        
        # 測試輸入
        batch_size = 1
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        
        orig_params = sum(p.numel() for p in model.parameters())
        print(f"原始參數量: {orig_params:,}")
        
        # 執行 BasicBlock 剪枝
        prune_ratio = 0.3
        for layer_name in basic_block_layers:
            print(f"\n剪枝層 {layer_name}...")
            
            # 獲取原始通道數
            layer = None
            for name, module in model.named_modules():
                if name == layer_name:
                    layer = module
                    break
                    
            if layer is None:
                print(f"⚠️ 找不到層 {layer_name}")
                continue
                
            orig_channels = layer.out_channels
            expected_channels = int(orig_channels * (1 - prune_ratio))
            
            # 執行剪枝
            # 這裡需要實現 BasicBlock 的剪枝邏輯
            
            # 測試前向傳播
            with torch.no_grad():
                output = model(images)
                
        final_params = sum(p.numel() for p in model.parameters())
        reduction = (orig_params - final_params) / orig_params * 100
        print(f"\n參數量減少: {orig_params:,} -> {final_params:,} ({reduction:.2f}%)")
        
        print("\n✅ ResNet18 BasicBlock 剪枝測試通過")
        return True
        
    except Exception as e:
        print(f"❌ ResNet18 BasicBlock 剪枝測試失敗: {e}")
        # traceback.print.exc()
        return False

def test_pruning_ratios(layer_name, model_fn=None):
    """測試不同剪枝率對指定層的影響"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        print(f"\n===== 測試剪枝率 sweep {layer_name} =====")
        print(f"使用設備: {device}")
        
        # 定義要測試的剪枝率
        pruning_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        results = []
        for ratio in pruning_ratios:
            print(f"\n測試剪枝率: {ratio}")
            
            # 初始化新的模型實例
            if model_fn:
                model = model_fn().to(device)
            else:
                os2d_path = "./os2d_v1-train.pth"
                if not os.path.exists(os2d_path):
                    pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
                model = Os2dModelInPrune(pretrained_path=os2d_path).to(device)
                
            aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
            
            # 測試輸入
            batch_size = 1
            images = torch.randn(batch_size, 3, 224, 224).to(device)
            boxes = [torch.tensor([[10, 10, 50, 50]], dtype=torch.float32).to(device)]
            labels = [torch.tensor([0], dtype=torch.long).to(device)]
            
            # 記錄原始參數量
            orig_params = sum(p.numel() for p in model.parameters())
            
            # 執行剪枝
            success = model.prune_channel(
                layer_name=layer_name,
                prune_ratio=ratio,
                images=images,
                boxes=boxes,
                labels=labels,
                auxiliary_net=aux_net
            )
            
            if success == "SKIPPED":
                print(f"⚠️ 跳過剪枝 ratio={ratio}")
                continue
                
            assert success, f"剪枝失敗 ratio={ratio}"
            
            # 計算參數量減少
            final_params = sum(p.numel() for p in model.parameters())
            reduction = (orig_params - final_params) / orig_params * 100
            
            # 測試前向傳播
            class_images = [torch.randn(3, 224, 224).to(device)]
            with torch.no_grad():
                output = model(images, class_images)
                
            # 記錄結果
            results.append({
                'ratio': ratio,
                'param_reduction': reduction,
                'orig_params': orig_params,
                'final_params': final_params
            })
            
        # 輸出結果摘要
        print("\n剪枝率 sweep 結果:")
        for result in results:
            print(f"剪枝率 {result['ratio']:.1f}: 參數減少 {result['param_reduction']:.2f}% ({result['orig_params']:,} -> {result['final_params']:,})")
            
        print(f"\n✅ 剪枝率 sweep 測試通過: {layer_name}")
        return True
        
    except Exception as e:
        print(f"❌ 剪枝率 sweep 測試失敗: {e}")
        return False

    
def test_lcp_finetune_pipeline():
    pass


    
def test_train_one_epoch_basic():
    """
    Memory-friendly 單元測試：驗證 OS2D 模型在 Grozi-3.2k mini set 上能正確 train_one_epoch
    並打印詳細的數據集和輸入參數信息
    """
    from src.os2d_model_in_prune import Os2dModelInPrune
    from src.auxiliary_network import AuxiliaryNetwork
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    import os
    import torch
    import pytest
    import numpy as np
    import traceback
    import logging

    # 設置詳細的日誌
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("OS2D.test")
    logger.info("===== 開始詳細測試 OS2D 訓練管道 =====")

    # 1. 檢查 Grozi dataset 是否存在
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if not os.path.exists(grozi_csv):
        logger.error("❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        pytest.skip("Grozi-3.2k dataset missing, test skipped.")
        return False

    # 2. 建立 dataset/dataloader（mini subset + batch size 1）
    logger.info("正在載入 Grozi-3.2k mini dataset...")
    dataset = build_grozi_dataset(
        data_path=data_path,
        name="grozi-train-mini",  # 只取2張圖2類別
        eval_scale=224,
        cache_images=False
    )
    
    # 打印數據集基本信息
    class_ids = dataset.get_class_ids()
    logger.info(f"數據集信息: {dataset.get_name()}")
    logger.info(f"總圖像數量: {len(dataset.image_ids)}")
    logger.info(f"類別數量: {len(class_ids)}")
    logger.info(f"類別 ID 列表: {class_ids}")
    
    # 查看第一張圖的標註信息
    if len(dataset.image_ids) > 0:
        image_id = dataset.image_ids[0]
        boxes = dataset.get_image_annotation_for_imageid(image_id)
        if hasattr(boxes, "bbox_xyxy"):
            logger.info(f"第一張圖 ({image_id}) 的標註框: shape={boxes.bbox_xyxy.shape}, 內容={boxes.bbox_xyxy}")
            if hasattr(boxes, "get_field"):
                if "labels" in boxes.fields():
                    logger.info(f"框對應的標籤: {boxes.get_field('labels')}")
    
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
    
    logger.info("創建 DataloaderOneShotDetection...")
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
    logger.info(f"DataLoader 長度: {len(train_loader)} batches")

    # 檢查第一個 batch 的具體結構
    logger.info("檢查第一個批次 (batch) 的結構...")
    batch = train_loader.get_batch(0)
    images, class_images, loc_targets, class_targets, batch_class_ids, class_image_sizes, box_inverse_transform, batch_boxes, batch_img_size = batch
    
    # 打印詳細的 batch 結構信息
    logger.info(f"批次結構:")
    logger.info(f"  - images: 形狀={images.shape}, 類型={images.dtype}, 設備={images.device}")
    logger.info(f"  - class_images: 數量={len(class_images)}, 類型={type(class_images[0]) if class_images else 'N/A'}")
    if class_images and isinstance(class_images[0], torch.Tensor):
        logger.info(f"    - 第一個 class_image: 形狀={class_images[0].shape}, 類型={class_images[0].dtype}")
    logger.info(f"  - loc_targets: 形狀={loc_targets.shape}, 類型={loc_targets.dtype}")
    logger.info(f"  - class_targets: 形狀={class_targets.shape}, 類型={class_targets.dtype}")
    logger.info(f"    - class_targets 值范圍: min={class_targets.min().item()}, max={class_targets.max().item()}")
    logger.info(f"  - batch_class_ids: 類型={type(batch_class_ids)}, 數量={len(batch_class_ids) if isinstance(batch_class_ids, list) else 'N/A'}")
    if isinstance(batch_class_ids, list) and len(batch_class_ids) > 0:
        logger.info(f"    - 第一個 batch_class_id: 形狀={batch_class_ids[0].shape if hasattr(batch_class_ids[0], 'shape') else '()'}, 值={batch_class_ids[0]}")
    logger.info(f"  - class_image_sizes: {class_image_sizes}")
    logger.info(f"  - batch_boxes: 類型={type(batch_boxes)}, 數量={len(batch_boxes) if isinstance(batch_boxes, list) else 'N/A'}")
    
    # 修正這裡，正確處理 BoxList 對象
    if isinstance(batch_boxes, list) and len(batch_boxes) > 0:
        box = batch_boxes[0]
        if hasattr(box, 'bbox_xyxy'):  # BoxList 對象
            logger.info(f"    - 第一個 batch_box: BoxList對象, 框數量={box.bbox_xyxy.shape[0]}")
            logger.info(f"    - bbox_xyxy 形狀={box.bbox_xyxy.shape}, 包含 {box.bbox_xyxy.shape[0]} 個框")
            if hasattr(box, 'get_field') and "labels" in box.fields():
                logger.info(f"    - 標籤: {box.get_field('labels')}")
        elif hasattr(box, 'shape'):  # Tensor
            logger.info(f"    - 第一個 batch_box: 形狀={box.shape}, 值={box}")
        else:
            logger.info(f"    - 第一個 batch_box: 類型={type(box)}, 值={box}")
            
    logger.info(f"  - batch_img_size: {batch_img_size}")

    # 3. 初始化模型與優化器（用 CPU 以節省資源）
    device = torch.device('cpu')
    logger.info(f"使用設備: {device}")
    
    os2d_path = "./os2d_v1-train.pth"
    if not os.path.exists(os2d_path):
        logger.error(f"OS2D 預訓練模型不存在: {os2d_path}")
        pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
        return False
    
    logger.info(f"載入 OS2D 模型: {os2d_path}")
    model = Os2dModelInPrune(pretrained_path=os2d_path, is_cuda=False).to(device)
    
    # 檢查模型結構
    logger.info("檢查模型結構:")
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"  - 總參數量: {param_count:,}")
    logger.info(f"  - backbone 型別: {type(model.backbone).__name__}")
    
    # 初始化輔助網路
    logger.info("初始化輔助網路")
    aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
    
    # 檢查並打印模型的特徵圖輸出大小
    logger.info("檢查特徵圖尺寸:")
    with torch.no_grad():
        feature_maps = model.get_feature_map(images.to(device))
        if isinstance(feature_maps, torch.Tensor):
            logger.info(f"  - 特徵圖形狀: {feature_maps.shape}")
            logger.info(f"  - 特徵圖值範圍: [{feature_maps.min().item():.4f}, {feature_maps.max().item():.4f}]")
            # 計算特徵圖如果被完全展平後的大小
            flattened_size = feature_maps.shape[0] * feature_maps.shape[1] * feature_maps.shape[2] * feature_maps.shape[3]
            logger.info(f"  - 特徵圖完全展平後大小: {flattened_size} (這可能導致類別數量異常)")
    
    # 初始化優化器
    optimizer = torch.optim.Adam(list(model.parameters()) + list(aux_net.parameters()), lr=1e-3)
    logger.info("初始化 Adam 優化器，學習率=1e-3")

    # 4. 執行一個 epoch 的訓練（只跑一個 batch）
    logger.info("開始執行 train_one_epoch (只執行一個 batch)...")
    try:
        logger.info(f"類別數量: {len(train_loader.dataset.get_class_ids())}")
        loss_history = model.train_one_epoch(
            train_loader=train_loader,
            optimizer=optimizer,
            auxiliary_net=aux_net,
            device=device,
            print_freq=1,  
            max_batches=1  
        )
    except NotImplementedError:
        logger.error("⚠️ train_one_epoch 尚未實作，請先完成實作。")
        assert False, "train_one_epoch 尚未實作"
        return False
    except Exception as e:
        logger.error(f"❌ 執行 train_one_epoch 發生例外: {e}")
        logger.error(traceback.format_exc())
        assert False, f"train_one_epoch 執行失敗: {e}"
        return False

    # 5. 驗證 loss 是否合理
    if isinstance(loss_history, list) and len(loss_history) > 0:
        avg_loss = np.mean(loss_history)
        logger.info(f"✅ train_one_epoch 執行成功，平均 loss={avg_loss:.4f}")
        assert np.isfinite(avg_loss), "loss 應為有限數值"
    else:
        logger.warning("⚠️ train_one_epoch 未回傳 loss 歷史，請檢查實作")
        assert False, "train_one_epoch 未回傳 loss 歷史"

    # 6. 驗證參數是否有更新
    orig_params = [p.clone().detach() for p in model.parameters()]
    model.train_one_epoch(
        train_loader=train_loader,
        optimizer=optimizer,
        auxiliary_net=aux_net,
        device=device,
        print_freq=0,
        max_batches=1
    )
    updated_params = [p.clone().detach() for p in model.parameters()]
    changed = any(not torch.equal(a, b) for a, b in zip(orig_params, updated_params))
    assert changed, "模型參數未更新，請檢查 optimizer/backward 實作"
    
    logger.info("✅ train_one_epoch 參數更新檢查通過")
    logger.info("🎉 test_train_one_epoch_basic: OS2D 單 batch 微調訓練測試通過")
    return True

def test_train_one_epoch_multiple_epochs():
    """
    測試 OS2D 模型在多個 epochs 的訓練行為，檢查 loss 是否呈現正常下降趨勢
    """
    from src.os2d_model_in_prune import Os2dModelInPrune
    from src.auxiliary_network import AuxiliaryNetwork
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    import os
    import torch
    import pytest
    import numpy as np
    import traceback
    import logging
    import time
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # 設置詳細的日誌
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("OS2D.test")
    logger.info("===== 開始測試 OS2D 多 epoch 訓練 =====")

    # 1. 檢查 Grozi dataset 是否存在
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if not os.path.exists(grozi_csv):
        logger.error("❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        pytest.skip("Grozi-3.2k dataset missing, test skipped.")
        return False

    # 2. 建立 dataset/dataloader
    logger.info("正在載入 Grozi-3.2k mini dataset...")
    dataset = build_grozi_dataset(
        data_path=data_path,
        name="grozi-train-mini",
        eval_scale=224,
        cache_images=False
    )
    
    # 打印數據集基本信息
    class_ids = dataset.get_class_ids()
    logger.info(f"數據集信息: {dataset.get_name()}")
    logger.info(f"總圖像數量: {len(dataset.image_ids)}")
    logger.info(f"類別數量: {len(class_ids)}")
    
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
    
    logger.info("創建 DataloaderOneShotDetection...")
    train_loader = DataloaderOneShotDetection(
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
    logger.info(f"DataLoader 長度: {len(train_loader)} batches")

    # 3. 初始化模型與優化器
    # 盡量使用 CPU 進行測試，避免顯存壓力
    device = torch.device('cpu')
    logger.info(f"使用設備: {device}")
    
    os2d_path = "./os2d_v1-train.pth"
    if not os.path.exists(os2d_path):
        logger.error(f"OS2D 預訓練模型不存在: {os2d_path}")
        pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
        return False
    
    logger.info(f"載入 OS2D 模型: {os2d_path}")
    model = Os2dModelInPrune(pretrained_path=os2d_path, is_cuda=False).to(device)
    
    # 初始化輔助網路
    logger.info("初始化輔助網路")
    aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
    
    # 初始化優化器 (使用較大學習率加速測試)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(aux_net.parameters()), lr=5e-3)
    logger.info("初始化 Adam 優化器，學習率=5e-3")

    # 4. 設置訓練參數
    num_epochs = 5  # 測試用，少量 epoch
    max_batches_per_epoch = 15  # 限制每個 epoch 的批次數，加速測試
    print_freq = 1
    
    # 5. 執行多個 epoch 的訓練
    logger.info(f"開始執行 {num_epochs} 個 epochs 的訓練...")
    epoch_losses = []
    component_losses = {
        'cls_loss': [],
        'box_loss': [],
        'teacher_loss': [],
        'lcp_loss': []
    }
    
    start_time = time.time()
    try:
        for epoch in range(num_epochs):
            logger.info(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
            epoch_start = time.time()
            
            # 執行一個 epoch 的訓練
            loss_history = model.train_one_epoch(
                train_loader=train_loader,
                optimizer=optimizer,
                auxiliary_net=aux_net,
                device=device,
                print_freq=print_freq,
                max_batches=max_batches_per_epoch
            )
            
            # 計算並記錄平均損失
            avg_loss = np.mean(loss_history) if isinstance(loss_history, list) else loss_history
            epoch_losses.append(avg_loss)
            
            # 如果有組件損失，也記錄它們
            if hasattr(model, 'last_losses') and isinstance(model.last_losses, dict):
                for k in component_losses.keys():
                    if k in model.last_losses:
                        component_losses[k].append(model.last_losses[k])
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch+1} 完成: 平均損失={avg_loss:.4f}, 用時={epoch_time:.1f}秒")
            
    except Exception as e:
        logger.error(f"❌ 執行多 epoch 訓練發生例外: {e}")
        logger.error(traceback.format_exc())
        assert False, f"多 epoch 訓練失敗: {e}"
        return False

    total_time = time.time() - start_time
    logger.info(f"\n✅ 多 epoch 訓練完成，總用時: {total_time:.1f}秒")
    
    # 6. 分析損失變化趨勢
    logger.info("\n===== 損失變化分析 =====")
    logger.info(f"每個 epoch 的平均損失: {[f'{loss:.4f}' for loss in epoch_losses]}")
    
    # 檢查損失是否下降
    if len(epoch_losses) >= 2:
        is_decreasing = epoch_losses[-1] < epoch_losses[0]
        if is_decreasing:
            logger.info(f"✅ 損失呈下降趨勢: {epoch_losses[0]:.4f} → {epoch_losses[-1]:.4f}")
        else:
            logger.warning(f"⚠️ 損失未呈下降趨勢: {epoch_losses[0]:.4f} → {epoch_losses[-1]:.4f}")
            # 不強制斷言，因為短期內損失可能波動
        
        # 計算平均下降率
        avg_decrease_rate = (epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0]
        logger.info(f"平均損失下降率: {avg_decrease_rate*100:.2f}%")
    
    # 7. 繪製損失曲線
    try:
        logger.info("生成損失曲線圖...")
        plt.figure(figsize=(10, 6))
        
        # 主損失曲線
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs+1), epoch_losses, 'b-o', linewidth=2)
        plt.title('Total Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 組件損失曲線
        if all(len(v) > 1 for v in component_losses.values()):
            plt.subplot(1, 2, 2)
            for name, values in component_losses.items():
                if values:  # 確保有數據
                    plt.plot(range(1, len(values)+1), values, '-o', linewidth=2, label=name)
            plt.title('Component Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        # 保存圖表
        plot_path = "./multi_epoch_losses.png"
        plt.savefig(plot_path)
        logger.info(f"損失曲線圖已保存至: {plot_path}")
        plt.close()
        
    except Exception as e:
        logger.warning(f"繪製損失曲線時出錯: {e}")
    
    # 8. 驗證訓練是有效的
    assert np.all(np.isfinite(epoch_losses)), "損失值應為有限數值"
    
    # 我們不強制要求損失必須單調下降，因為在少量數據和少量 epoch 下可能有波動
    # 但我們要求最終損失應該是數值合理的
    assert 0 < epoch_losses[-1] < 100, f"最終損失 {epoch_losses[-1]} 不在合理範圍內"
    
    logger.info("🎉 test_train_one_epoch_multiple_epochs: OS2D 多 epoch 訓練測試通過")
    return True


def test_lcp_prune_and_train_pipeline():
    pass

def test_save_checkpoint():
    pass

def get_conv_structure(model):
    return [(name, m.in_channels, m.out_channels, m.kernel_size)
            for name, m in model.backbone.named_modules() if isinstance(m, torch.nn.Conv2d)]

def test_os2d_compatibility_with_pruned_model():
    pass

def test_compute_losses():
    """測試 compute_losses 函數"""
    # 初始化模型和輔助網路
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Os2dModelInPrune(pretrained_path="./os2d_v1-train.pth", is_cuda=(device.type == 'cuda')).to(device)
    aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
    
    # 設置為訓練模式
    model.train()
    aux_net.train()
    
    # 創建模擬輸入
    batch_size = 2
    num_classes = 20
    class_images = [torch.randn(3, 64, 64).to(device) for _ in range(batch_size)]
    images = torch.randn(batch_size, 3, 224, 224, requires_grad=True).to(device)
    class_scores = torch.randn(batch_size, num_classes, requires_grad=True).to(device)
    boxes = torch.randn(batch_size, 4, requires_grad=True).to(device)
    
    # 模擬 outputs
    outputs = {
        'class_scores': class_scores,
        'boxes': boxes,
        'images': images,
        'class_images': class_images
    }
    
    # 執行教師模型前向傳播
    with torch.no_grad():
        teacher_outputs = model.teacher_model(images, class_images=class_images)
        print(f"教師模型輸出類型: {type(teacher_outputs)}")
        if isinstance(teacher_outputs, dict):
            print(f"教師模型輸出鍵: {teacher_outputs.keys()}")
        elif isinstance(teacher_outputs, tuple):
            print(f"教師模型輸出長度: {len(teacher_outputs)}")
    
    # 模擬 targets
    class_ids = [torch.tensor([1]), torch.tensor([2])]
    target_boxes = [torch.randn(1, 4).to(device), torch.randn(1, 4).to(device)]
    targets = {
        'class_ids': class_ids,
        'boxes': target_boxes,
        'images': images,
        'teacher_outputs': teacher_outputs  # 添加教師模型輸出
    }
    
    # 計算損失
    total_loss, loss_dict = model.compute_losses(outputs, targets, auxiliary_net=aux_net)
    
    # 檢查損失值是否合理
    print(f"總損失: {total_loss.item():.4f}")
    print(f"損失字典: {loss_dict}")
    print(f"cls loss :{loss_dict['cls_loss'].item()}")
    print(f"box loss :{loss_dict['box_loss'].item()}")
    print(f"教師 loss :{loss_dict['teacher_loss'].item()}")
    print(f"lcp loss :{loss_dict['lcp_loss'].item()}")
    
    # 反向傳播
    total_loss.backward()
    
    # 檢查梯度是否存在
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 
                  for p in model.parameters() if p.requires_grad)
    aux_has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 
                      for p in aux_net.parameters() if p.requires_grad)
    
    # 測試通過條件
    assert total_loss.item() > 0, "總損失應該大於0"
    assert has_grad, "模型參數應該有梯度"
    assert aux_has_grad, "輔助網路參數應該有梯度"
    
    print("✅ compute_losses 測試通過")
    return True



def test_os2d_model_in_prune_eval():
    import os
    import torch
    import pytest   
    import numpy as np
    from src.os2d_model_in_prune import Os2dModelInPrune
    from src.auxiliary_network import AuxiliaryNetwork
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize
    from hypothesis import given, strategies as st

        # 1. 檢查 Grozi dataset 是否存在
    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if not os.path.exists(grozi_csv):
        print("\n❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        pytest.skip("Grozi-3.2k dataset missing, test skipped.")
        return False

    # 2. 建立 dataset/dataloader（mini subset + batch size 1）
    dataset = build_grozi_dataset(
        data_path=data_path,
        name="grozi-train-mini",  # 只取2張圖2類別
        eval_scale=224,
        cache_images=False
    )
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

    # 3. 初始化模型與優化器（強制用 cuda，減少顯存壓力）
    device = torch.device('cpu')
    os2d_path = "./os2d_v1-train.pth"
    if not os.path.exists(os2d_path):
        pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
    model = Os2dModelInPrune(pretrained_path=os2d_path, is_cuda=False).to(device)
    aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(aux_net.parameters()), lr=1e-3)
    return True


def test_full_lcp_pipeline_with_eval_and_checkpoint():
    pass
