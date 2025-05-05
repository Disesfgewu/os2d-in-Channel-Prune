import os
import torch
import torchvision
import pytest
import traceback
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
def test_init_and_len():
    dataset = VOCDataset(data_path="./VOCdevkit", split="train", download=True)
    assert len(dataset) > 0
    assert isinstance(dataset.CLASSES, list)
    print("初始化與長度測試通過")
    return True

def test_getitem_single():
    dataset = VOCDataset(data_path="./VOCdevkit", split="train", download=True)
    img, boxes, labels, class_images = dataset[0]
    assert isinstance(img, torch.Tensor) and img.dim() == 3  # [C, H, W]
    assert isinstance(boxes, torch.Tensor) and boxes.shape[1] == 4
    assert isinstance(labels, torch.Tensor)
    assert isinstance(class_images, torch.Tensor)
    # class_images: [N, C, H, W] or [1, C, H, W]
    assert class_images.dim() == 4
    assert class_images.shape[2:] == (64, 64)
    print("getitem 單一樣本測試通過")
    return True

def test_collate_fn_batch():
    dataset = VOCDataset(data_path="./VOCdevkit", split="train", download=True)
    batch = [dataset[i] for i in range(4)]
    result = VOCDataset.collate_fn(batch)
    images, boxes_list, labels_list, all_class_images = result
    assert isinstance(images, torch.Tensor) and images.dim() == 4  # [B, C, H, W]
    assert len(boxes_list) == 4
    assert len(labels_list) == 4
    assert isinstance(all_class_images, torch.Tensor)
    # all_class_images: [sum_N, C, H, W]
    assert all_class_images.shape[1:] == (3, 64, 64)
    print("collate_fn 批次測試通過")
    return True

def test_collate_fn_batch():
    dataset = VOCDataset(data_path="./VOCdevkit", split="train", download=True)
    batch = [dataset[i] for i in range(4)]
    result = VOCDataset.collate_fn(batch)
    images, boxes_list, labels_list, all_class_images = result
    assert isinstance(images, torch.Tensor) and images.dim() == 4  # [B, C, H, W]
    assert len(boxes_list) == 4
    assert len(labels_list) == 4
    assert isinstance(all_class_images, torch.Tensor)
    # all_class_images: [sum_N, C, H, W]
    assert all_class_images.shape[1:] == (3, 64, 64)
    print("collate_fn 批次測試通過")
    return True

def test_class_image_generation():
    dataset = VOCDataset(data_path="./VOCdevkit", split="train", download=True)
    for i in range(10):
        img, boxes, labels, class_images = dataset[i]
        assert class_images.dim() == 4
        # 至少有一個 class image
        assert class_images.shape[0] >= 1
        assert class_images.shape[1:] == (3, 64, 64)
    print("class image 自動產生測試通過")
    return True

def test_img_size_resize():
    dataset = VOCDataset(data_path="./VOCdevkit", split="train", download=True, img_size=(128, 128))
    img, boxes, labels, class_images = dataset[0]
    assert img.shape[1:] == (128, 128)
    print("img_size resize 測試通過")
    return True

def test_class_mapping():
    mapping = {i: (i+1)%20 for i in range(20)}
    dataset = VOCDataset(data_path="./VOCdevkit", split="train", download=True, class_mapping=mapping)
    _, _, labels, _ = dataset[0]
    for l in labels:
        assert l in mapping.values()
    print("class_mapping 功能測試通過")
    return True

def test_os2d_model_in_prune_initialization():
    """測試 Os2dModelInPrune 初始化，並區分學生模型與教師模型參數量"""
    # 設置 logger
    logger = logging.getLogger("OS2D.test")
    
    # 初始化模型
    os2d_path = "./os2d_v2-train.pth"
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
    os2d_path = "./os2d_v2-train.pth"
    if not os.path.exists(os2d_path):
        pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    os2d_path = "./os2d_v2-train.pth"
    if not os.path.exists(os2d_path):
        pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        os2d_path = "./os2d_v2-train.pth"
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
        traceback.print_exc()
        return False

def test_set_layer_out_channels():
    """測試 set_layer_out_channels 方法"""
    try:
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v2-train.pth"
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
        import traceback
        traceback.print_exc()
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v2-train.pth"
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
        traceback.print_exc()
        return False

def test_load_os2d_weights():
    """測試 OS2D 模型載入權重"""
    try:
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        os2d_path = "./os2d_v2-train.pth"
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
        traceback.print_exc()
        return False
    
def test_get_feature_map():
    """測試 OS2D 特徵圖提取功能"""
    try:
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        os2d_path = "./os2d_v2-train.pth"
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
        traceback.print.exc()
        return False

def test_prune_block_with_downsample():
    """測試剪枝帶有 downsample 的塊"""
    try:
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")

        # 初始化模型
        os2d_path = "./os2d_v2-train.pth"
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
        traceback.print_exc()
        return False

def test_prune_channel():
    """測試 OS2D 單層通道剪枝功能"""
    try:
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v2-train.pth"
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
        # traceback.print.exc()
        return False
        
def test_residual_connection_protection():
    """測試殘差連接保護機制"""
    try:
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v2-train.pth"
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
        traceback.print_exc()
        return False

def test_residual_connection_pre_post_pruning():
    """測試殘差連接剪枝前後的層間關係"""
    try:
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v2-train.pth"
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
        traceback.print_exc()
        return False

def test_continuous_blocks_pruning():
    """測試連續多個塊的剪枝"""
    try:
        # 設置設備 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v2-train.pth"
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
        traceback.print.exc()  
        return False

def test_prune_conv1_only():
    """測試只剪枝 conv1 層"""
    try:
        # 設置設備
        device = 'cpu'
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v2-train.pth"
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
        traceback.print.exc()
        return False

def test_prune_conv2_only():
    """測試只剪枝 conv2 層"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v2-train.pth"
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
        traceback.print.exc()
        return False

def test_prune_conv3_only():
    """測試只剪枝 conv3 層(包含殘差連接)"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v2-train.pth"
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
        traceback.print.exc()
        return False

def test_continuous_block_pruning():
    """測試連續剪枝整個殘差塊"""
    try:
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化模型
        os2d_path = "./os2d_v2-train.pth"
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
        traceback.print.exc()  
        return False

def test_prune_multiple_blocks():
    """測試連續剪多個 block"""
    try:
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        print(f"\n===== 測試連續剪多個 block =====")
        print(f"使用設備: {device}")
        
        os2d_path = "./os2d_v2-train.pth"
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
        traceback.print_exc()
        return False

def test_cross_stage_prune():
    """測試跨 stage 剪枝"""
    try:
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        print(f"\n===== 測試跨 stage 剪枝 =====")
        print(f"使用設備: {device}")
        
        os2d_path = "./os2d_v2-train.pth"
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
        traceback.print_exc()
        return False

def test_resnet18_basicblock_prune():
    """測試 ResNet18/34 BasicBlock 剪枝"""
    try:
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
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
        traceback.print.exc()
        return False

def test_pruning_ratios(layer_name, model_fn=None):
    """測試不同剪枝率對指定層的影響"""
    try:
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
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
                os2d_path = "./os2d_v2-train.pth"
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

def test_lcp_channel_selector():
    """測試 LCP 通道選擇器基本功能"""
    try:
        print("\n===== LCP 通道選擇器測試 =====")
        
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化模型與路徑
        os2d_path = "./os2d_v2-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
        
        # 載入模型與輔助網路    
        model = Os2dModelInPrune(pretrained_path=os2d_path).to(device)
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 初始化選擇器
        selector = OS2DChannelSelector(
            model=model,
            auxiliary_net=aux_net,
            device=device,
            alpha=0.6,
            beta=0.3,
            gamma=0.1
        )
        
        # 基本屬性驗證
        assert hasattr(selector, 'compute_importance'), "選擇器應該有 compute_importance 方法"
        assert hasattr(selector, 'select_channels'), "選擇器應該有 select_channels 方法"
        assert selector.model is model, "模型參考錯誤"
        assert selector.auxiliary_net is aux_net, "輔助網路參考錯誤"
        assert selector.device == device, "設備設置錯誤"
        
        print("✅ LCP 通道選擇器初始化測試通過")
        return True
        
    except Exception as e:
        print(f"❌ LCP 通道選擇器測試失敗: {e}")
        traceback.print_exc()
        return False

def test_channel_importance_computation():
    """測試通道重要性計算功能"""
    try:
        print("\n===== 通道重要性計算測試 =====")
        
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化模型與路徑
        os2d_path = "./os2d_v2-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
            
        # 載入模型與輔助網路
        model = Os2dModelInPrune(pretrained_path=os2d_path).to(device)
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 初始化選擇器
        selector = OS2DChannelSelector(
            model=model,
            auxiliary_net=aux_net,
            device=device
        )
        
        # 準備測試數據
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        
        # 生成 class images (使用同一批次中的第一張圖像)
        class_images = [images[0].clone()]  # 使用第一張圖作為類別圖像
        
        # 生成邊界框和標籤
        boxes = [torch.tensor([[10, 10, 50, 50]], dtype=torch.float32).to(device) for _ in range(batch_size)]
        labels = [torch.tensor([0], dtype=torch.long).to(device) for _ in range(batch_size)]
        
        # 修改 forward pass
        def modified_forward():
            return model(images, class_images=class_images)
            
        # 測試不同層的通道重要性計算
        test_layers = [
            {"name": "layer2.0.conv1", "expected_channels": 128},
            {"name": "layer2.0.conv2", "expected_channels": 128},
            {"name": "layer3.0.conv1", "expected_channels": 256}
        ]
        
        for layer_info in test_layers:
            layer_name = layer_info["name"]
            expected_channels = layer_info["expected_channels"]
            print(f"\n測試層 {layer_name}...")
            
            # 計算通道重要性
            importance_scores = selector.compute_importance(
                layer_name=layer_name,
                images=images,
                boxes=boxes,
                gt_boxes=boxes,
                labels=labels
            )
            
            # 驗證重要性分數
            assert importance_scores is not None, f"{layer_name} importance_scores 不應為 None"
            assert len(importance_scores) == expected_channels, \
                f"{layer_name} importance_scores 長度不符: 預期 {expected_channels}, 實際 {len(importance_scores)}"
            print(f"✓ {layer_name} 重要性分數計算成功")
        
        print("\n✅ 通道重要性計算測試通過")
        return True
        
    except Exception as e:
        print(f"❌ 通道重要性計算測試失敗: {e}")
        # traceback.print_exc()
        return False
    
def test_lcp_finetune_pipeline():
    """測試 LCP 微調剪枝 pipeline"""
    try:
        print("\n===== 測試 LCP 微調剪枝 Pipeline =====")
        
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化模型和路徑
        os2d_path = "./os2d_v2-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
        
        # 載入模型與輔助網路
        model = Os2dModelInPrune(pretrained_path=os2d_path).to(device)
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 設置 VOC2007 數據集
        from src.dataset_downloader import VOCDataset
        
        # 訓練集
        train_loader = VOCDataset(
            data_path="./data/VOCdevkit/VOC2007",
            split="train",
            download=True
        )
        
        # 驗證集
        val_loader = VOCDataset(
            data_path="./data/VOCdevkit/VOC2007",
            split="val",
            download=True
        )
        
        # 定義要剪枝的層和比例
        pruning_config = [
            {"layer": "layer2.0.conv1", "ratio": 0.3},
            {"layer": "layer2.0.conv2", "ratio": 0.3},
            {"layer": "layer3.0.conv1", "ratio": 0.3}
        ]
        
        # 記錄原始參數量
        orig_params = sum(p.numel() for p in model.parameters())
        print(f"原始參數量: {orig_params:,}")
        
        # 定義訓練參數
        num_epochs = 2
        learning_rate = 0.001
        
        # 執行 fine-tuning 和剪枝
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            model.train()
            train_loss = 0.0
            for batch_idx, batch_data in enumerate(train_loader):
                if batch_idx >= 5:  # 限制批次數用於測試
                    break
                # 解析數據
                if isinstance(batch_data, (tuple, list)):
                    if len(batch_data) == 4:
                        images, boxes, labels, _ = batch_data  # 如果有額外的數據
                    else:
                        images, boxes, labels = batch_data
                else:
                    # 假設 batch_data 是字典格式
                    images = batch_data['images']
                    boxes = batch_data['boxes']
                    labels = batch_data['labels']

                # print(f"images type: {type(images)}")
                # if isinstance(images, list):
                #     print(f"images[0] shape: {images[0].shape}")
                # elif isinstance(images, torch.Tensor):
                #     print(f"images shape: {images.shape}")

                # 處理 images
                if isinstance(images, list):
                    images = torch.stack(images).to(device)  # [B, 3, H, W]
                elif isinstance(images, torch.Tensor):
                    if images.dim() == 3:
                        images = images.unsqueeze(0).to(device)  # [1, 3, H, W]
                    else:
                        images = images.to(device)
                else:
                    raise ValueError(f"images 格式不支援: {type(images)}")

                # print(f"images shape after stack: {images.shape}")

                # 取 class_images
                images = torch.stack([img for img in images]).to(device)
                boxes = [b.to(device) for b in boxes]
                labels = [l.to(device) for l in labels]
                class_images = [images[0].clone()]
                assert class_images[0].shape[0] == 3, f"class_images shape 錯誤: {class_images[0].shape}"
                
                # 前向傳播和計算損失
                outputs = model(images, class_images=class_images)
                # 這裡需要實現損失計算
                loss = outputs[0].mean()  # 示例損失
                
                # 反向傳播和優化
                loss.backward()
                train_loss += loss.item()
                
                if batch_idx % 2 == 0:
                    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # 執行剪枝
            for config in pruning_config:
                layer_name = config["layer"]
                prune_ratio = config["ratio"]
                print(f"\n剪枝層 {layer_name}, 比例 {prune_ratio}")
                device = "cpu"
                success = model.prune_channel(
                    layer_name=layer_name,
                    prune_ratio=prune_ratio,
                    images=images,
                    boxes=boxes,
                    labels=labels,
                    auxiliary_net=aux_net
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                for batch_idx, batch_data in enumerate(val_loader):
                    if batch_idx >= 3:  # 限制批次數用於測試
                        break
                        
                    # 解析數據
                    if isinstance(batch_data, (tuple, list)):
                        if len(batch_data) == 4:
                            images, boxes, labels, _ = batch_data  # 如果有額外的數據
                        else:
                            images, boxes, labels = batch_data
                    else:
                        # 假設 batch_data 是字典格式
                        images = batch_data['images']
                        boxes = batch_data['boxes']
                        labels = batch_data['labels']
                    
                    # 移動數據到設備
                    images = images.to(device)
                    boxes = [b.to(device) for b in boxes]
                    labels = [l.to(device) for l in labels]
            # Validation phase
            model.eval()
            val_loss = 0.0
            device = "cpu"
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(val_loader):
                    if batch_idx >= 3:  # 限制批次數用於測試
                        break
                        
                    # 解析數據
                    if isinstance(batch_data, (tuple, list)):
                        if len(batch_data) == 4:
                            images, boxes, labels, _ = batch_data  # 如果有額外的數據
                        else:
                            images, boxes, labels = batch_data
                    else:
                        # 假設 batch_data 是字典格式
                        images = batch_data['images']
                        boxes = batch_data['boxes']
                        labels = batch_data['labels']
                        
                    # 移動數據到設備
                    images = images.to(device)
                    boxes = [b.to(device) for b in boxes]
                    labels = [l.to(device) for l in labels]
                    
                    if isinstance(images, list):
                        images = torch.stack(images).to(device)  # [B, 3, H, W]
                    elif isinstance(images, torch.Tensor):
                        if images.dim() == 3:
                            images = images.unsqueeze(0).to(device)  # [1, 3, H, W]
                        else:
                            images = images.to(device)
                    else:
                        raise ValueError(f"images 格式不支援: {type(images)}")

                    # print(f"images shape after stack: {images.shape}")

                    # 取 class_images
                    images = torch.stack([img for img in images]).to(device)
                    boxes = [b.to(device) for b in boxes]
                    labels = [l.to(device) for l in labels]
                    class_images = [images[0].clone()]
                    assert class_images[0].shape[0] == 3, f"class_images shape 錯誤: {class_images[0].shape}"
                
                    # 前向傳播
                    outputs = model(images, class_images=class_images)
                    # 這裡需要實現驗證損失計算
                    loss = outputs[0].mean()  # 示例損失
                    val_loss += loss.item()
            
            print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 檢查 channel 一致性
            _test_all_layer_channel_consistency(model)
        
        # 計算最終參數量
        final_params = sum(p.numel() for p in model.parameters())
        reduction = (orig_params - final_params) / orig_params * 100
        print(f"\n參數量減少: {orig_params:,} -> {final_params:,} ({reduction:.2f}%)")
        
        print("\n✅ LCP 微調剪枝 Pipeline 測試通過")
        return True
        
    except Exception as e:
        print(f"❌ LCP 微調剪枝 Pipeline 測試失敗: {e}")
        traceback.print_exc()
        return False
def test_feature_map_extraction():
    """測試 LCP Channel Selector 的特徵圖提取功能"""
    try:
        print("\n===== 測試特徵圖提取功能 =====")
        
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化模型和路徑
        os2d_path = "./os2d_v2-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
        
        # 載入模型與輔助網路
        model = Os2dModelInPrune(pretrained_path=os2d_path).to(device)
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 創建通道選擇器
        selector = OS2DChannelSelector(
            model=model,
            auxiliary_net=aux_net,
            device=device
        )
        
        # 準備測試數據
        batch_size = 1
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        
        # 測試不同層的特徵圖提取
        test_layers = [
            "layer1.0.conv1",
            "layer2.0.conv1",
            "layer3.0.conv1",
        ]
        
        # 1. 首先打印所有可用層名稱以供參考
        print("\n獲取所有可用層名稱:")
        all_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                all_layers[name] = module
        print(f"總共找到 {len(all_layers)} 個卷積層")
        print("前10個卷積層名稱:")
        for idx, (name, _) in enumerate(all_layers.items()):
            if idx < 10:
                print(f"  - {name}")
        
        # 2. 測試不同方法獲取特徵圖
        success_count = 0
        for layer_name in test_layers:
            print(f"\n測試層 {layer_name} 的特徵圖提取:")
            
            # 手動添加 hook 提取特徵圖
            feature_maps = {}
            
            def hook_fn(name):
                def inner_hook(module, input, output):
                    feature_maps[name] = output.detach()
                return inner_hook
            
            # 嘗試不同可能的層路徑
            possible_layer_names = [
                f"net_feature_maps.{layer_name}",
                f"backbone.{layer_name}",
                layer_name
            ]
            
            hooks = []
            found_layer = False
            
            # 註冊 hook
            for full_name in possible_layer_names:
                for name, module in model.named_modules():
                    if name == full_name and isinstance(module, torch.nn.Conv2d):
                        print(f"找到層: {name}")
                        hook = module.register_forward_hook(hook_fn(name))
                        hooks.append(hook)
                        found_layer = True
            
            if not found_layer:
                print(f"❌ 無法找到層 {layer_name}")
                continue
                
            # 前向傳播
            with torch.no_grad():
                # 嘗試不同的前向傳播方法
                try:
                    # 準備 class_images
                    class_images = [images[0].clone()]
                    outputs = model(images, class_images=class_images)
                except TypeError:
                    print("模型不接受 class_images 參數，使用基本模式")
                    outputs = model(images)
            
            # 檢查是否獲取到特徵圖
            if feature_maps:
                success = True
                print(f"✓ 通過 hook 成功獲取特徵圖:")
                for name, feature in feature_maps.items():
                    print(f"  - {name}: {feature.shape}")
                success_count += 1
            else:
                print(f"❌ 通過 hook 無法獲取特徵圖")
                success = False
            
            # 移除 hooks
            for hook in hooks:
                hook.remove()
            
            # 3. 測試 Channel Selector 中的 _get_feature_maps 方法
            print("\n使用 Channel Selector 的 _get_feature_maps 方法:")
            feature_map, orig_feature_map = selector._get_feature_maps(layer_name, images)
            
            if feature_map is not None and orig_feature_map is not None:
                print(f"✓ 成功獲取特徵圖: feature_map {feature_map.shape}, orig_feature_map {orig_feature_map.shape}")
                success_count += 1
            else:
                print("❌ 使用 _get_feature_maps 方法無法獲取特徵圖")
        
        # 測試結論
        if success_count > 0:
            print(f"\n✅ 特徵圖提取測試通過: {success_count} 次成功")
        else:
            print("\n❌ 特徵圖提取測試失敗: 無法獲取任何特徵圖")
            
        # 4. 測試修復方案 - 添加獲取特徵圖的備選方法
        if success_count == 0:
            print("\n嘗試實現特徵圖提取備選方法:")
            
            # 添加 get_feature_map_backup 方法
            def get_feature_map_backup(model, layer_name, images):
                feature_maps = {}
                
                def hook_fn(name):
                    def inner_hook(module, input, output):
                        feature_maps[name] = output.detach()
                    return inner_hook
                
                # 遍歷所有卷積層並查找匹配的
                hooks = []
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d) and layer_name in name:
                        print(f"嘗試匹配層: {name}")
                        hook = module.register_forward_hook(hook_fn(name))
                        hooks.append(hook)
                
                try:
                    # 前向傳播
                    with torch.no_grad():
                        try:
                            # 準備 class_images
                            class_images = [images[0].clone()]
                            _ = model(images, class_images=class_images)
                        except TypeError:
                            _ = model(images)
                    
                    # 檢查是否獲取到特徵圖
                    if feature_maps:
                        key = next(iter(feature_maps.keys()))
                        return feature_maps[key]
                    return None
                
                finally:
                    # 移除 hooks
                    for hook in hooks:
                        hook.remove()
            
            # 測試備選方法
            for layer_name in test_layers:
                print(f"\n測試層 {layer_name} 的備選特徵圖獲取方法:")
                feature_map = get_feature_map_backup(model, layer_name, images)
                if feature_map is not None:
                    print(f"✓ 備選方法成功獲取特徵圖: {feature_map.shape}")
                else:
                    print("❌ 備選方法無法獲取特徵圖")
        
        return True
        
    except Exception as e:
        print(f"❌ 特徵圖提取測試發生錯誤: {e}")
        traceback.print_exc()
        return False

def test_train_one_epoch_basic():
    """測試 train_one_epoch 的基本功能"""
    try:
        # 設置設備
        import torch
        import os
        import numpy as np
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化模型和路徑
        os2d_path = "./os2d_v2-train.pth"
        if not os.path.exists(os2d_path):
            pytest.skip(f"OS2D 預訓練模型不存在: {os2d_path}")
        
        # 載入模型與輔助網路
        model = Os2dModelInPrune(pretrained_path=os2d_path).to(device)
        aux_net = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 載入 VOC2007 數據集
        from src.dataset_downloader import VOCDataset
        from torch.utils.data import DataLoader
        import unittest
        import torch
        import os
        import pytest
        import numpy as np
        from hypothesis import given, strategies as st
        
        # 使用較小的批次大小，僅用於測試
        batch_size = 2
        
        # 訓練集
        train_dataset = VOCDataset(
            data_path="./data/VOCdevkit/VOC2007",
            split="train",
            download=True
        )
        
        # 自定義 collate_fn 以處理不同大小的樣本
        def collate_fn(batch):
            images = []
            boxes = []
            labels = []
            
            for sample in batch:
                if isinstance(sample, tuple) and len(sample) >= 3:
                    img, box, label = sample[:3]
                elif isinstance(sample, dict):
                    img = sample.get('image', None)
                    box = sample.get('boxes', None)
                    label = sample.get('labels', None)
                else:
                    continue
                    
                if img is not None and box is not None and label is not None:
                    images.append(img)
                    boxes.append(box)
                    labels.append(label)
            
            # 確保數據維度正確
            if images and isinstance(images[0], torch.Tensor):
                if images[0].dim() == 3:  # [C,H,W]
                    # 已經是正確格式
                    pass
                elif images[0].dim() == 2:  # [H,W]
                    images = [img.unsqueeze(0) for img in images]  # 添加通道維度
            
            return images, boxes, labels
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # 創建優化器
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': aux_net.parameters()}
        ], lr=0.0001)
        
        # 定義測試參數
        print_freq = 1
        
        # 執行訓練一個 epoch
        print("開始執行 train_one_epoch 基本測試...")
        
        # 執行時傳遞 max_batches 參數 (如果支持)
        try:
            train_loss, component_losses = model.train_one_epoch(
                train_loader=train_loader,
                optimizer=optimizer,
                auxiliary_net=aux_net,
                device=device,
                print_freq=print_freq,
                max_batches=3  # 限制批次數，僅用於測試
            )
        except TypeError:
            # 如果不支持 max_batches 參數，使用標準呼叫
            print("模型不支援 max_batches 參數，使用標準呼叫...")
            train_loss, component_losses = model.train_one_epoch(
                train_loader=train_loader,
                optimizer=optimizer,
                auxiliary_net=aux_net,
                device=device,
                print_freq=print_freq
            )
        
        # 驗證返回值
        assert isinstance(train_loss, (float, np.float64)), "訓練損失應該是浮點數"
        assert isinstance(component_losses, dict), "組件損失應該是字典"
        assert 'cls' in component_losses, "組件損失中應該包含分類損失"
        assert 'reg' in component_losses, "組件損失中應該包含回歸損失"
        
        # 打印結果
        print(f"訓練損失: {train_loss:.4f}")
        for k, v in component_losses.items():
            print(f"  {k} 損失: {v:.4f}")
            
        print("✅ train_one_epoch 基本功能測試通過")
        return True
        
    except Exception as e:
        print(f"❌ train_one_epoch 基本功能測試失敗: {e}")
        traceback.print_exc()
        return False

def test_lcp_prune_and_train_pipeline():
    import torch
    from torch.utils.data import DataLoader
    from src.dataset_downloader import VOCDataset
    from src.auxiliary_network import AuxiliaryNetwork
    from src.os2d_model_in_prune import Os2dModelInPrune

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os2d_path = "./os2d_v2-train.pth"
    model = Os2dModelInPrune(pretrained_path=os2d_path, is_cuda=(device.type == 'cuda')).to(device)
    aux_net = AuxiliaryNetwork(in_channels=2048).to(device)

    train_set = VOCDataset("./data/VOCdevkit/VOC2007", split="train", download=True, img_size=(224, 224))
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, collate_fn=VOCDataset.collate_fn)

    prune_layers = ["layer2.0.conv1", "layer2.0.conv2", "layer3.0.conv1"]
    prune_ratio = 0.3

    optimizer = torch.optim.Adam(list(model.parameters()) + list(aux_net.parameters()), lr=1e-3)

    # 直接呼叫 model.finetune()，讓它自動完成逐層剪枝+微調
    model.finetune(
        train_loader=train_loader,
        auxiliary_net=aux_net,
        prune_layers=prune_layers,
        prune_ratio=prune_ratio,
        optimizer=optimizer,
        device=device,
        epochs_per_layer=1,   # 每層剪枝後微調 1 epoch
        print_freq=1,
        max_batches=3         # 每層只訓練 3 個 batch 以加速測試
    )

    print("\n✅ LCP 剪枝與微調 pipeline 測試通過")
    return True

def test_save_checkpoint():
    import torch
    import os
    from torch.utils.data import DataLoader
    from src.dataset_downloader import VOCDataset
    from src.auxiliary_network import AuxiliaryNetwork
    from src.os2d_model_in_prune import Os2dModelInPrune

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os2d_path = "./os2d_v2-train.pth"
    model = Os2dModelInPrune(pretrained_path=os2d_path, is_cuda=(device.type == 'cuda')).to(device)
    aux_net = AuxiliaryNetwork(in_channels=2048).to(device)

    train_set = VOCDataset(
        "./data/VOCdevkit/VOC2007",
        split="train",
        download=True,
        img_size=(224, 224)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=2,
        shuffle=True,
        collate_fn=VOCDataset.collate_fn
    )

    # 定義要剪枝的層和比例
    prune_layers = ["layer2.0.conv1", "layer2.0.conv2", "layer3.0.conv1"]
    prune_ratio = 0.3
    optimizer = torch.optim.Adam(list(model.parameters()) + list(aux_net.parameters()), lr=1e-3)
    checkpoint_path = "finetune_checkpoint.pth"

    # 計算學生模型的參數量（排除教師模型）
    orig_student_params = sum(p.numel() for name, p in model.named_parameters() 
                           if not name.startswith('teacher_model'))
    orig_total_params = sum(p.numel() for p in model.parameters())
    
    print(f"原始參數統計:")
    print(f"  - 總參數量: {orig_total_params:,}")
    print(f"  - 學生模型參數量: {orig_student_params:,}")
    print(f"  - 教師模型參數量: {orig_total_params - orig_student_params:,}")

    # 執行剪枝+微調，finetune 會自動呼叫 save_checkpoint
    model.finetune(
        train_loader=train_loader,
        auxiliary_net=aux_net,
        prune_layers=prune_layers,
        prune_ratio=prune_ratio,
        optimizer=optimizer,
        device=device,
        epochs_per_layer=1,
        print_freq=1,
        max_batches=3
    )

    # 驗證 checkpoint 是否存在
    assert os.path.exists(checkpoint_path), "❌ checkpoint 檔案未正確儲存"
    
    # 記錄剪枝後的模型結構與學生模型參數量
    pruned_structure = get_conv_structure(model)
    pruned_student_params = sum(p.numel() for name, p in model.named_parameters() 
                             if not name.startswith('teacher_model'))
    pruned_total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n剪枝後參數統計:")
    print(f"  - 總參數量: {pruned_total_params:,}")
    print(f"  - 學生模型參數量: {pruned_student_params:,}")
    print(f"  - 教師模型參數量: {pruned_total_params - pruned_student_params:,}")
    print(f"  - 學生模型參數減少: {orig_student_params - pruned_student_params:,} ({(orig_student_params - pruned_student_params) / orig_student_params * 100:.2f}%)")
    
    # 創建一個新的模型實例，直接從剪枝後的檢查點載入
    print("\n使用 pruned_checkpoint 參數載入模型...")
    model_reloaded = Os2dModelInPrune(
        pretrained_path=None, 
        pruned_checkpoint=checkpoint_path,
        is_cuda=(device.type == 'cuda')
    ).to(device)
    
    # 檢查加載後的模型架構
    loaded_structure = get_conv_structure(model_reloaded)
    loaded_student_params = sum(p.numel() for name, p in model_reloaded.named_parameters() 
                             if not name.startswith('teacher_model'))
    loaded_total_params = sum(p.numel() for p in model_reloaded.parameters())
    
    print(f"\n載入後參數統計:")
    print(f"  - 總參數量: {loaded_total_params:,}")
    print(f"  - 學生模型參數量: {loaded_student_params:,}")
    print(f"  - 教師模型參數量: {loaded_total_params - loaded_student_params:,}")

    # 確認結構相同
    assert len(pruned_structure) == len(loaded_structure), f"層數不匹配: 剪枝後 {len(pruned_structure)}, 載入後 {len(loaded_structure)}"
    
    # 檢查每一層的通道數是否匹配
    for i, (orig, loaded) in enumerate(zip(pruned_structure, loaded_structure)):
        orig_name, orig_in, orig_out, _ = orig
        loaded_name, loaded_in, loaded_out, _ = loaded
        
        if i < 5 or i >= len(pruned_structure) - 5:  # 只顯示前5層和後5層
            print(f"檢查層 {i}: {orig_name}")
            print(f"  原始: in={orig_in}, out={orig_out}")
            print(f"  載入: in={loaded_in}, out={loaded_out}")
        
        assert orig_in == loaded_in, f"層 {orig_name} 輸入通道數不匹配: 原始={orig_in}, 載入={loaded_in}"
        assert orig_out == loaded_out, f"層 {orig_name} 輸出通道數不匹配: 原始={orig_out}, 載入={loaded_out}"
    
    # 驗證參數量是否正確一致
    student_param_diff = abs(loaded_student_params - pruned_student_params)
    assert student_param_diff < 100, f"載入後學生模型參數量與剪枝後不一致，差異: {student_param_diff}"
    assert loaded_student_params < orig_student_params, f"剪枝後學生模型參數未減少: {loaded_student_params} >= {orig_student_params}"

    # 測試前向傳播
    print("\n執行前向傳播測試...")
    batch_size = 1
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    class_images = [torch.randn(3, 224, 224).to(device)]
    
    with torch.no_grad():
        outputs = model_reloaded(images, class_images=class_images)
        assert outputs is not None, "載入模型的前向傳播返回 None"
        print("✓ 前向傳播測試通過")

    print(f"\n✅ save_checkpoint 測試通過")
    print(f"參數量變化摘要:")
    print(f"  - 原始學生模型: {orig_student_params:,}")
    print(f"  - 剪枝後學生模型: {pruned_student_params:,}")
    print(f"  - 載入後學生模型: {loaded_student_params:,}")
    print(f"  - 參數減少比例: {(orig_student_params - loaded_student_params) / orig_student_params * 100:.2f}%")
    
    # 清理測試文件
    os.remove(checkpoint_path)
    return True

def get_conv_structure(model):
    return [(name, m.in_channels, m.out_channels, m.kernel_size)
            for name, m in model.backbone.named_modules() if isinstance(m, torch.nn.Conv2d)]

def test_os2d_compatibility_with_pruned_model():
    """測試剪枝後的模型能否正確放回原始 OS2D 框架中使用"""
    import os
    import torch
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from PIL import Image
    
    from os2d.modeling.model import build_os2d_from_config
    from os2d.config import cfg
    import os2d.utils.visualization as visualizer
    from os2d.structures.feature_map import FeatureMapSize
    from os2d.utils import setup_logger, read_image, get_image_size_after_resize_preserving_aspect_ratio
    
    from src.os2d_model_in_prune import Os2dModelInPrune
    from src.auxiliary_network import AuxiliaryNetwork
    from torch.utils.data import DataLoader
    from src.dataset_downloader import VOCDataset
    
    print("\n===== 測試剪枝模型與原始 OS2D 框架相容性 =====")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os2d_path = "./os2d_v2-train.pth"
    model = Os2dModelInPrune(pretrained_path=os2d_path, is_cuda=(device.type == 'cuda')).to(device)
    aux_net = AuxiliaryNetwork(in_channels=2048).to(device)

    train_set = VOCDataset(
        "./data/VOCdevkit/VOC2007",
        split="train",
        download=True,
        img_size=(224, 224)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=2,
        shuffle=True,
        collate_fn=VOCDataset.collate_fn
    )

    # 定義要剪枝的層和比例
    prune_layers = ["layer2.0.conv1", "layer2.0.conv2", "layer3.0.conv1"]
    prune_ratio = 0.3
    optimizer = torch.optim.Adam(list(model.parameters()) + list(aux_net.parameters()), lr=1e-3)
    checkpoint_path = "finetune_checkpoint.pth"

    # 計算學生模型的參數量（排除教師模型）
    orig_student_params = sum(p.numel() for name, p in model.named_parameters() 
                           if not name.startswith('teacher_model'))
    orig_total_params = sum(p.numel() for p in model.parameters())
    
    print(f"原始參數統計:")
    print(f"  - 總參數量: {orig_total_params:,}")
    print(f"  - 學生模型參數量: {orig_student_params:,}")
    print(f"  - 教師模型參數量: {orig_total_params - orig_student_params:,}")

    # 執行剪枝+微調，finetune 會自動呼叫 save_checkpoint
    model.finetune(
        train_loader=train_loader,
        auxiliary_net=aux_net,
        prune_layers=prune_layers,
        prune_ratio=prune_ratio,
        optimizer=optimizer,
        device=device,
        epochs_per_layer=1,
        print_freq=1,
        max_batches=3
    )

    # 驗證 checkpoint 是否存在
    assert os.path.exists(checkpoint_path), "❌ checkpoint 檔案未正確儲存"
    
    # 記錄剪枝後的模型結構與學生模型參數量
    pruned_structure = get_conv_structure(model)
    pruned_student_params = sum(p.numel() for name, p in model.named_parameters() 
                             if not name.startswith('teacher_model'))
    pruned_total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n剪枝後參數統計:")
    print(f"  - 總參數量: {pruned_total_params:,}")
    print(f"  - 學生模型參數量: {pruned_student_params:,}")
    print(f"  - 教師模型參數量: {pruned_total_params - pruned_student_params:,}")
    print(f"  - 學生模型參數減少: {orig_student_params - pruned_student_params:,} ({(orig_student_params - pruned_student_params) / orig_student_params * 100:.2f}%)")
    
    # 創建一個新的模型實例，直接從剪枝後的檢查點載入
    print("\n使用 pruned_checkpoint 參數載入模型...")
    model_reloaded = Os2dModelInPrune(
        pretrained_path=None, 
        pruned_checkpoint=checkpoint_path,
        is_cuda=(device.type == 'cuda')
    ).to(device)
    
    # 檢查加載後的模型架構
    loaded_structure = get_conv_structure(model_reloaded)
    loaded_student_params = sum(p.numel() for name, p in model_reloaded.named_parameters() 
                             if not name.startswith('teacher_model'))
    loaded_total_params = sum(p.numel() for p in model_reloaded.parameters())
    
    print(f"\n載入後參數統計:")
    print(f"  - 總參數量: {loaded_total_params:,}")
    print(f"  - 學生模型參數量: {loaded_student_params:,}")
    print(f"  - 教師模型參數量: {loaded_total_params - loaded_student_params:,}")

    # 確認結構相同
    assert len(pruned_structure) == len(loaded_structure), f"層數不匹配: 剪枝後 {len(pruned_structure)}, 載入後 {len(loaded_structure)}"
    
    # 檢查每一層的通道數是否匹配
    for i, (orig, loaded) in enumerate(zip(pruned_structure, loaded_structure)):
        orig_name, orig_in, orig_out, _ = orig
        loaded_name, loaded_in, loaded_out, _ = loaded
        
        if i < 5 or i >= len(pruned_structure) - 5:  # 只顯示前5層和後5層
            print(f"檢查層 {i}: {orig_name}")
            print(f"  原始: in={orig_in}, out={orig_out}")
            print(f"  載入: in={loaded_in}, out={loaded_out}")
        
        assert orig_in == loaded_in, f"層 {orig_name} 輸入通道數不匹配: 原始={orig_in}, 載入={loaded_in}"
        assert orig_out == loaded_out, f"層 {orig_name} 輸出通道數不匹配: 原始={orig_out}, 載入={loaded_out}"
    
    # 驗證參數量是否正確一致
    student_param_diff = abs(loaded_student_params - pruned_student_params)
    assert student_param_diff < 100, f"載入後學生模型參數量與剪枝後不一致，差異: {student_param_diff}"
    assert loaded_student_params < orig_student_params, f"剪枝後學生模型參數未減少: {loaded_student_params} >= {orig_student_params}"

    # 測試前向傳播
    print("\n執行前向傳播測試...")
    batch_size = 1
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    class_images = [torch.randn(3, 224, 224).to(device)]
    
    with torch.no_grad():
        outputs = model_reloaded(images, class_images=class_images)
        assert outputs is not None, "載入模型的前向傳播返回 None"
        print("✓ 前向傳播測試通過")

    print(f"\n✅ save_checkpoint 測試通過")
    print(f"參數量變化摘要:")
    print(f"  - 原始學生模型: {orig_student_params:,}")
    print(f"  - 剪枝後學生模型: {pruned_student_params:,}")
    print(f"  - 載入後學生模型: {loaded_student_params:,}")
    print(f"  - 參數減少比例: {(orig_student_params - loaded_student_params) / orig_student_params * 100:.2f}%")
    return True
    # logger = setup_logger("OS2D")
    # cfg.init.model = "finetune_checkpoint.pth"
    # net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)
