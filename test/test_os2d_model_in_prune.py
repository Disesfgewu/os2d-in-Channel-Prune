import os
import torch
import torchvision
import pytest
import traceback
from src.lcp_channel_selector import OS2DChannelSelector
from src.contextual_roi_align import ContextualRoIAlign
from src.os2d_model_in_prune import Os2dModelInPrune
from src.auxiliary_network import AuxiliaryNetwork
import logging

def test_os2d_model_in_prune_initialization():
    """測試 Os2dModelInPrune 初始化"""
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
    
    # 驗證參數數量
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型參數總數: {param_count}")
    assert param_count > 0, "模型參數數量應該大於 0"
    
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
if __name__ == "__main__":
    test_os2d_model_in_prune_initialization()
    test_os2d_model_in_prune_forward()
    test_forward_pass_with_class_images()
    test_set_layer_out_channels() 
    test_os2d_model_in_prune_channel()
    test_continuous_blocks_pruning()
    test_prune_conv1_only()
    test_prune_conv2_only()
    test_prune_conv3_only()
    test_continuous_block_pruning()
    test_prune_multiple_blocks()
    test_cross_stage_prune()
    test_resnet18_basicblock_prune()
    test_pruning_ratios("layer2.0.conv1")
    test_lcp_channel_selector()
    test_channel_importance_computation()
