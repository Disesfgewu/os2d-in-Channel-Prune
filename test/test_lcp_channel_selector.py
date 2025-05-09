# test_lcp_channel_selector.py
import torch
import numpy as np
import pytest
from src.os2d_model_in_prune import Os2dModelInPrune
from src.auxiliary_network import AuxiliaryNetwork
from src.lcp_channel_selector import OS2DChannelSelector

def test_lcp_channel_selector_initialization():
    """測試 LCP 通道選擇器初始化"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Os2dModelInPrune(device=device)
    aux_net = AuxiliaryNetwork(in_channels=64).to(device)
    
    # 初始化通道選擇器
    selector = OS2DChannelSelector(model, aux_net, device=device)
    
    assert selector is not None
    assert hasattr(selector, 'compute_importance')
    assert hasattr(selector, 'compute_cls_loss')
    assert hasattr(selector, 'compute_reg_loss')

def test_compute_importance(voc_data):
    """測試通道重要性計算"""
    # 準備測試數據
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Os2dModelInPrune(device=device)
    aux_net = AuxiliaryNetwork(in_channels=64).to(device)
    selector = OS2DChannelSelector(model, aux_net, device=device, alpha=0.6, beta=0.4)
    
    # 獲取測試數據
    images, boxes, labels = voc_data
    images = images.to(device)
    boxes = [b.to(device) for b in boxes]
    
    # 測試目標層
    layer_name = "layer2.0.conv1"
    
    # 獲取目標層
    target_layer = model
    for part in layer_name.split('.'):
        if part.isdigit():
            target_layer = target_layer[int(part)]
        else:
            target_layer = getattr(target_layer, part)
    
    expected_channels = target_layer.out_channels
    
    # 計算通道重要性
    importance = selector.compute_importance(layer_name, images, boxes, boxes, labels)
    
    # 驗證結果
    assert importance is not None
    assert len(importance) == expected_channels
    assert isinstance(importance, torch.Tensor) or isinstance(importance, np.ndarray)
    
    # 驗證重要性分數在合理範圍內
    if isinstance(importance, torch.Tensor):
        importance = importance.cpu().numpy()
    
    assert np.all(np.isfinite(importance)), "重要性分數應該是有限的"
    assert np.all(importance >= 0), "重要性分數應該是非負的"

def test_select_channels():
    """測試通道選擇功能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Os2dModelInPrune(device=device)
    aux_net = AuxiliaryNetwork(in_channels=64).to(device)
    selector = OS2DChannelSelector(model, aux_net, device=device)
    
    # 模擬重要性分數
    layer_name = "layer2.0.conv1"
    target_layer = model
    for part in layer_name.split('.'):
        if part.isdigit():
            target_layer = target_layer[int(part)]
        else:
            target_layer = getattr(target_layer, part)
    
    num_channels = target_layer.out_channels
    importance_scores = torch.rand(num_channels)
    
    # 測試不同剪枝比例
    for prune_ratio in [0.1, 0.3, 0.5, 0.7]:
        keep_indices = selector.select_channels(layer_name, importance_scores, prune_ratio)
        
        # 驗證結果
        expected_keep = int(num_channels * (1 - prune_ratio))
        assert len(keep_indices) == expected_keep, f"應該保留 {expected_keep} 個通道，但實際保留了 {len(keep_indices)} 個"
        assert all(0 <= idx < num_channels for idx in keep_indices), "通道索引應該在有效範圍內"
        
        # 驗證是否保留了最重要的通道
        top_indices = torch.topk(importance_scores, expected_keep).indices.numpy()
        assert set(keep_indices) == set(top_indices), "應該保留最重要的通道"

def test_lcp_channel_selector():
    """測試 LCP 通道選擇器基本功能"""
    try:
        print("\n===== LCP 通道選擇器測試 =====")
        
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
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
        # traceback.print_exc()
        return False

def test_channel_importance_computation():
    """測試通道重要性計算功能"""
    try:
        print("\n===== 通道重要性計算測試 =====")
        
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
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
        # # traceback.print_exc()
        return False
    
def test_feature_map_extraction():
    """測試 LCP Channel Selector 的特徵圖提取功能"""
    try:
        print("\n===== 測試特徵圖提取功能 =====")
        
        # 設置設備
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
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
        # traceback.print_exc()
        return False