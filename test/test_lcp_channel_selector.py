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
