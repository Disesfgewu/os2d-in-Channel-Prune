# test_auxiliary_network.py
import torch
import pytest
import traceback
from src.auxiliary_network import AuxiliaryNetwork
from src.contextual_roi_align import ContextualRoIAlign

def test_initialization():
    """測試輔助網路初始化"""
    try:
        # 測試不同輸入通道數
        for in_channels in [64, 256, 512, 2048]:
            print(f"測試初始化 in_channels={in_channels}")
            aux_net = AuxiliaryNetwork(in_channels=in_channels)
            
            # 檢查關鍵屬性
            assert hasattr(aux_net, 'conv'), "缺少 conv 層"
            assert hasattr(aux_net, 'contextual_roi_align'), "缺少 contextual_roi_align 層"
            assert hasattr(aux_net, 'cls_head'), "缺少 cls_head 層"
            assert hasattr(aux_net, 'reg_head'), "缺少 reg_head 層"
            
            # 檢查通道數
            assert aux_net.conv.in_channels == in_channels, f"輸入通道數不匹配: 預期 {in_channels}, 實際 {aux_net.conv.in_channels}"
            assert aux_net.conv.out_channels == 256, f"輸出通道數不匹配: 預期 256, 實際 {aux_net.conv.out_channels}"
            
            # 檢查類型
            assert isinstance(aux_net.contextual_roi_align, ContextualRoIAlign), "contextual_roi_align 類型錯誤"
            assert aux_net.cls_head[-1].out_features == 20, f"分類頭輸出特徵數不匹配: 預期 20, 實際 {aux_net.cls_head[-1].out_features}"
            assert aux_net.reg_head[-1].out_features == 4, f"回歸頭輸出特徵數不匹配: 預期 4, 實際 {aux_net.reg_head[-1].out_features}"
        
        print("✅ 初始化測試通過")
        return True
    except Exception as e:
        print(f"❌ 初始化測試失敗: {e}")
        traceback.print_exc()
        return False

def test_forward_pass():
    """測試前向傳播"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用設備: {device}")
        
        # 初始化輔助網路
        in_channels = 64
        print(f"初始化輔助網路 in_channels={in_channels}")
        aux_net = AuxiliaryNetwork(in_channels=in_channels).to(device)
        
        # 創建測試輸入
        batch_size = 2
        feature_map = torch.randn(batch_size, in_channels, 28, 28).to(device)
        boxes = [torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32).to(device) for _ in range(batch_size)]
        
        print(f"特徵圖形狀: {feature_map.shape}")
        print(f"框數量: {[len(b) for b in boxes]}")
        
        # 測試前向傳播
        print("執行前向傳播...")
        cls_scores, bbox_preds = aux_net(feature_map, boxes)
        
        # 驗證輸出形狀
        expected_num_boxes = sum(len(b) for b in boxes)
        print(f"分類分數形狀: {cls_scores.shape}")
        print(f"邊界框預測形狀: {bbox_preds.shape}")
        
        assert cls_scores.shape[0] == expected_num_boxes, f"分類分數批次大小不匹配: 預期 {expected_num_boxes}, 實際 {cls_scores.shape[0]}"
        assert cls_scores.shape[1] == 20, f"分類分數類別數不匹配: 預期 20, 實際 {cls_scores.shape[1]}"
        assert bbox_preds.shape[0] == expected_num_boxes, f"邊界框預測批次大小不匹配: 預期 {expected_num_boxes}, 實際 {bbox_preds.shape[0]}"
        assert bbox_preds.shape[1] == 4, f"邊界框預測坐標數不匹配: 預期 4, 實際 {bbox_preds.shape[1]}"
        
        # 測試空輸入
        print("測試空框輸入...")
        empty_boxes = [torch.zeros((0, 4), dtype=torch.float32).to(device) for _ in range(batch_size)]
        cls_scores, bbox_preds = aux_net(feature_map, empty_boxes)
        
        print(f"空框分類分數形狀: {cls_scores.shape}")
        print(f"空框邊界框預測形狀: {bbox_preds.shape}")
        
        assert cls_scores.shape[0] == 0, f"空框分類分數批次大小不匹配: 預期 0, 實際 {cls_scores.shape[0]}"
        assert bbox_preds.shape[0] == 0, f"空框邊界框預測批次大小不匹配: 預期 0, 實際 {bbox_preds.shape[0]}"
        
        print("✅ 前向傳播測試通過")
        return True
    except Exception as e:
        print(f"❌ 前向傳播測試失敗: {e}")
        traceback.print_exc()
        return False

def test_update_input_channels():
    """測試動態更新輸入通道功能"""
    try:
        # 初始化輔助網路
        initial_channels = 64
        print(f"初始化輔助網路 in_channels={initial_channels}")
        aux_net = AuxiliaryNetwork(in_channels=initial_channels)
        
        # 檢查初始通道數
        assert aux_net.get_current_channels() == initial_channels, f"初始通道數不匹配: 預期 {initial_channels}, 實際 {aux_net.get_current_channels()}"
        
        # 測試通道減少
        new_channels = 32
        print(f"測試通道減少: {initial_channels} -> {new_channels}")
        aux_net.update_input_channels(new_channels)
        assert aux_net.get_current_channels() == new_channels, f"減少後通道數不匹配: 預期 {new_channels}, 實際 {aux_net.get_current_channels()}"
        
        # 測試通道增加
        new_channels = 128
        print(f"測試通道增加: {aux_net.get_current_channels()} -> {new_channels}")
        aux_net.update_input_channels(new_channels)
        assert aux_net.get_current_channels() == new_channels, f"增加後通道數不匹配: 預期 {new_channels}, 實際 {aux_net.get_current_channels()}"
        
        # 測試無變化
        print(f"測試無變化: {new_channels} -> {new_channels}")
        aux_net.update_input_channels(new_channels)
        assert aux_net.get_current_channels() == new_channels, f"無變化後通道數不匹配: 預期 {new_channels}, 實際 {aux_net.get_current_channels()}"
        
        # 測試無效輸入
        print("測試無效輸入...")
        try:
            aux_net.update_input_channels(-1)
            assert False, "應該拋出 ValueError 但沒有"
        except ValueError as e:
            print(f"✓ 正確拋出 ValueError: {e}")
        
        try:
            aux_net.update_input_channels("invalid")
            assert False, "應該拋出 TypeError 或 ValueError 但沒有"
        except (TypeError, ValueError) as e:
            print(f"✓ 正確拋出異常: {e}")
        
        print("✅ 更新通道測試通過")
        return True
    except Exception as e:
        print(f"❌ 更新通道測試失敗: {e}")
        traceback.print_exc()
        return False
