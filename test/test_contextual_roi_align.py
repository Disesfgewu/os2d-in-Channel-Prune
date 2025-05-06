# test_contextual_roi_align.py
import torch
import pytest
from src.contextual_roi_align import ContextualRoIAlign

def test_forward():
    """測試 ContextualRoIAlign 前向傳播"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化 ContextualRoIAlign
    roi_align = ContextualRoIAlign(output_size=7)
    
    # 創建測試輸入
    batch_size = 2
    channels = 64
    feature_map = torch.randn(batch_size, channels, 28, 28).to(device)
    boxes = [torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32).to(device) for _ in range(batch_size)]
    
    # 測試前向傳播 (無 gt_boxes)
    features = roi_align(feature_map, boxes)
    
    # 驗證輸出形狀
    assert features.shape[0] == batch_size * 2  # 每個圖像2個框
    assert features.shape[1] == channels  # 通道數保持不變
    assert features.shape[2] == 7 and features.shape[3] == 7  # 輸出大小為7x7
    
    # 測試前向傳播 (有 gt_boxes)
    gt_boxes = [torch.tensor([[15, 15, 55, 55], [25, 25, 65, 65]], dtype=torch.float32).to(device) for _ in range(batch_size)]
    features_with_context = roi_align(feature_map, boxes, gt_boxes)
    
    # 驗證輸出形狀
    assert features_with_context.shape[0] == batch_size * 2
    assert features_with_context.shape[1] == channels
    assert features_with_context.shape[2] == 7 and features_with_context.shape[3] == 7
    
    print(f"✅ 特徵形狀: {features.shape}")
    print(f"✅ 上下文特徵形狀: {features_with_context.shape}")
    return True

def test_empty_boxes():
    """測試空框情況"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化 ContextualRoIAlign
    roi_align = ContextualRoIAlign(output_size=7)
    
    # 創建測試輸入
    batch_size = 2
    channels = 64
    feature_map = torch.randn(batch_size, channels, 28, 28).to(device)
    
    # 空框
    empty_boxes = [torch.zeros((0, 4), dtype=torch.float32).to(device) for _ in range(batch_size)]
    
    # 測試前向傳播 (無 gt_boxes)
    features = roi_align(feature_map, empty_boxes)
    
    # 驗證輸出形狀
    assert features.shape[0] == 0
    assert features.shape[1] == channels
    assert features.shape[2] == 7 and features.shape[3] == 7
    
    # 測試前向傳播 (有 gt_boxes)
    gt_boxes = [torch.tensor([[15, 15, 55, 55]], dtype=torch.float32).to(device) for _ in range(batch_size)]
    features_with_context = roi_align(feature_map, empty_boxes, gt_boxes)
    
    # 驗證輸出形狀
    assert features_with_context.shape[0] == 0
    assert features_with_context.shape[1] == channels
    assert features_with_context.shape[2] == 7 and features_with_context.shape[3] == 7
    
    print(f"✅ 空框特徵形狀: {features.shape}")
    return True


def test_contextual_roi_align_all_cases():
    import torch
    from src.contextual_roi_align import ContextualRoIAlign
    from os2d.structures.bounding_box import BoxList, FeatureMapSize

    device = torch.device("cpu")
    roi_align = ContextualRoIAlign(output_size=3)

    # 1. 普通 tensor box
    feature_map = torch.randn(2, 8, 32, 32, device=device)
    boxes = [torch.tensor([[1,2,10,12],[5,5,20,20]], dtype=torch.float32),
             torch.tensor([[3,4,15,18]], dtype=torch.float32)]
    out = roi_align(feature_map, boxes)
    assert isinstance(out, torch.Tensor)
    assert out.shape[1:] == (8, 3, 3)

    # 2. BoxList (正確屬性 bbox_xyxy)
    img_size = FeatureMapSize(w=32, h=32)
    boxlist1 = BoxList(torch.tensor([[1,2,10,12],[5,5,20,20]]), img_size)
    boxlist2 = BoxList(torch.tensor([[3,4,15,18]]), img_size)
    out = roi_align(feature_map, [boxlist1, boxlist2])
    assert isinstance(out, torch.Tensor)
    assert out.shape[1:] == (8, 3, 3)

    # 3. 上下文增強模式 (gt_boxes 為 tensor)
    gt_boxes = [torch.tensor([[0,0,8,8]], dtype=torch.float32),
                torch.tensor([[10,10,30,30]], dtype=torch.float32)]
    out = roi_align(feature_map, boxes, gt_boxes=gt_boxes)
    assert isinstance(out, torch.Tensor)
    assert out.shape[1:] == (8, 3, 3)

    # 4. 上下文增強模式 (gt_boxes 為 BoxList)
    gt_boxlist1 = BoxList(torch.tensor([[0,0,8,8]]), img_size)
    gt_boxlist2 = BoxList(torch.tensor([[10,10,30,30]]), img_size)
    out = roi_align(feature_map, [boxlist1, boxlist2], gt_boxes=[gt_boxlist1, gt_boxlist2])
    assert isinstance(out, torch.Tensor)
    assert out.shape[1:] == (8, 3, 3)

    # 5. 空 box/tensor
    empty_box = torch.empty((0,4))
    out = roi_align(feature_map, [empty_box, empty_box])
    assert out.shape[0] == 0

    # 6. 空 BoxList
    empty_boxlist = BoxList.create_empty(img_size)
    out = roi_align(feature_map, [empty_boxlist, empty_boxlist])
    assert out.shape[0] == 0

    # 7. 混合型別
    out = roi_align(feature_map, [boxlist1, torch.tensor([[3,4,15,18]])])
    assert isinstance(out, torch.Tensor)

    print("✅ ContextualRoIAlign 測試全部通過，涵蓋 BoxList/tensor/空box/上下文/混合型別等所有情境。")
    return True