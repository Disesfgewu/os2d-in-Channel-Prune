import os
import torch
import pytest
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from os2d.data.dataloader import build_eval_dataloaders_from_cfg, build_train_dataloader_from_config
from os2d.config import cfg
from os2d.modeling.box_coder import Os2dBoxCoder
import os
import math
import numbers
import time
import logging
from collections import OrderedDict
from functools import lru_cache
from os2d.modeling.model import Os2dModel
from os2d.utils import mkdir, setup_logger
from os2d.utils import get_data_path
import torch
import torch.nn as nn
import torch.nn.functional as F

from os2d.modeling.feature_extractor import build_feature_extractor

from os2d.engine.objective import Os2dObjective
from os2d.utils import count_model_parameters, masked_select_or_fill_constant
from os2d.structures.feature_map import FeatureMapSize
from src.auxiliary_network import AuxiliaryNetwork, ContextualRoIAlign

def get_data_path():
    """獲取 Grozi 數據集路徑"""
    return os.environ.get("DATA_PATH", "./data")

def setup_grozi_dataset():
    """設置 Grozi 數據集"""
    data_path = get_data_path()
    grozi_path = os.path.join(data_path, "grozi")
    
    # 檢查數據集是否已下載
    if not os.path.exists(grozi_path):
        # 下載 Grozi 數據集
        os.makedirs(data_path, exist_ok=True)
        os.system(f"cd {data_path} && ../os2d/utils/wget_gdrive.sh grozi.zip 1Fx9lvmjthe3aOqjvKc6MJpMuLF22I1Hp")
        os.system(f"unzip {data_path}/grozi.zip -d {data_path}")
    
    return grozi_path

@pytest.fixture
def device():
    """設置設備"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def grozi_dataset():
    """設置 Grozi 數據集"""
    # 確保 Grozi 數據集已下載
    setup_grozi_dataset()
    
    # 設置配置
    cfg.merge_from_file("experiments/config_training.yml")
    
    # 修改配置以使用 Grozi 數據集
    cfg.train.dataset_name = "grozi-train"
    cfg.eval.dataset_names = ["grozi-val-new-cl"]
    cfg.eval.dataset_scales = [1280.0]
    
    # 添加缺失的配置參數
    cfg.train.positive_iou_threshold = 0.7
    cfg.train.negative_iou_threshold = 0.3
    cfg.train.remap_classification_targets_iou_pos = 0.7
    cfg.train.remap_classification_targets_iou_neg = 0.3
    logger = logging.getLogger("OS2D")
    img_normalization = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    net = Os2dModel(logger=logger,
                    is_cuda=cfg.is_cuda,
                    backbone_arch=cfg.model.backbone_arch,
                    merge_branch_parameters=cfg.model.merge_branch_parameters,
                    use_group_norm=cfg.model.use_group_norm,
                    use_inverse_geom_model=cfg.model.use_inverse_geom_model,
                    simplify_affine=cfg.model.use_simplified_affine_model,
                    img_normalization=img_normalization)
    box_coder = Os2dBoxCoder(positive_iou_threshold=cfg.train.objective.positive_iou_threshold,
                             negative_iou_threshold=cfg.train.objective.negative_iou_threshold,
                             remap_classification_targets_iou_pos=cfg.train.objective.remap_classification_targets_iou_pos,
                             remap_classification_targets_iou_neg=cfg.train.objective.remap_classification_targets_iou_neg,
                             output_box_grid_generator=net.os2d_head_creator.box_grid_generator_image_level,
                             function_get_feature_map_size=net.get_feature_map_size,
                             do_nms_across_classes=cfg.eval.nms_across_classes)

    # 圖像標準化參數
    # 加載數據集
    data_path = get_data_path()
    dataloader_train, datasets_train_for_eval = build_train_dataloader_from_config(
        cfg, box_coder, img_normalization, data_path=data_path
    )
    
    dataloaders_eval = build_eval_dataloaders_from_cfg(
        cfg, box_coder, img_normalization,
        datasets_for_eval=datasets_train_for_eval,
        data_path=data_path
    )
    
    return {
        "dataloader_train": dataloader_train,
        "dataloaders_eval": dataloaders_eval
    }

@pytest.fixture
def sample_batch(grozi_dataset):
    """獲取樣本批次"""
    dataloader = grozi_dataset["dataloader_train"]
    # Access the PyTorch DataLoader inside the DataloaderOneShotDetection object
    # Get the first batch directly from the dataloader
    batch = dataloader.get_batch(0)  # Get the first batch (index 0)
    return batch

class TestAuxiliaryNetwork:
    def test_initialization(self, device):
        """測試初始化"""
        # 創建不同輸入通道的網絡
        aux_net = AuxiliaryNetwork(in_channels=64, hidden_channels=128, num_classes=1063).to(device)
        
        # 檢查網絡結構
        assert isinstance(aux_net.conv, nn.Conv2d)
        assert isinstance(aux_net.bn, nn.BatchNorm2d)
        assert isinstance(aux_net.relu, nn.ReLU)
        assert isinstance(aux_net.cls_head, nn.Sequential)
        assert isinstance(aux_net.reg_head, nn.Sequential)
        assert isinstance(aux_net.contextual_roi_align, ContextualRoIAlign)
        
        # 檢查輸入通道數
        assert aux_net.conv.in_channels == 64
        assert aux_net.conv.out_channels == 128
        
        # 檢查類別數
        assert aux_net.cls_head[-1].out_channels == 1063
    
    def test_update_input_channels(self, device):
        """測試更新輸入通道數"""
        # 創建不同輸入通道的網絡
        aux_net_64 = AuxiliaryNetwork(in_channels=64).to(device)
        aux_net_128 = AuxiliaryNetwork(in_channels=128).to(device)
        aux_net_2048 = AuxiliaryNetwork(in_channels=2048).to(device)
        
        # 檢查輸入通道數是否正確設置
        assert aux_net_64.conv.in_channels == 64
        assert aux_net_128.conv.in_channels == 128
        assert aux_net_2048.conv.in_channels == 2048
    
    def test_forward_pass(self, device):
        """測試前向傳播"""
        # 創建隨機特徵圖
        batch_size = 2
        channels = 64
        height, width = 32, 32
        features = torch.randn(batch_size, channels, height, width).to(device)
        
        # 創建隨機框
        num_boxes = 5
        boxes = torch.rand(batch_size, num_boxes, 4).to(device)
        boxes[:, :, 2:] += boxes[:, :, :2]  # 確保 x2 > x1, y2 > y1
        boxes = [boxes[i] for i in range(batch_size)]  # 轉換為列表格式
        
        # 創建隨機真實框
        gt_boxes = torch.rand(batch_size, num_boxes, 4).to(device)
        gt_boxes[:, :, 2:] += gt_boxes[:, :, :2]  # 確保 x2 > x1, y2 > y1
        gt_boxes = [gt_boxes[i] for i in range(batch_size)]  # 轉換為列表格式
        
        # 初始化網絡
        auxiliary_network = AuxiliaryNetwork(in_channels=64).to(device)
        
        # 測試有框的情況
        cls_scores, bbox_preds = auxiliary_network(features, boxes, gt_boxes)
        
        # 檢查輸出形狀 - 現在是列表
        assert len(cls_scores) == batch_size
        assert len(bbox_preds) == batch_size
        
        # 檢查每個批次的輸出
        for i in range(batch_size):
            assert cls_scores[i].shape[0] == num_boxes
            assert bbox_preds[i].shape[0] == num_boxes
        
        # 測試無框的情況
        cls_scores, bbox_preds = auxiliary_network(features)
        
        # 檢查輸出形狀
        assert cls_scores.shape[0] == batch_size
        assert cls_scores.shape[1] == 20  # 類別數
        assert cls_scores.shape[2] == height
        assert cls_scores.shape[3] == width
        assert bbox_preds.shape[0] == batch_size
        assert bbox_preds.shape[1] == 4  # 邊界框座標
        assert bbox_preds.shape[2] == height
        assert bbox_preds.shape[3] == width

    
    def test_with_grozi_data(self, sample_batch, device):
        """使用 Grozi 數據測試"""
        # 解析批次數據
        # sample_batch is a tuple, extract the images and boxes
        images = sample_batch[0].to(device)  # Assuming first element contains images
        boxes = sample_batch[1] if len(sample_batch) > 1 else []
        
        # 創建隨機特徵圖 (模擬骨幹網絡的輸出)
        batch_size = images.shape[0]
        channels = 64
        height, width = 32, 32
        features = torch.randn(batch_size, channels, height, width).to(device)
        
        # 初始化網絡 - 使用 Grozi 數據集的類別數 (1063)
        auxiliary_network = AuxiliaryNetwork(in_channels=64, num_classes=1063).to(device)
        
        # 前向傳播
        cls_scores, bbox_preds = auxiliary_network(features, boxes)
        
        # 檢查輸出
        assert cls_scores is not None
        assert bbox_preds is not None
        
        # Grozi 特定檢查 - 檢查類別數量是否符合 Grozi 數據集
        if hasattr(cls_scores, "shape") and len(cls_scores.shape) > 1:
            assert cls_scores.shape[-1] == 1063

class TestContextualRoIAlign:
    def test_forward(self, device):
        """測試前向傳播"""
        # 創建隨機特徵圖
        batch_size = 2
        channels = 64
        height, width = 32, 32
        features = torch.randn(batch_size, channels, height, width).to(device)
        
        # 創建隨機框
        num_boxes = 5
        boxes = torch.rand(batch_size, num_boxes, 4).to(device)
        boxes[:, :, 2:] += boxes[:, :, :2]  # 確保 x2 > x1, y2 > y1
        
        # 創建隨機真實框
        gt_boxes = torch.rand(batch_size, num_boxes, 4).to(device)
        gt_boxes[:, :, 2:] += gt_boxes[:, :, :2]  # 確保 x2 > x1, y2 > y1
        
        # 初始化 ContextualRoIAlign
        contextual_roi_align = ContextualRoIAlign(output_size=7).to(device)
        
        # 前向傳播
        roi_features = contextual_roi_align(features, boxes, gt_boxes)
        
        # 檢查輸出形狀
        assert roi_features is not None
    
    def test_empty_boxes(self, device):
        """測試空框情況"""
        # 創建隨機特徵圖
        batch_size = 2
        channels = 64
        height, width = 32, 32
        features = torch.randn(batch_size, channels, height, width).to(device)
        
        # 創建空框
        empty_boxes = torch.zeros(batch_size, 0, 4).to(device)
        
        # 初始化 ContextualRoIAlign
        contextual_roi_align = ContextualRoIAlign(output_size=7).to(device)
        
        # 前向傳播
        roi_features = contextual_roi_align(features, empty_boxes)
        
        # 檢查輸出形狀
        assert roi_features is not None
        # Since roi_features is a list, check its length or elements instead
        assert len(roi_features) == batch_size  # Check if we have one element per batch
        for batch_features in roi_features:
            assert batch_features.shape[0] == 0  # 沒有框
    
    def test_with_grozi_data(self, sample_batch, device):
        """使用 Grozi 數據測試"""
        # 解析批次數據
        # sample_batch is a tuple, extract the images and boxes
        images = sample_batch[0].to(device)  # Assuming first element contains images
        boxes = sample_batch[1] if len(sample_batch) > 1 else []
        
        # 創建隨機特徵圖
        batch_size = images.shape[0]
        channels = 64
        height, width = 32, 32
        features = torch.randn(batch_size, channels, height, width).to(device)
        
        # 初始化 ContextualRoIAlign
        contextual_roi_align = ContextualRoIAlign(output_size=7).to(device)
        
        # 前向傳播 - 使用實際的 Grozi 數據
        roi_features = contextual_roi_align(features, boxes[0].unsqueeze(0))
        
        # 檢查輸出
        assert roi_features is not None
        assert isinstance(roi_features, list)
        # Check the first batch in the list
        assert len(roi_features) > 0
        # Since roi_features is a list of tensors, we need to check individual tensors
        if roi_features[0].numel() > 0:  # Check if tensor is not empty
            assert roi_features[0].shape[1] == channels
            assert roi_features[0].shape[2] == 7  # output_size
            assert roi_features[0].shape[3] == 7  # output_size
class TestIntegration:
    def test_end_to_end_with_grozi(self, sample_batch, device):
        """使用 Grozi 數據的端到端測試"""
        # 解析批次數據
        images = sample_batch[0].to(device)
        batch_size = images.shape[0]
        
        # Grozi 數據集的特點是每個圖像有多個類別的物體
        # 獲取所有框，每個批次的框作為列表的一個元素
        all_boxes = []
        if len(sample_batch) > 1:
            # Ensure we have one list element per batch
            for i in range(batch_size):
                if i < len(sample_batch[1]) and sample_batch[1][i] is not None:
                    all_boxes.append(sample_batch[1][i].to(device))
                else:
                    # Add empty tensor if no boxes for this batch
                    all_boxes.append(torch.empty(0, 4).to(device))
        
        # 創建簡單的骨幹網絡模擬
        backbone = nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)
        
        # 初始化輔助網絡 - 使用 Grozi 數據集的類別數 (1063)
        auxiliary_network = AuxiliaryNetwork(in_channels=64, num_classes=1063).to(device)
        
        # 通過骨幹網絡
        features = backbone(images)
        
        # 通過輔助網絡
        cls_scores, bbox_preds = auxiliary_network(features, all_boxes)
        
        # 檢查輸出
        assert cls_scores is not None
        assert bbox_preds is not None
        
        # Check that we have a list with the correct number of elements
        assert len(cls_scores) == batch_size
        assert len(bbox_preds) == batch_size
        
        # Check that each element in the list has the correct format
        for i in range(batch_size):
            if isinstance(cls_scores[i], torch.Tensor) and cls_scores[i].numel() > 0:
                assert cls_scores[i].dim() > 0
            if isinstance(bbox_preds[i], torch.Tensor) and bbox_preds[i].numel() > 0:
                assert bbox_preds[i].dim() > 0
