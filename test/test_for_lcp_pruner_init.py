import os
import torch
import pytest
import numpy as np
from torch import nn

from os2d.data.dataloader import build_eval_dataloaders_from_cfg, build_train_dataloader_from_config
from os2d.config import cfg
from os2d.modeling.box_coder import Os2dBoxCoder
from src.lcp_pruner import LCPPruner
from os2d.engine.objective import Os2dObjective
from os2d.utils import count_model_parameters, masked_select_or_fill_constant
from os2d.structures.feature_map import FeatureMapSize
from src.auxiliary_network import AuxiliaryNetwork
from os2d.modeling.model import Os2dModel
from os2d.utils.logger import setup_logger
import logging
import torch.nn.functional as F

logger = setup_logger("OS2D")

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
    try:
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
        net = LCPPruner(
            logger=logger,
            is_cuda=torch.cuda.is_available(),
            backbone_arch="resnet50",
            alpha=0.6,
            beta=0.4,
            pruneratio=0.5
        )
        box_coder = Os2dBoxCoder(positive_iou_threshold=cfg.train.objective.positive_iou_threshold,
                                negative_iou_threshold=cfg.train.objective.negative_iou_threshold,
                                remap_classification_targets_iou_pos=cfg.train.objective.remap_classification_targets_iou_pos,
                                remap_classification_targets_iou_neg=cfg.train.objective.remap_classification_targets_iou_neg,
                                output_box_grid_generator=net.os2d_head_creator.box_grid_generator_image_level,
                                function_get_feature_map_size=net.get_feature_map_size,
                                do_nms_across_classes=cfg.eval.nms_across_classes)

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
    except Exception as e:
        # Fallback for testing without actual dataset
        print(f"Error loading Grozi dataset: {e}")
        return {
            "dataloader_train": None,
            "dataloaders_eval": None
        }

@pytest.fixture
def sample_batch(grozi_dataset):
    """獲取樣本批次"""
    try:
        dataloader = grozi_dataset["dataloader_train"]
        if dataloader is not None:
            # Get the first batch directly from the dataloader
            batch = dataloader.get_batch(0)  # Get the first batch (index 0)
            return batch
    except Exception as e:
        print(f"Error getting sample batch: {e}")
    
    # Fallback: create mock data
    images = torch.randn(2, 3, 224, 224)
    boxes = [
        torch.tensor([[50, 50, 150, 150], [100, 100, 200, 200]], dtype=torch.float32),
        torch.tensor([[30, 30, 130, 130]], dtype=torch.float32)
    ]
    return (images, boxes)

class TestLCPPrunerInit:
    def test_initialization(self, device):
        """測試初始化"""
        # 初始化 LCPPruner
        pruner = LCPPruner(
            logger=logger,
            is_cuda=torch.cuda.is_available(),
            backbone_arch="resnet50",
            alpha=0.6,
            beta=0.4,
            pruneratio=0.5
        )
        
        # 檢查屬性
        assert pruner.alpha == 0.6
        assert pruner.beta == 0.4
        assert pruner.pruneratio == 0.5
        assert hasattr(pruner, 'auxiliary_network')
        assert hasattr(pruner, 'channel_selector')
        assert hasattr(pruner, 'lcp_loss')
        
        # 檢查輔助網絡
        assert isinstance(pruner.auxiliary_network, AuxiliaryNetwork)
        
        # 檢查通道選擇器
        assert hasattr(pruner.channel_selector, 'model')
        assert hasattr(pruner.channel_selector, 'auxiliarynet')
        assert pruner.channel_selector.alpha == 0.6
        assert pruner.channel_selector.beta == 0.4
    
    def test_build_auxiliary_network(self, device):
        """測試構建輔助網絡"""
        # 初始化 LCPPruner
        pruner = LCPPruner(
            logger=logger,
            is_cuda=torch.cuda.is_available(),
            backbone_arch="resnet50"
        )
        
        # 檢查輔助網絡
        aux_net = pruner.auxiliary_network
        assert isinstance(aux_net, AuxiliaryNetwork)
        
        # 檢查輸入通道數
        # Based on log, the input channels should be 2048 for ResNet50
        expected_channels = 2048  # ResNet50's final feature map channels
        assert aux_net.conv.in_channels == expected_channels
        
        # 檢查類別數
        assert aux_net.cls_head[-1].out_channels == 1063  # Grozi 數據集的類別數
    
    def test_init_channel_selector(self, device):
        """測試初始化通道選擇器"""
        # 初始化 LCPPruner
        pruner = LCPPruner(
            logger=logger,
            is_cuda=torch.cuda.is_available(),
            backbone_arch="resnet50",
            alpha=0.7,
            beta=0.3
        )
        
        # 檢查通道選擇器
        selector = pruner.channel_selector
        assert selector.model == pruner
        assert selector.auxiliarynet == pruner.auxiliary_network
        assert selector.alpha == 0.7
        assert selector.beta == 0.3
    
    def test_with_different_parameters(self, device):
        """測試不同參數的初始化"""
        # 測試不同的骨幹網絡
        pruner1 = LCPPruner(
            logger=logger,
            backbone_arch="resnet50",
            alpha=0.6,
            beta=0.4,
            pruneratio=0.5
        )
        # Use the actual model's attribute instead of a non-existent backbone_arch attribute
        # Check the backbone type through the feature extractor's class name
        # Get the backbone name by navigating through the pruner's attributes
        backbone_name = ""
        if hasattr(pruner1, 'net_feature_maps') and hasattr(pruner1.net_feature_maps, 'backbone'):
            backbone_name = pruner1.net_feature_maps.backbone.__class__.__name__
        else:
            # If net_feature_maps.backbone is not accessible, use net_feature_maps directly
            backbone_name = pruner1.net_feature_maps.__class__.__name__
        assert "ResNet" in backbone_name

        # 測試不同的剪枝比例
        pruner2 = LCPPruner(
            logger=logger,
            backbone_arch="resnet50",
            alpha=0.6,
            beta=0.4,
            pruneratio=0.3
        )
        assert pruner2.pruneratio == 0.3
        
        # 測試不同的損失權重
        pruner3 = LCPPruner(
            logger=logger,
            backbone_arch="resnet50",
            alpha=0.8,
            beta=0.2,
            pruneratio=0.5
        )
        assert pruner3.alpha == 0.8
        assert pruner3.beta == 0.2
        assert pruner3.channel_selector.alpha == 0.8
        assert pruner3.channel_selector.beta == 0.2
