import os
import torch
import pytest
from torch import nn
from os2d.data.dataloader import build_eval_dataloaders_from_cfg, build_train_dataloader_from_config
from os2d.config import cfg
from os2d.modeling.box_coder import Os2dBoxCoder
from os2d.modeling.feature_extractor import build_feature_extractor
from src.auxiliary_network import AuxiliaryNetwork
from src.channel_selector import OS2DChannelSelector
from os2d.modeling.feature_extractor import build_feature_extractor

from os2d.engine.objective import Os2dObjective
from os2d.utils import count_model_parameters, masked_select_or_fill_constant
from os2d.structures.feature_map import FeatureMapSize
from src.auxiliary_network import AuxiliaryNetwork, ContextualRoIAlign
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

class TestChannelSelector:
    def test_initialization(self, device):
        """測試初始化"""
        # 初始化模型和輔助網絡
        model = build_feature_extractor("resnet50").to(device)
        auxnet = AuxiliaryNetwork(in_channels=64).to(device)
        
        # 初始化通道選擇器
        selector = OS2DChannelSelector(model=model, auxiliarynet=auxnet, alpha=0.6, beta=0.4)
        
        # 檢查屬性
        assert selector.model == model
        assert selector.auxiliarynet == auxnet
        assert selector.alpha == 0.6
        assert selector.beta == 0.4
    
    def test_get_features(self, device):
        """測試獲取特徵"""
        # 初始化模型和輔助網絡
        model = build_feature_extractor("resnet50").to(device)
        auxnet = AuxiliaryNetwork(in_channels=64).to(device)
        
        # 初始化通道選擇器
        selector = OS2DChannelSelector(model=model, auxiliarynet=auxnet)
        
        # 創建隨機輸入
        batch_size = 2
        channels = 3
        height, width = 224, 224
        images = torch.randn(batch_size, channels, height, width).to(device)
        
        # 獲取特徵
        layer_name = "layer1.0.conv1"
        features = selector.get_features(layer_name, images)
        
        # 檢查特徵形狀
        assert features is not None
        assert features.shape[0] == batch_size
        
        # 獲取目標層
        target_layer = model
        for part in layer_name.split('.'):
            if part.isdigit():
                target_layer = target_layer[int(part)]
            else:
                target_layer = getattr(target_layer, part)
        
        # 檢查通道數
        assert features.shape[1] == target_layer.out_channels
    
    def test_compute_cls_loss(self, device):
        """測試計算分類損失"""
        # 初始化模型和輔助網絡
        model = build_feature_extractor("resnet50").to(device)
        auxnet = AuxiliaryNetwork(in_channels=64).to(device)
        
        # 初始化通道選擇器
        selector = OS2DChannelSelector(model=model, auxiliarynet=auxnet)
        
        # 創建隨機特徵圖
        batch_size = 2
        channels = 64
        height, width = 32, 32
        features = torch.randn(batch_size, channels, height, width).to(device)
        
        # 創建隨機框
        num_boxes = 5
        boxes = torch.rand(batch_size, num_boxes, 4).to(device)
        boxes[:, :, 2:] += boxes[:, :, :2]  # 確保 x2 > x1, y2 > y1
        
        # 計算分類損失
        cls_loss = selector.compute_cls_loss(features, boxes)
        
        # 檢查損失
        assert cls_loss is not None
        assert cls_loss.item() > 0.0
    
    def compute_aux_loss(self, cls_scores, reg_preds, boxes):
        """
        計算輔助網絡損失
        
        Args:
            cls_scores (Tensor): 分類分數
            reg_preds (Tensor): 回歸預測
            boxes (Tensor): 邊界框
        
        Returns:
            Tensor: 輔助網絡損失
        """
        # 檢查 cls_scores 是列表還是張量
        if isinstance(cls_scores, list):
            # 如果是列表，處理第一個元素
            cls_scores = cls_scores[0]
        
        # 確保 cls_scores 是合適的形狀 [batch_size, num_classes]
        if cls_scores.dim() > 2:
            # 如果 cls_scores 是多維的，將其重塑為 [batch_size, num_classes]
            cls_scores = cls_scores.view(cls_scores.size(0), -1)
        
        # 創建分類目標
        cls_targets = torch.zeros(cls_scores.size(0), dtype=torch.long, device=cls_scores.device)
        
        # 處理回歸目標，確保形狀匹配
        if isinstance(boxes, list):
            # 如果是列表，取第一個元素
            reg_targets = boxes[0]
        else:
            # 如果是張量，檢查形狀
            if boxes.dim() == 3:  # [batch_size, num_boxes, 4]
                # 取每個批次的第一個框
                reg_targets = boxes[:, 0, :]  # [batch_size, 4]
            else:
                reg_targets = boxes
        
        # 確保 reg_targets 與 reg_preds 形狀匹配
        if reg_targets.shape != reg_preds.shape:
            # 如果形狀不匹配，調整 reg_targets
            if reg_targets.size(0) == reg_preds.size(0):
                # 批次大小相同，但其他維度不同
                reg_targets = reg_targets[:, :reg_preds.size(1)]
            else:
                # 批次大小不同，創建一個與 reg_preds 相同形狀的零張量
                reg_targets = torch.zeros_like(reg_preds)
        
        # 計算損失
        cls_loss = F.cross_entropy(cls_scores, cls_targets)
        reg_loss = F.smooth_l1_loss(reg_preds, reg_targets)
        
        # 結合損失
        aux_loss = cls_loss + reg_loss
        
        return aux_loss

    def test_compute_importance(self, device):
        """測試計算通道重要性"""
        # 初始化模型和輔助網絡
        model = build_feature_extractor("resnet50").to(device)
        auxnet = AuxiliaryNetwork(in_channels=64).to(device)
        
        # 初始化通道選擇器
        selector = OS2DChannelSelector(model=model, auxiliarynet=auxnet)
        
        # 創建隨機輸入
        batch_size = 2
        channels = 3
        height, width = 224, 224
        images = torch.randn(batch_size, channels, height, width).to(device)
        
        # 創建隨機框
        num_boxes = 5
        boxes = torch.rand(batch_size, num_boxes, 4).to(device)
        boxes[:, :, 2:] += boxes[:, :, :2]  # 確保 x2 > x1, y2 > y1
        
        # 計算通道重要性
        layer_name = "layer1.0.conv1"
        importance = selector.compute_importance(layer_name, images, boxes)
        
        # 獲取目標層
        target_layer = model
        for part in layer_name.split('.'):
            if part.isdigit():
                target_layer = target_layer[int(part)]
            else:
                target_layer = getattr(target_layer, part)
        
        # 檢查重要性形狀
        assert importance is not None
        assert importance.shape[0] == target_layer.out_channels
    
    def test_select_channels(self, device):
        """測試選擇通道"""
        # 初始化模型和輔助網絡
        model = build_feature_extractor("resnet50").to(device)
        auxnet = AuxiliaryNetwork(in_channels=64).to(device)
        
        # 初始化通道選擇器
        selector = OS2DChannelSelector(model=model, auxiliarynet=auxnet)
        
        # 創建隨機輸入
        batch_size = 2
        channels = 3
        height, width = 224, 224
        images = torch.randn(batch_size, channels, height, width).to(device)
        
        # 創建隨機框
        num_boxes = 5
        boxes = torch.rand(batch_size, num_boxes, 4).to(device)
        boxes[:, :, 2:] += boxes[:, :, :2]  # 確保 x2 > x1, y2 > y1
        
        # 選擇通道
        layer_name = "layer1.0.conv1"
        percentage = 0.5
        selected_indices = selector.select_channels(layer_name, images, boxes, percentage)
        
        # 獲取目標層
        target_layer = model
        for part in layer_name.split('.'):
            if part.isdigit():
                target_layer = target_layer[int(part)]
            else:
                target_layer = getattr(target_layer, part)
        
        # 檢查選擇的通道數量
        num_channels = target_layer.out_channels
        num_to_keep = int(num_channels * percentage)
        assert len(selected_indices) == num_to_keep
        
        # 檢查選擇的通道索引是否在有效範圍內
        assert all(0 <= idx < num_channels for idx in selected_indices)
    
    def test_with_grozi_data(self, sample_batch, device):
        """使用 Grozi 數據測試"""
        # 解析批次數據
        images = sample_batch[0].to(device)  # Assuming first element contains images
        boxes = sample_batch[1] if len(sample_batch) > 1 else []

        # 初始化模型和輔助網絡
        model = build_feature_extractor("resnet50").to(device)
        auxnet = AuxiliaryNetwork(in_channels=64, num_classes=1063).to(device)
        
        # 初始化通道選擇器
        selector = OS2DChannelSelector(model=model, auxiliarynet=auxnet)
        
        # 選擇通道
        layer_name = "layer1.0.conv1"
        selected_indices = selector.select_channels(layer_name, images, boxes, percentage=0.5)
        
        # 檢查選擇的通道索引
        assert selected_indices is not None
        assert len(selected_indices) > 0
        
        # 獲取目標層
        target_layer = model
        for part in layer_name.split('.'):
            if part.isdigit():
                target_layer = target_layer[int(part)]
            else:
                target_layer = getattr(target_layer, part)
        
        # 檢查選擇的通道數量
        num_channels = target_layer.out_channels
        num_to_keep = int(num_channels * 0.5)
        assert len(selected_indices) == num_to_keep
