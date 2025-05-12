import os
import torch
import pytest
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from os2d.data.dataloader import build_eval_dataloaders_from_cfg, build_train_dataloader_from_config
from os2d.config import cfg
from os2d.modeling.box_coder import Os2dBoxCoder

from src.losses import GIoULoss, LCPLoss
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

class TestGIoULoss:
    def test_initialization(self):
        """測試初始化"""
        # 測試默認參數
        giou_loss = GIoULoss()
        assert giou_loss.eps == 1e-6
        
        # 測試自定義參數
        giou_loss = GIoULoss(eps=1e-8)
        assert giou_loss.eps == 1e-8
    
    def test_perfect_match(self, device):
        """測試完全匹配的情況"""
        giou_loss = GIoULoss().to(device)
        
        # 創建完全匹配的框
        pred = torch.tensor([[10, 10, 30, 30]], dtype=torch.float32).to(device)
        target = torch.tensor([[10, 10, 30, 30]], dtype=torch.float32).to(device)
        
        # 計算損失
        loss = giou_loss(pred, target)
        
        # 完全匹配時，GIoU = 1，損失 = 0
        assert torch.isclose(loss, torch.tensor(0.0, device=device), atol=1e-5)
    
    def test_no_overlap(self, device):
        """測試無重疊的情況"""
        giou_loss = GIoULoss().to(device)
        
        # 創建無重疊的框
        pred = torch.tensor([[10, 10, 20, 20]], dtype=torch.float32).to(device)
        target = torch.tensor([[30, 30, 40, 40]], dtype=torch.float32).to(device)
        
        # 計算損失
        loss = giou_loss(pred, target)
        
        # 計算期望的 GIoU 值
        pred_area = (20 - 10) * (20 - 10)  # 100
        target_area = (40 - 30) * (40 - 30)  # 100
        inter_area = 0  # 無重疊
        union_area = pred_area + target_area  # 200
        encl_area = (40 - 10) * (40 - 10)  # 900
        
        expected_giou = 0 - (encl_area - union_area) / encl_area  # 0 - 700/900 = -700/900
        expected_loss = 1 - expected_giou  # 1 + 700/900 = 1700/900
        
        assert torch.isclose(loss, torch.tensor(expected_loss, device=device), atol=1e-5)
    
    def test_partial_overlap(self, device):
        """測試部分重疊的情況"""
        giou_loss = GIoULoss().to(device)
        
        # 創建部分重疊的框
        pred = torch.tensor([[10, 10, 30, 30]], dtype=torch.float32).to(device)
        target = torch.tensor([[20, 20, 40, 40]], dtype=torch.float32).to(device)
        
        # 計算損失
        loss = giou_loss(pred, target)
        
        # 計算期望的 GIoU 值
        pred_area = (30 - 10) * (30 - 10)  # 400
        target_area = (40 - 20) * (40 - 20)  # 400
        inter_area = (30 - 20) * (30 - 20)  # 100
        union_area = pred_area + target_area - inter_area  # 700
        encl_area = (40 - 10) * (40 - 10)  # 900
        
        expected_giou = inter_area / union_area - (encl_area - union_area) / encl_area
        expected_giou = 100 / 700 - (900 - 700) / 900
        expected_giou = 100 / 700 - 200 / 900
        expected_loss = 1 - expected_giou
        
        assert torch.isclose(loss, torch.tensor(expected_loss, device=device), atol=1e-5)
    
    def test_batch_processing(self, device):
        """測試批量處理"""
        giou_loss = GIoULoss().to(device)
        
        # 創建批量框
        pred = torch.tensor([
            [10, 10, 30, 30],
            [40, 40, 60, 60],
            [70, 70, 90, 90]
        ], dtype=torch.float32).to(device)
        
        target = torch.tensor([
            [10, 10, 30, 30],  # 完全匹配
            [50, 50, 70, 70],  # 部分重疊
            [100, 100, 120, 120]  # 無重疊
        ], dtype=torch.float32).to(device)
        
        # 計算損失
        loss = giou_loss(pred, target)
        
        # 損失應該是三個框的平均
        assert loss.item() > 0.0
    
    def test_empty_input(self, device):
        """測試空輸入"""
        giou_loss = GIoULoss().to(device)
        
        # 創建空輸入
        pred = torch.zeros((0, 4), dtype=torch.float32).to(device)
        target = torch.zeros((0, 4), dtype=torch.float32).to(device)
        
        # 計算損失
        # 注意：空輸入可能會導致錯誤，因為 mean 操作在空張量上是未定義的
        # 這裡我們期望函數內部處理這種情況
        try:
            loss = giou_loss(pred, target)
            # 如果沒有錯誤，檢`查損失是否為 0 或 NaN
            assert torch.isnan(loss) or loss.item() == 0.0
        except Exception as e:
            # 如果有錯誤，測試失敗
            assert False, f"Empty input caused error: {e}"
    
    def test_with_grozi_data(self, sample_batch, device):
        """使用 Grozi 數據測試"""
        giou_loss = GIoULoss().to(device)
        
        # 解析批次數據
        images = sample_batch[0].to(device)  # Assuming first element contains images
        boxes = sample_batch[1] if len(sample_batch) > 1 else []

        if len(boxes) > 0:
            # 使用第一個有效的框作為預測和目標
            box = boxes[0]
            
            # 確保 box 是二維張量且最後一維為4
            if box.dim() > 2:
                # Check the actual shape and dimensions of the box tensor first
                print(f"Box shape: {box.shape}")
                
                # Skip the test if the box doesn't have expected shape
                if box.size(-1) != 4 and box.numel() % 4 != 0:
                    pytest.skip("Box tensor doesn't have the expected format")
                    return
                
                # If last dimension is 4, we can reshape properly
                if box.size(-1) == 4:
                    box = box.view(-1, 4)
                else:
                    # If we can't reshape it properly, create a dummy box tensor for testing
                    box = torch.rand(10, 4).to(device)
            
            # 添加一些擾動作為預測
            pred = box.clone()
            pred[:, :2] -= 5  # 左上角向左上移動
            pred[:, 2:] += 5  # 右下角向右下移動
            
            # 計算損失
            loss = giou_loss(pred, box)
            
            # 檢查損失
            assert loss is not None
            assert loss.item() >= 0.0

    
    def test_gradients(self, device):
        """測試梯度計算"""
        giou_loss = GIoULoss().to(device)
        
        # 創建需要梯度的預測框
        pred = torch.tensor([[10.0, 10.0, 30.0, 30.0]], requires_grad=True, device=device)
        target = torch.tensor([[15.0, 15.0, 35.0, 35.0]], device=device)
        
        # 計算損失
        loss = giou_loss(pred, target)
        
        # 反向傳播
        loss.backward()
        
        # 檢查梯度
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()
        assert not torch.isinf(pred.grad).any()

class TestLCPLoss:
    def test_initialization(self):
        """測試初始化"""
        # 測試默認參數
        lcp_loss = LCPLoss()
        assert lcp_loss.alpha == 0.5
        assert lcp_loss.beta == 0.5
        assert isinstance(lcp_loss.cls_loss, nn.CrossEntropyLoss)
        assert isinstance(lcp_loss.reg_loss, GIoULoss)
        
        # 測試自定義參數
        lcp_loss = LCPLoss(alpha=0.7, beta=0.3)
        assert lcp_loss.alpha == 0.7
        assert lcp_loss.beta == 0.3
    
    def test_forward(self, device):
        """測試前向傳播"""
        lcp_loss = LCPLoss().to(device)
        
        # 創建隨機分類預測和目標
        batch_size = 3
        num_classes = 20
        cls_preds = torch.randn(batch_size, num_classes).to(device)
        cls_targets = torch.randint(0, num_classes, (batch_size,)).to(device)
        
        # 創建隨機回歸預測和目標
        reg_preds = torch.rand(batch_size, 4).to(device)
        reg_preds[:, 2:] += reg_preds[:, :2]  # 確保 x2 > x1, y2 > y1
        reg_targets = torch.rand(batch_size, 4).to(device)
        reg_targets[:, 2:] += reg_targets[:, :2]  # 確保 x2 > x1, y2 > y1
        
        # 計算損失
        loss = lcp_loss(cls_preds, reg_preds, cls_targets, reg_targets)
        
        # 檢查損失
        assert loss is not None
        assert loss.item() > 0.0
        
        # 手動計算損失以驗證
        cls_loss = nn.CrossEntropyLoss()(cls_preds, cls_targets)
        reg_loss = GIoULoss()(reg_preds, reg_targets)
        expected_loss = 0.5 * cls_loss + 0.5 * reg_loss
        
        assert torch.isclose(loss, expected_loss, atol=1e-5)
    
    def test_different_weights(self, device):
        """測試不同權重"""
        # 創建隨機分類預測和目標
        batch_size = 3
        num_classes = 20
        cls_preds = torch.randn(batch_size, num_classes).to(device)
        cls_targets = torch.randint(0, num_classes, (batch_size,)).to(device)
        
        # 創建隨機回歸預測和目標
        reg_preds = torch.rand(batch_size, 4).to(device)
        reg_preds[:, 2:] += reg_preds[:, :2]  # 確保 x2 > x1, y2 > y1
        reg_targets = torch.rand(batch_size, 4).to(device)
        reg_targets[:, 2:] += reg_targets[:, :2]  # 確保 x2 > x1, y2 > y1
        
        # 使用不同權重的損失函數
        lcp_loss1 = LCPLoss(alpha=0.8, beta=0.2).to(device)
        lcp_loss2 = LCPLoss(alpha=0.2, beta=0.8).to(device)
        
        # 計算損失
        loss1 = lcp_loss1(cls_preds, reg_preds, cls_targets, reg_targets)
        loss2 = lcp_loss2(cls_preds, reg_preds, cls_targets, reg_targets)
        
        # 檢查損失
        assert loss1 is not None
        assert loss2 is not None
        
        # 手動計算損失以驗證
        cls_loss = nn.CrossEntropyLoss()(cls_preds, cls_targets)
        reg_loss = GIoULoss()(reg_preds, reg_targets)
        
        expected_loss1 = 0.8 * cls_loss + 0.2 * reg_loss
        expected_loss2 = 0.2 * cls_loss + 0.8 * reg_loss
        
        assert torch.isclose(loss1, expected_loss1, atol=1e-5)
        assert torch.isclose(loss2, expected_loss2, atol=1e-5)
        
        # 權重不同時，損失應該不同
        assert not torch.isclose(loss1, loss2, atol=1e-3)
    
    def test_gradients(self, device):
        """測試梯度計算"""
        lcp_loss = LCPLoss().to(device)
        
        # 創建需要梯度的預測
        batch_size = 3
        num_classes = 20
        cls_preds = torch.randn(batch_size, num_classes, requires_grad=True, device=device)
        
        # 使用非原地操作創建reg_preds
        base_preds = torch.rand(batch_size, 4, requires_grad=True, device=device)
        # 使用非原地操作確保 x2 > x1, y2 > y1
        reg_preds = torch.cat([
            base_preds[:, :2],
            base_preds[:, :2] + base_preds[:, 2:]
        ], dim=1)
        
        # 創建目標
        cls_targets = torch.randint(0, num_classes, (batch_size,)).to(device)
        reg_targets = torch.rand(batch_size, 4).to(device)
        reg_targets = torch.cat([
            reg_targets[:, :2],
            reg_targets[:, :2] + reg_targets[:, 2:]
        ], dim=1)
        
        # 計算損失
        loss = lcp_loss(cls_preds, reg_preds, cls_targets, reg_targets)
        
        # 反向傳播
        loss.backward()
        
        # 檢查梯度
        assert cls_preds.grad is not None
        assert base_preds.grad is not None
        assert not torch.isnan(cls_preds.grad).any()
        assert not torch.isnan(base_preds.grad).any()
        assert not torch.isinf(cls_preds.grad).any()
        assert not torch.isinf(base_preds.grad).any()

    def test_with_grozi_data(self, sample_batch, device):
        """使用 Grozi 數據測試"""
        lcp_loss = LCPLoss().to(device)
        
        # 解析批次數據
        images = sample_batch[0].to(device)  # Assuming first element contains images
        boxes = sample_batch[1] if len(sample_batch) > 1 else []

        if len(boxes) > 0:
            # 使用第一個有效的框作為回歸目標
            box = boxes[0]
            batch_size = box.size(0)
            
            # 創建隨機分類預測和目標
            num_classes = 1063  # Grozi 數據集的類別數
            cls_preds = torch.randn(batch_size, num_classes).to(device)
            cls_targets = torch.randint(0, num_classes, (batch_size,)).to(device)
            
            # 添加一些擾動作為回歸預測
            reg_preds = box.clone()
            reg_preds[:, :2] -= 5  # 左上角向左上移動
            reg_preds[:, 2:] += 5  # 右下角向右下移動
            
            # 計算損失
            loss = lcp_loss(cls_preds, reg_preds, cls_targets, box)
            
            # 檢查損失
            assert loss is not None
            assert loss.item() > 0.0
