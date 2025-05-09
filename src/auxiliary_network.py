# src/auxiliary_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .contextual_roi_align import ContextualRoIAlign

# VOC 數據集的類別
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class AuxiliaryNetwork(nn.Module):
    """用於 LCP 通道剪枝的輔助分類網路，具有更強的學習能力和穩定性"""
    
    def __init__(self, in_channels=2048, hidden_dim=512, num_classes=len(VOC_CLASSES), roi_size=7, dropout_prob=0.3):
        super(AuxiliaryNetwork, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # RoI 對齊
        self.contextual_roi_align = ContextualRoIAlign(output_size=roi_size)
        
        # 特徵處理
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # 添加額外卷積層增強特徵提取能力
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        
        # 添加殘差連接
        self.res_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1) if in_channels != hidden_dim else nn.Identity()
        
        # 分類頭
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),  # 添加Dropout增強泛化能力
            nn.Linear(hidden_dim, num_classes),
        )
        
        # 回歸頭 (預測框調整)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim//2, 4),  # 預測框調整量
        )
        
        # 重置參數
        self._init_weights()
        
        # 記錄最近的損失值
        self.recent_losses = []
        
    def _init_weights(self):
        """初始化網路參數"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def update_input_channels(self, new_channels):
        """更新輸入通道數"""
        if new_channels != self.in_channels:
            print(f"更新輔助網路輸入通道: {self.in_channels} → {new_channels}")
            
            # 保留原始設備信息
            device = self.conv1.weight.device
            
            # 更新第一個卷積層
            old_conv1 = self.conv1
            self.conv1 = nn.Conv2d(new_channels, self.hidden_dim, kernel_size=3, padding=1).to(device)
            
            # 初始化新的卷積層
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            if self.conv1.bias is not None:
                nn.init.constant_(self.conv1.bias, 0)
                
            # 更新殘差連接的卷積層
            if new_channels != self.hidden_dim:
                self.res_conv = nn.Conv2d(new_channels, self.hidden_dim, kernel_size=1).to(device)
                nn.init.kaiming_normal_(self.res_conv.weight, mode='fan_out', nonlinearity='relu')
                if self.res_conv.bias is not None:
                    nn.init.constant_(self.res_conv.bias, 0)
            else:
                self.res_conv = nn.Identity()
            
            # 更新通道數
            self.in_channels = new_channels
            
            print(f"✓ 輔助網路輸入通道更新完成: {self.in_channels}")
            
    def get_current_channels(self):
        """獲取當前輸入通道數"""
        return self.in_channels
        
    def forward(self, features, boxes, gt_boxes=None):
        """
        前向傳播處理
        Args:
            features: 特徵圖，形狀為 [B, C, H, W]
            boxes: RoI 坐標
            gt_boxes: 可選，真實框坐標
        Returns:
            tuple: (分類分數, 預測框調整量)
        """
        # 輸入驗證
        if features.dim() != 4:
            raise ValueError(f"輸入特徵圖維度錯誤，應為4D張量，得到 {features.dim()}D")
        
        # 提取 RoI 特徵
        x = self.contextual_roi_align(features, boxes, gt_boxes)
        
        # 空張量處理
        if x.numel() == 0:
            empty_cls = torch.empty((0, self.num_classes), device=features.device)
            empty_reg = torch.empty((0, 4), device=features.device)
            return empty_cls, empty_reg
        
        # 特徵處理
        identity = x  # 保存輸入以創建殘差連接
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # 殘差連接
        identity = self.res_conv(identity)
        x = x + identity
        x = F.relu(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 雙分支輸出
        return (
            self.cls_head(x),
            self.reg_head(x)
        )
    
    def record_loss(self, loss_value):
        """記錄損失值用於監控"""
        self.recent_losses.append(loss_value)
        # 只保留最近 100 個損失值
        if len(self.recent_losses) > 100:
            self.recent_losses = self.recent_losses[-100:]
            
    def get_loss_stats(self):
        """獲取損失統計信息"""
        if not self.recent_losses:
            return {"mean": 0, "min": 0, "max": 0, "std": 0}
            
        losses = torch.tensor(self.recent_losses)
        return {
            "mean": losses.mean().item(),
            "min": losses.min().item(),
            "max": losses.max().item(),
            "std": losses.std().item()
        }

    def train(self, mode=True):
        """
        設置網路為訓練模式，並確保所有子模塊同步更新狀態
        參數:
            mode (bool): True為訓練模式，False為評估模式
        返回:
            self: 返回網路實例以支持鏈式調用
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self