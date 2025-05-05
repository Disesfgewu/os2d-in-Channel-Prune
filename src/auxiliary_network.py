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
    """
    LCP 輔助網路，用於評估通道重要性
    支持動態更新輸入通道數，保持梯度連續性
    """
    def __init__(self, in_channels=64, num_classes=len(VOC_CLASSES)):
        super(AuxiliaryNetwork, self).__init__()
        self.hidden_dim = 256  # Define hidden dimension
        self._init_layers(in_channels, num_classes)
    
    def _init_layers(self, in_channels, num_classes):
        """模塊化初始化層結構"""
        # 特徵轉換層
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.contextual_roi_align = ContextualRoIAlign(output_size=7)
        
        # 分類分支
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )
        
        # 回歸分支
        self.reg_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 4)  # 邊界框坐標 (x1,y1,x2,y2)
        )
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """使用 Kaiming 初始化保持梯度穩定性"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def update_input_channels(self, new_channels):
        """動態更新輸入通道 (保持梯度連續性)"""
        if not isinstance(new_channels, int) or new_channels <= 0:
            raise ValueError(f"無效的通道數: {new_channels}，必須是正整數")
        
        if self.conv.in_channels == new_channels:
            return  # 無變化時快速返回
        
        # 保留原始設備信息
        device = self.conv.weight.device
        old_weight = self.conv.weight.data
        old_bias = self.conv.bias.data if self.conv.bias is not None else None
        
        # 創建新卷積層
        new_conv = nn.Conv2d(
            new_channels,
            self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            bias=self.conv.bias is not None
        ).to(device)
        
        # 智能權重移植策略
        with torch.no_grad():
            if new_channels <= self.conv.in_channels:
                # 通道減少：直接裁剪
                new_conv.weight.copy_(old_weight[:, :new_channels])
            else:
                # 通道增加：混合策略
                # 1. 移植現有通道
                new_conv.weight[:, :self.conv.in_channels].copy_(old_weight)
                # 2. 新通道使用現有通道的均值初始化
                channel_mean = old_weight.mean(dim=1, keepdim=True)
                new_channels_to_add = new_channels - self.conv.in_channels
                new_conv.weight[:, self.conv.in_channels:].copy_(
                    channel_mean.expand(-1, new_channels_to_add, -1, -1)
                )
            
            # 偏置項處理
            if old_bias is not None:
                new_conv.bias.copy_(old_bias)
        
        # 無縫替換卷積層
        self.conv = new_conv
        print(f"✓ 輔助網路輸入通道更新: {self.conv.in_channels} → {new_channels}")
    
    def forward(self, features, boxes, gt_boxes=None):
        """增強型前向傳播，增加張量驗證"""
        # 輸入驗證
        if features.dim() != 4:
            raise ValueError(f"輸入特徵圖維度錯誤，應為4D張量，得到 {features.dim()}D")
        
        # 特徵轉換
        x = F.relu(self.conv(features))
        
        # ROI 對齊
        x = self.contextual_roi_align(x, boxes, gt_boxes)
        
        # 空張量處理
        if x.numel() == 0:
            empty_cls = torch.empty((0, self.cls_head[-1].out_features), device=features.device)
            empty_reg = torch.empty((0, 4), device=features.device)
            return empty_cls, empty_reg
        
        # 雙分支輸出
        return (
            self.cls_head(x),
            self.reg_head(x)
        )
    
    def get_current_channels(self):
        """獲取當前輸入通道數 (用於驗證)"""
        return self.conv.in_channels

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