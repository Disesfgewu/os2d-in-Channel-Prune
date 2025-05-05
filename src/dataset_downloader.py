import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.ops as ops
import argparse
import os
import re
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from tqdm import tqdm
import copy
import time
import tarfile
from torchvision.models import resnet50
import urllib.request
import shutil
# from channel_selector import *
# from auxiliary_network import *
# from check_point import *
# from context_roi_align import *
# from gIoU_loss import *
# from lcp_channel_selector import *
# from os2d_resnet import *

# 全局變數 - 使類別對象可訪問這些類別
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class VOCDataset(torch.utils.data.Dataset):
    # 添加靜態類變數以便測試可以訪問
    CLASSES = VOC_CLASSES
    
    def __init__(self, data_path, split='train', transform=None, target_transform=None, download=True, 
                 random_seed=None, img_size=(224,224), class_mapping=None):
        # 如果需要下載且路徑不存在有效數據集，則下載
        if download:
            self.data_path = self._download_voc(data_path)
        else:
            self.data_path = self._resolve_data_root(data_path)
            
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        self.cache = {}
        self.img_size = img_size
        self.class_mapping = class_mapping or {i: i for i in range(len(VOC_CLASSES))}
        
        # 設置隨機種子以確保確定性行為
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        # 路徑結構與分割檔案檢查
        self._fix_nested_path_structure()
        if not self._validate_dataset_structure():
            if download:
                print(f"⚠️ 找不到有效的VOC資料集，嘗試下載...")
                self.data_path = self._download_voc(data_path)
            else:
                raise FileNotFoundError(f"無效的VOC資料集結構於 {self.data_path}，請設置 download=True 自動下載")
        
        self._verify_split_files()
        self._precache_metadata()
        print(f"📦 載入 VOC2007 {split}集: {len(self.samples)} 個樣本 (快取版)")

    def _download_voc(self, path):
        """下載並解壓 Pascal VOC 數據集"""
        import tarfile
        import urllib.request
        import os
        
        # 創建目標目錄
        os.makedirs(path, exist_ok=True)
        
        # 檢查是否已存在 VOC2007 資料夾
        voc_path = os.path.join(path, 'VOC2007')
        if os.path.exists(voc_path) and os.path.exists(os.path.join(voc_path, 'JPEGImages')):
            print(f"✅ 找到已存在的 VOC2007 資料集: {voc_path}")
            return voc_path
            
        voc_devkit_path = os.path.join(path, 'VOCdevkit', 'VOC2007')
        if os.path.exists(voc_devkit_path) and os.path.exists(os.path.join(voc_devkit_path, 'JPEGImages')):
            print(f"✅ 找到已存在的 VOC2007 資料集: {voc_devkit_path}")
            return voc_devkit_path
            
        # VOC 下載 URL
        DOWNLOAD_URLS = [
            ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
             '34ed68851bce2a36e2a223fa52c661d592c66b3c'),
            ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
             '41a8d6e12baa5ab18ee7f8f8029b9e11805b4ef1')
        ]
        
        for url, checksum in DOWNLOAD_URLS:
            filename = os.path.join(path, url.split('/')[-1])
            # 檢查文件是否已經存在
            if not os.path.exists(filename):
                print(f"📥 下載 {url}")
                try:
                    urllib.request.urlretrieve(url, filename)
                    print(f"✅ 下載完成: {filename}")
                except Exception as e:
                    print(f"❌ 下載失敗: {e}")
                    continue
                    
            # 解壓文件
            print(f"📦 解壓 {filename}")
            try:
                with tarfile.open(filename) as tar:
                    tar.extractall(path=path)
                print(f"✅ 解壓完成")
            except Exception as e:
                print(f"❌ 解壓失敗: {e}")
                continue
                
        # 檢查最終路徑
        if os.path.exists(voc_devkit_path):
            print(f"📂 VOC2007 資料集位置: {voc_devkit_path}")
            return voc_devkit_path
        else:
            print(f"❌ 下載後找不到 VOC2007 資料集!")
            raise FileNotFoundError(f"下載後找不到 VOC2007 資料集!")

    def _resolve_data_root(self, data_root):
        """自動修正 VOCdevkit 路徑"""
        potential_paths = [
            data_root,
            os.path.join(data_root, 'VOCdevkit/VOC2007'),
            os.path.join(data_root, 'VOC2007'),
            os.path.join(data_root, 'VOCdevkit/VOCdevkit/VOC2007')
        ]
        for path in potential_paths:
            if os.path.exists(os.path.join(path, 'Annotations')):
                print(f"🔍 偵測到有效資料集路徑: {path}")
                return path
        return data_root

    def _fix_nested_path_structure(self):
        if 'VOCdevkit/VOCdevkit' in self.data_path:
            corrected = self.data_path.replace('VOCdevkit/VOCdevkit', 'VOCdevkit')
            if os.path.exists(corrected):
                print(f"🛠️ 自動修正嵌套路徑: {self.data_path} → {corrected}")
                self.data_path = corrected

    def _validate_dataset_structure(self):
        required_dirs = ['Annotations', 'JPEGImages', 'ImageSets/Main']
        return all(os.path.isdir(os.path.join(self.data_path, d)) for d in required_dirs)

    def _verify_split_files(self):
        # 支持的分割文件
        valid_splits = ['train', 'val', 'test', 'trainval']
        if self.split not in valid_splits:
            raise ValueError(f"無效的分割類型: {self.split}，支持的類型: {valid_splits}")
            
        split_file = os.path.join(self.data_path, 'ImageSets/Main', f'{self.split}.txt')
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"找不到分割檔案: {split_file}")

    def _precache_metadata(self):
        """並行預載所有標註資料到記憶體"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        split_file = os.path.join(self.data_path, 'ImageSets/Main', f'{self.split}.txt')
        with open(split_file, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]

        self.samples = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(self._safe_parse_annotation, img_id): img_id for img_id in ids}
            for future in tqdm(as_completed(futures), total=len(ids), desc="🚀 預載標註資料"):
                img_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        self.samples.append(img_id)
                        self.cache[img_id] = result
                except Exception as e:
                    print(f"⚠️ 跳過無效樣本 {img_id}: {str(e)}")

    def _safe_parse_annotation(self, img_id):
        img_path = os.path.join(self.data_path, 'JPEGImages', f'{img_id}.jpg')
        annot_path = os.path.join(self.data_path, 'Annotations', f'{img_id}.xml')
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"圖片檔案不存在: {img_path}")
        if not os.path.isfile(annot_path):
            raise FileNotFoundError(f"標註檔案不存在: {annot_path}")

        tree = ET.parse(annot_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in VOC_CLASSES:
                continue
            bbox = obj.find('bndbox')
            box = [
                float(bbox.find('xmin').text),
                float(bbox.find('ymin').text),
                float(bbox.find('xmax').text),
                float(bbox.find('ymax').text)
            ]
            boxes.append(box)
            label_idx = VOC_CLASSES.index(name)
            labels.append(label_idx)
            
        return img_path, boxes, labels

    def __getitem__(self, idx):
        """獲取資料集中的一個樣本，並提取類別圖像"""
        img_id = self.samples[idx]
        img_path, boxes, labels = self.cache[img_id]
        
        # 載入圖像
        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img).copy()
        
        # 確保所有數據都是張量格式
        if not isinstance(img, torch.Tensor):
            # 將圖像轉換為張量
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            else:
                img = torch.tensor(img, dtype=torch.float)
        
        # 確保邊界框是張量
        if isinstance(boxes, list):
            if len(boxes) > 0:
                boxes = torch.tensor(boxes, dtype=torch.float)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float)
        
        # 確保標籤是張量
        if isinstance(labels, list):
            if len(labels) > 0:
                # 應用類別映射
                mapped_labels = [self.class_mapping.get(l, l) for l in labels]
                labels = torch.tensor(mapped_labels, dtype=torch.long)
            else:
                labels = torch.zeros((0,), dtype=torch.long)
        
        # 轉換圖像通道順序
        if img.shape[0] == 3:  # 已經是 [C, H, W] 格式
            pass
        elif img.shape[2] == 3:  # 如果是 [H, W, C] 格式
            img = img.permute(2, 0, 1)
        
        # 正規化像素值到 [0, 1]
        if img.max() > 1.0:
            img = img / 255.0
        
        # 調整圖像尺寸 (如果指定了目標尺寸)
        original_size = (img.shape[2], img.shape[1])  # (W, H)
        if self.img_size is not None:
            # 調整圖像尺寸
            img = F.interpolate(img.unsqueeze(0), size=self.img_size, 
                               mode='bilinear', align_corners=False).squeeze(0)
            
            # 調整框座標 - 根據縮放比例
            if boxes.numel() > 0:
                scale_w = self.img_size[0] / original_size[0]
                scale_h = self.img_size[1] / original_size[1]
                
                # 應用縮放
                boxes[:, 0] *= scale_w  # x_min
                boxes[:, 2] *= scale_w  # x_max
                boxes[:, 1] *= scale_h  # y_min
                boxes[:, 3] *= scale_h  # y_max
                
                # 確保框座標在有效範圍內
                boxes[:, 0].clamp_(0, self.img_size[0])
                boxes[:, 2].clamp_(0, self.img_size[0])
                boxes[:, 1].clamp_(0, self.img_size[1])
                boxes[:, 3].clamp_(0, self.img_size[1])
                
        # 應用圖像轉換
        if self.transform:
            img = self.transform(img)
            
        # 應用目標轉換
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        
        # 關鍵改進：從目標區域提取類別圖像
        class_images = []
        class_size = 64  # 類別圖像標準尺寸
        
        if len(boxes) > 0:
            for i in range(min(len(boxes), 3)):  # 最多取3個目標區域作為類別圖像
                # 獲取框座標
                x1, y1, x2, y2 = boxes[i].tolist()
                
                # 確保座標是整數且在有效範圍內
                h, w = img.shape[1], img.shape[2]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
                
                if x2 > x1 and y2 > y1:
                    # 提取目標區域
                    class_img = img[:, y1:y2, x1:x2].clone()
                    
                    # 調整尺寸為標準類別圖像尺寸
                    class_img = torch.nn.functional.interpolate(
                        class_img.unsqueeze(0),  # [1, C, H, W]
                        size=(class_size, class_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # [C, class_size, class_size]
                    
                    class_images.append(class_img)
        
        # 如果沒有有效類別圖像，創建一個默認的
        if not class_images:
            # 使用中心區域作為默認類別圖像
            h, w = img.shape[1], img.shape[2]
            center_h, center_w = h // 2, w // 2
            size_h, size_w = h // 4, w // 4
            
            y1, y2 = max(0, center_h - size_h), min(h, center_h + size_h)
            x1, x2 = max(0, center_w - size_w), min(w, center_w + size_w)
            
            default_class = img[:, y1:y2, x1:x2].clone()
            default_class = torch.nn.functional.interpolate(
                default_class.unsqueeze(0),
                size=(class_size, class_size),  # 修正：添加大小參數
                mode='bilinear',  # 修正：指定插值模式
                align_corners=False
            ).squeeze(0)
            
            class_images.append(default_class)
        
        # 將類別圖像堆疊為單一張量
        if len(class_images) > 1:
            # 多個類別圖像，堆疊為 [num_classes, C, H, W]
            class_images = torch.stack(class_images)
        else:
            # 單個類別圖像，確保為 [1, C, H, W]
            class_images = class_images[0].unsqueeze(0)
        
        # print(f"VOC 返回: 圖像形狀={img.shape}, 框形狀={boxes.shape}, 類別形狀={labels.shape}, 類別圖像形狀={class_images.shape}")
        
        # 返回圖像、目標框、類別標籤和類別圖像
        return img, boxes, labels, class_images

    def __len__(self):
        return len(self.samples)
        
    @staticmethod
    def collate_fn(batch):
        """自定義批次整合函數，將主圖像resize為224×224，類別圖像resize為64×64（或224×224），並組成batch"""
        images = []
        boxes_list = []
        labels_list = []
        class_images_list = []

        for img, boxes, labels, class_images in batch:
            # 保證主圖像為 [3, 224, 224]
            if img.shape[1:] != (224, 224):
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
                ).squeeze(0)
            images.append(img)

            boxes_list.append(boxes)
            labels_list.append(labels)

            # 保證每個 class image 為 [N, 3, 64, 64] 或 [N, 3, 224, 224]
            resized_class_images = []
            for cimg in class_images:
                if cimg.shape[1:] != (64, 64):  # 你也可以用 (224, 224)
                    cimg = torch.nn.functional.interpolate(
                        cimg.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False
                    ).squeeze(0)
                resized_class_images.append(cimg)
            # 堆疊
            resized_class_images = torch.stack(resized_class_images)
            class_images_list.append(resized_class_images)

        # 組 batch
        images = torch.stack(images)  # [B, 3, 224, 224]
        # 類別圖像合併成一個大 tensor [sum_N, 3, 64, 64]
        all_class_images = torch.cat(class_images_list, dim=0)

        return [images, boxes_list, labels_list, all_class_images]
