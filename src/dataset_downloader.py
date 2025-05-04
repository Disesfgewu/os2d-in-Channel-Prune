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

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split='train', transform=None, download=True):
        # 如果需要下載且路徑不存在有效數據集，則下載
        if download:
            self.data_path = self._download_voc(data_path)
        else:
            self.data_path = self._resolve_data_root(data_path)
            
        self.split = split
        self.transform = transform
        self.samples = []
        self.cache = {}

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
            labels.append(VOC_CLASSES.index(name))
        return (img_path, boxes, labels)

    def __getitem__(self, idx):
        img_id = self.samples[idx]
        img_path, boxes, labels = self.cache[img_id]
        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img).copy()
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(boxes), torch.tensor(labels), img_id

    def __len__(self):
        return len(self.samples)
