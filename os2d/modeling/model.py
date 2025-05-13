import os
import math
import numbers
import time
import logging
from collections import OrderedDict
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import build_feature_extractor

from .box_coder import Os2dBoxCoder, BoxGridGenerator
from os2d.engine.objective import Os2dObjective
from os2d.utils import count_model_parameters, masked_select_or_fill_constant
from os2d.structures.feature_map import FeatureMapSize
from .head import build_os2d_head_creator


def build_os2d_from_config(cfg):
    logger = logging.getLogger("OS2D")

    logger.info("Building the OS2D model")
    img_normalization = {"mean":cfg.model.normalization_mean, "std": cfg.model.normalization_std}
    net = LCPPruner(logger=logger,
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
    criterion = Os2dObjective(class_loss=cfg.train.objective.class_objective,
                              margin=cfg.train.objective.neg_margin,
                              margin_pos=cfg.train.objective.pos_margin,
                              class_loss_neg_weight=cfg.train.objective.class_neg_weight,
                              remap_classification_targets=cfg.train.objective.remap_classification_targets,
                              localization_weight=cfg.train.objective.loc_weight,
                              neg_to_pos_ratio=cfg.train.objective.neg_to_pos_ratio,
                              rll_neg_weight_ratio=cfg.train.objective.rll_neg_weight_ratio)
    optimizer_state = net.init_model_from_file(cfg.init.model, init_affine_transform_path=cfg.init.transform)

    num_params, num_param_groups = count_model_parameters(net)
    logger.info("OS2D has {0} blocks of {1} parameters (before freezing)".format(num_param_groups, num_params))

    # freeze transform parameters if requested
    if cfg.train.model.freeze_transform:
        logger.info("Freezing the transform parameters")
        net.freeze_transform_params()

    num_frozen_extractor_blocks = cfg.train.model.num_frozen_extractor_blocks
    if num_frozen_extractor_blocks > 0:
        logger.info("Freezing {0} of {1} blocks of the feature extractor network".format(num_frozen_extractor_blocks, net.get_num_blocks_in_feature_extractor()))
        net.freeze_extractor_blocks(num_blocks=num_frozen_extractor_blocks)

    num_params, num_param_groups = count_model_parameters(net)
    logger.info("OS2D has {0} blocks of {1} trainable parameters".format(num_param_groups, num_params))

    return net, box_coder, criterion, img_normalization, optimizer_state


class LabelFeatureExtractor(nn.Module):
    """LabelFeatureExtractor implements the feature extractor from query images.
    The main purpose pf this class is to run on a list of images of different sizes.
    """
    def __init__(self, feature_extractor):
        super(LabelFeatureExtractor, self).__init__()
        # backbone extractor
        self.net_class_features = feature_extractor
        
    def forward(self, class_image_list):
        list_of_feature_maps = []
        logger = logging.getLogger("FeatureExtractor")
        for class_image in class_image_list:
            # extract features from images
            logger.info(f"Extracting features for class images : class image shape { class_image.shape }")
            class_feature_maps = self.net_class_features(class_image.unsqueeze(0))
            list_of_feature_maps.append(class_feature_maps)
        
        return list_of_feature_maps

    def freeze_bn(self):
        # Freeze BatchNorm layers
        self.net_class_features.freeze_bn()

    def freeze_blocks(self, num_blocks=0):
        self.net_class_features.freeze_blocks(num_blocks)


def get_feature_map_size_for_network(img_size, net, is_cuda=False):
    """get_feature_map_size_for_network computes the size of the feature map when the network is applied to an image of specific size.
    The function creates a dummy image of required size, and just runs a network on it.
    This approach is very robust, but can be quite slow, so these calls shoulb be cached.
    Args:
        img_size (FeatureMapSize) - size of the input image
        net - the net to run
        is_cuda (bool) -flag showing where to put the dummy image on a GPU.
    Output:
        feature_map_size (FeatureMapSize) - the size of the feature map
    """
    dummy_image = torch.zeros(1, 3, img_size.h, img_size.w)  # batch_size, num_channels, height, width
    if is_cuda:
        dummy_image = dummy_image.cuda()

    with torch.no_grad():
        dummy_feature_maps = net(dummy_image)
        feature_map_size = FeatureMapSize(img=dummy_feature_maps)

    if is_cuda:
        torch.cuda.empty_cache()

    return feature_map_size


class Os2dModel(nn.Module):
    """The main class implementing the OS2D model.
    """
    default_normalization = {}
    default_normalization["mean"] = (0.485, 0.456, 0.406)
    default_normalization["std"] = (0.229, 0.224, 0.225)

    def __init__(self, logger,
                       is_cuda=True,
                       merge_branch_parameters=False, use_group_norm=False,
                       backbone_arch="resnet50",
                       use_inverse_geom_model=True,
                       simplify_affine=False,
                       img_normalization=None):
        super(Os2dModel, self).__init__()
        self.logger = logger
        self.use_group_norm = use_group_norm
        if img_normalization:
            self.img_normalization = img_normalization
        else:
            self.img_normalization = self.default_normalization
        self.net_feature_maps = build_feature_extractor(backbone_arch, use_group_norm)
        self.merge_branch_parameters = merge_branch_parameters
        extractor = self.net_feature_maps if self.merge_branch_parameters else build_feature_extractor(backbone_arch, use_group_norm)

        # net to regress parameters of the transform
        self.simplify_affine = simplify_affine
        self.use_inverse_geom_model = use_inverse_geom_model

        # new code fot the network heads
        self.os2d_head_creator = build_os2d_head_creator(self.simplify_affine, is_cuda, self.use_inverse_geom_model,
                                                         self.net_feature_maps.feature_map_stride,
                                                         self.net_feature_maps.feature_map_receptive_field)

        self.net_label_features = LabelFeatureExtractor(feature_extractor=extractor)

        # set the net to the eval mode by default, other wise batchnorm statistics get screwed when using dummy images to find feature map sizes
        self.eval()

        # decide GPU usage
        self.is_cuda = is_cuda

        if self.is_cuda:
            self.logger.info("Creating model on one GPU")
            self.cuda()
        else:            
            self.logger.info("Creating model on CPU")

    def train(self, mode=True, freeze_bn_in_extractor=False, freeze_transform_params=False, freeze_bn_transform=False):
        super(Os2dModel, self).train(mode)
        if freeze_bn_in_extractor:
            self.freeze_bn()
        if freeze_transform_params:
            self.freeze_transform_params()
        if freeze_bn_transform:
            self.os2d_head_creator.aligner.parameter_regressor.freeze_bn()

    def freeze_bn(self):
        # Freeze BatchNorm layers
        self.net_feature_maps.freeze_bn()
        self.net_label_features.freeze_bn()

    def freeze_transform_params(self):
        self.os2d_head_creator.aligner.parameter_regressor.eval()
        for param in self.os2d_head_creator.aligner.parameter_regressor.parameters():
            param.requires_grad = False
               
    def freeze_extractor_blocks(self, num_blocks=0):
        self.net_feature_maps.freeze_blocks(num_blocks)
        self.net_label_features.freeze_blocks(num_blocks)

    def get_num_blocks_in_feature_extractor(self):
        return self.net_feature_maps.get_num_blocks_in_feature_extractor()

    def apply_class_heads_to_feature_maps(self, feature_maps, class_head):
        """Applies class heads to the feature maps

        Args:
            feature_maps (Tensor) - feature maps of size batch_size x num_labels x height x width
            class_head (Os2dHead) - heads detecting some classes, created by an instance of Os2dHeadCreator

        Outputs:
            loc_scores (Tensor) - localization scores, size batch_size x num_labels x 4 x num_anchors
            class_scores (Tensor) - classification scores, size batch_size x num_labels x num_anchors
            class_scores_transform_detached (Tensor) - same as class_scores, but with transformations detached from the computational graph
            transform_corners (Tensor) - points representings transformations, size batch_size x num_labels x 8 x num_anchors
        """
        # self.logger.wa( "Applying class heads to feature maps")
        num_images = feature_maps.size(0)

        outputs = class_head(feature_maps)
        loc_scores = outputs[0]
        class_scores = outputs[1]
        num_labels = class_scores.size(1)
        class_scores_transform_detached = outputs[2]
        transform_corners = outputs[3]

        assert loc_scores.size(-2) == class_scores.size(-2), "Class and loc score should have same spatial sizes, but have {0} and {1}".format(class_scores.size(), loc_scores.size())
        assert loc_scores.size(-1) == class_scores.size(-1), "Class and loc score should have same spatial sizes, but have {0} and {1}".format(class_scores.size(), loc_scores.size())

        fmH = class_scores.size(-2)
        fmW = class_scores.size(-1)

        class_scores = class_scores.view(num_images, -1, fmH, fmW)
        class_scores_transform_detached = class_scores_transform_detached.view(num_images, -1, fmH, fmW)

        class_scores = class_scores.contiguous().view(num_images, num_labels, -1)
        class_scores_transform_detached = class_scores_transform_detached.contiguous().view(num_images, num_labels, -1)
        loc_scores = loc_scores.contiguous().view(num_images, num_labels, 4, -1) if loc_scores is not None else None
        transform_corners = transform_corners.contiguous().view(num_images, num_labels, 8, -1) if transform_corners is not None else None

        return loc_scores, class_scores, class_scores_transform_detached, transform_corners

    def forward(self, images=None, class_images=None,
                      feature_maps=None, class_head=None,
                      train_mode=False, fine_tune_features=True):
        """ Forward pass of the OS2D model. Cant function in several different regimes:
            [training mode] Extract features from input and class images, and applies the model to get 
                clasificaton/localization scores of all classes on all images
                Args:
                    images (tensor) - batch of input images
                    class_images (list of tensors) - list of class images (possibly of different sizes)
                    train_mode (bool) - should be True
                    fine_tune_features (bool) - flag showing whether to enable gradients over features
            [evaluation mode]
                    feature_maps (tensor) - pre-extracted feature maps, sized batch_size x feature_dim x height x width
                    class_head (Os2dHead) - head created to detect some classes,
                        inside has class_feature_maps, sized class_batch_size x feature_dim x class_height x class_width
                    train_mode (bool) - should be False
        Outputs:
            loc_scores (tensor) - localization prediction, sized batch_size x num_classes x 4 x num_anchors (bbox parameterization)
            class_scores (tensor) - classification prediction, sized batch_size x num_classes x num_anchors
            class_scores_transform_detached (tensor) - same, but with transofrms detached from the computational graph
                used not to tune transofrmation on the negative examples
            fm_sizes (FeatureMapSize) - size of the output score map, num_anchors == fm_sizes.w * fm_sizes.h
            transform_corners (tensor) - points defining parallelograms showing transformations, sized batch_size x num_classes x 8 x num_anchors
        """
        if images is not None and self.is_cuda:
            if images.device.type != 'cuda':
                images = images.cuda()
            
        if class_images is not None and self.is_cuda:
            if isinstance(class_images, list):
                class_images = [img.cuda() if img.device.type != 'cuda' else img for img in class_images]
            elif class_images.device.type != 'cuda':
                class_images = class_images.cuda()
        with torch.set_grad_enabled(train_mode and fine_tune_features):
            # extract features
            if feature_maps is None:
                assert images is not None, "If feature_maps is None than images cannot be None"
                feature_maps = self.net_feature_maps(images)

            # get features for labels
            if class_head is None:
                assert class_images is not None, "If class_conv_layer is None than class_images cannot be None"
                self.logger.info(f"Extracting features for class images : class image shpae { len(class_images) }")
                class_feature_maps = self.net_label_features(class_images)
                class_head = self.os2d_head_creator.create_os2d_head(class_feature_maps)

        # process features maps of different pyramid levels
        loc_scores, class_scores, class_scores_transform_detached, transform_corners = \
            self.apply_class_heads_to_feature_maps(feature_maps, class_head)

        fm_size = FeatureMapSize(img=feature_maps)
        return loc_scores, class_scores, class_scores_transform_detached, fm_size, transform_corners

    @lru_cache()
    def get_feature_map_size(self, img_size):
        """Computes the size of the feature map when the feature extractor is applied to the image of specific size.
        The calls are cached with @lru_cache() for speed.
        Args:
            img_size (FeatureMapSize)
        Output: FeatureMapSize
        """
        return get_feature_map_size_for_network(img_size=img_size,
                                                net=self.net_feature_maps,
                                                is_cuda=self.is_cuda)

    def init_model_from_file(self, path, init_affine_transform_path=""):
        """init_model_from_file loads weights from a binary file.
        It will try several ways to load the weights (in the order below) by doing the follwoing steps:
        1) Load full checkpoint (created by os2d.utils.logger.checkpoint_model)
            - reads file with torch.load(path)
            - expects a dict with "net" as a key which is expected to load with self.load_state_dict
            - if finds key "optimizer" as well outputs it to try to init from it later
        2) in (1) is not a success will try to init the backbone separately see _load_network
        3) if init_affine_transform_path is provided will try to additionally load the transformation model
            (CAREFUL! it will override weights from both (1) and (2))
        """
        # read the checkpoint file
        optimizer = None
        try:
            if path:
                self.logger.info("Reading model file {}".format(path))
                checkpoint = torch.load(path, weights_only=False)
            else:
                checkpoint = None

            if checkpoint and "net" in checkpoint:
                self.load_state_dict(checkpoint["net"])
                self.logger.info("Loaded complete model from checkpoint")
            else:
                self.logger.info("Cannot find 'net' in the checkpoint file")
                raise RuntimeError()

            if checkpoint and "optimizer" in checkpoint:
                optimizer = checkpoint["optimizer"]
                self.logger.info("Loaded optimizer from checkpoint")
            else:
                self.logger.info("Cannot find 'optimizer' in the checkpoint file. Initializing optimizer from scratch.")

        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.logger.info("Failed to load the full model, trying to init feature extractors")
            self._load_network(self.net_label_features.net_class_features, path=path)
            if not self.merge_branch_parameters:
                self._load_network(self.net_feature_maps, model_data=self.net_label_features.net_class_features.state_dict())

        if init_affine_transform_path:
            try:
                self.logger.info("Trying to init affine transform from {}".format(init_affine_transform_path))
                try:
                    model_data = torch.load(init_affine_transform_path, weights_only=False)
                except:
                    self.logger.info("Could not read the model file {0}.".format(path))
                assert hasattr(self, "os2d_head_creator") and hasattr(self.os2d_head_creator, "aligner") and hasattr(self.os2d_head_creator.aligner, "parameter_regressor"), "Need to have the affine regressor part to inialize it"
                init_from_weakalign_model(model_data["state_dict"], None,
                                          affine_regressor=self.os2d_head_creator.aligner.parameter_regressor)
                self.logger.info("Successfully initialized the affine transform from the provided weakalign model.")
            except:
                self.logger.info("Could not init affine transform from {0}.".format(init_affine_transform_path))

        return optimizer

    def _load_network(self, net, path=None, model_data=None):
        """_load_network loads weights from the provided path to a network net
        It will try several ways to load the weights (in the order below) by doing the follwoing steps:
        0) in model_data one can provide the already loaded weights (useful not to load many times) otherwise reads from path with torch.load
        1) tries to load complete network as net.load_state_dict(model_data)
        2) if fails, tries to load as follows: net.load_state_dict(model_data["net"], strict=False)
        3) if fails, tries to load from the weak align format with init_from_weakalign_model(model_data["state_dict"], self.net_feature_maps)
        4) if fails tries to partially init backbone with net.load_state_dict(model_data, strict=False) - works for the standard pytorch models
        """
        if model_data is None:
            try:
                self.logger.info("Trying to init from {}".format(path))
                model_data = torch.load(path, weights_only=False)
            except:
                self.logger.info("Could not read the model file {0}. Starting from scratch.".format(path))
        else:
            self.logger.info("Initializing from provided weights")
        if model_data is not None:
            try:
                net.load_state_dict(model_data)
            except:
                self.logger.info("FAILED to load as network")
                try:
                    self.logger.info("Trying to init from {} as checkpoint".format(path))
                    net.load_state_dict(model_data["net"], strict=False)
                    self.logger.info("Loaded epoch {0} with loss {1}".format(model_data["epoch"], model_data["loss"]))
                except:
                    self.logger.info("FAILED to load as checkpoint")
                    self.logger.info("Could not init the full feature extractor. Trying to init form a weakalign model")
                    try:
                        init_from_weakalign_model(model_data["state_dict"],
                                                  self.net_feature_maps)
                        self.logger.info("Successfully initialized form the provided weakalign model.")
                    except:
                        self.logger.info("Could not init from the weakalign network. Trying to init backbone from {}.".format(path))
                        try:
                            net.load_state_dict(model_data, strict=False)
                            self.logger.info("Successfully initialized backbone.")
                        except:
                            self.logger.info("Could not init anything. Starting from scratch.")


def init_from_weakalign_model(src_state_dict, feature_extractor=None, affine_regressor=None, tps_regressor=None):
    # init feature extractor - hve three blocks of ResNet101
    layer_prefix_map = {}  # layer_prefix_map[target prefix] = source prefix
    layer_prefix_map["conv1."] = "FeatureExtraction.model.0."
    layer_prefix_map["bn1."] = "FeatureExtraction.model.1."
    for idx in range(3):
        layer_prefix_map["layer1." + str(idx)] = "FeatureExtraction.model.4." + str(idx)
    for idx in range(4):
        layer_prefix_map["layer2." + str(idx)] = "FeatureExtraction.model.5." + str(idx)
    for idx in range(23):
        layer_prefix_map["layer3." + str(idx)] = "FeatureExtraction.model.6." + str(idx)

    if feature_extractor is not None:
        for k, v in feature_extractor.state_dict().items():
            found_init = False
            for k_map in layer_prefix_map:
                if k.startswith(k_map):
                    found_init = True
                    break
            if found_init:
                k_target = k.replace(k_map, layer_prefix_map[k_map])
                if k.endswith("num_batches_tracked"):
                    continue
                # print("Copying from {0} to {1}, size {2}".format(k_target, k, v.size()))
                v.copy_(src_state_dict[k_target])
    
    for regressor, prefix in zip([affine_regressor, tps_regressor], ["FeatureRegression.", "FeatureRegression2."]):
        if regressor is not None:
            for k, v in regressor.state_dict().items():
                k_target = prefix + k
                if k.endswith("num_batches_tracked"):
                    continue
                # print("Copying from {0} to {1}, size {2}".format(k_target, k, v.size()))
                if k != "linear.weight":
                    v.copy_(src_state_dict[k_target])
                else:
                    # HACK to substitute linear layer with convolution
                    v.copy_(src_state_dict[k_target].view(-1, 64, 5, 5))
from src.contextual_roi_align import ContextualRoIAlign
from src.auxiliary_network import AuxiliaryNetwork
from src.channel_selector import OS2DChannelSelector
from src.losses import LCPLoss, GIoULoss

from os2d.modeling.model import Os2dModel

from src.contextual_roi_align import ContextualRoIAlign
from src.auxiliary_network import AuxiliaryNetwork
from src.channel_selector import OS2DChannelSelector
from src.losses import LCPLoss, GIoULoss

from os2d.modeling.model import Os2dModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class LCPPruner(Os2dModel):
    # 初始化與設置
    def __init__(self, logger, is_cuda=True, merge_branch_parameters=False, use_group_norm=False,
                 backbone_arch="resnet50", use_inverse_geom_model=True, simplify_affine=False,
                 img_normalization=None, alpha=0.6, beta=0.4, pruneratio=0.5):
        """
        初始化 LCP-OS2D 模型
        
        Args:
            logger: 日誌記錄器
            is_cuda (bool): 是否使用 CUDA
            merge_branch_parameters (bool): 是否合併分支參數
            use_group_norm (bool): 是否使用組標準化
            backbone_arch (str): 骨幹網絡架構
            use_inverse_geom_model (bool): 是否使用逆幾何模型
            simplify_affine (bool): 是否簡化仿射變換
            img_normalization (dict): 圖像標準化參數
            alpha (float): 重建誤差的權重係數
            beta (float): 輔助網絡損失的權重係數
            pruneratio (float): 剪枝比例
        """
        super(LCPPruner, self).__init__(
            logger=logger,
            is_cuda=is_cuda,
            merge_branch_parameters=merge_branch_parameters,
            use_group_norm=use_group_norm,
            backbone_arch=backbone_arch,
            use_inverse_geom_model=use_inverse_geom_model,
            simplify_affine=simplify_affine,
            img_normalization=img_normalization
        )
        
        # LCP 相關參數
        self.alpha = alpha
        self.beta = beta
        self.pruneratio = pruneratio
        
        # 初始化輔助網絡
        self.auxiliary_network = self._build_auxiliary_network()
        
        # 初始化通道選擇器
        self.channel_selector = self._init_channel_selector()
        
        # 初始化損失函數
        self.lcp_loss = LCPLoss(alpha=alpha, beta=beta)
        
        self.logger.info("Initialized LCPOs2dModel with pruning ratio: {}".format(pruneratio))

    def _build_auxiliary_network(self):
        """
        構建輔助網絡，包含 Contextual RoIAlign 層
        
        Returns:
            AuxiliaryNetwork: 輔助網絡實例
        """
        # 獲取特徵提取器的輸出通道數
        # ResNet feature extractor 的最後一層通常是 layer4，最後一個 BasicBlock 或 Bottleneck 的輸出通道數
        if hasattr(self.net_feature_maps, 'layer4'):
            in_channels = self.net_feature_maps.layer4[-1].conv3.out_channels if hasattr(self.net_feature_maps.layer4[-1], 'conv3') else self.net_feature_maps.layer4[-1].conv2.out_channels
        else:
            # 對於其他架構，可以根據具體情況調整
            in_channels = 2048  # ResNet50 默認輸出通道數
            
        # 創建輔助網絡，使用 Grozi 數據集的類別數 (1063)
        auxiliary_network = AuxiliaryNetwork(
            in_channels=in_channels,
            hidden_channels=128,
            num_classes=1063  # Grozi 數據集有 1063 個類別
        )
        
        if self.is_cuda:
            auxiliary_network = auxiliary_network.cuda()
            
        self.logger.info("Built auxiliary network with input channels: {} for Grozi dataset".format(in_channels))
        return auxiliary_network

    def _init_channel_selector(self):
        """
        初始化通道選擇器
        Returns:
            OS2DChannelSelector: 通道選擇器實例
        """
        channel_selector = OS2DChannelSelector(
            model=self.net_feature_maps,
            auxiliarynet=self.auxiliary_network,
            alpha=self.alpha,
            beta=self.beta
        )

        self.logger.info("Initialized channel selector with alpha: {}, beta: {}".format(
            self.alpha, self.beta))
        return channel_selector

    # 特徵提取與處理
    def get_features(self, layer_name, images):
        """
        獲取指定層的特徵
        
        Args:
            layer_name (str): 層名稱
            images (Tensor): 輸入圖像
        
        Returns:
            Tensor: 特徵圖
        """
        # 獲取目標層
        target_layer = self.net_feature_maps
        for part in layer_name.split('.'):
            if part.isdigit():
                target_layer = target_layer[int(part)]
            else:
                target_layer = getattr(target_layer, part)
                
        # 註冊鉤子函數
        features = []
        def hook_fn(module, input, output):
            features.append(output.detach())
        
        handle = target_layer.register_forward_hook(hook_fn)
        
        # 前向傳播
        with torch.no_grad():
            self.net_feature_maps(images)
            
        # 移除鉤子
        handle.remove()
        
        return features[0]
    
    def compute_reconstruction_error(self, original_features, pruned_features):
        """
        計算重建誤差
        
        Args:
            original_features (Tensor): 原始特徵圖
            pruned_features (Tensor): 剪枝後特徵圖
            
        Returns:
            Tensor: 重建誤差
        """
        # 計算特徵圖的總元素數量 Q = M × H × Y
        Q = original_features.size(0) * original_features.size(2) * original_features.size(3)
        
        # 計算歐氏距離（均方誤差）並除以 2Q
        # 這對應於論文中的公式 (12): Lre = (1/2Q) * ||F - X * Wc||^2
        return 1.0 / (2.0 * Q) * F.mse_loss(
            pruned_features, original_features, reduction='sum')
    
    def compute_joint_loss(self, reconstruction_error, auxiliary_loss):
        """
        計算聯合損失
        
        Args:
            reconstruction_error (Tensor): 重建誤差
            auxiliary_loss (Tensor): 輔助網絡損失
        
        Returns:
            Tensor: 聯合損失
        """
        return reconstruction_error + self.alpha * auxiliary_loss

    # 通道選擇與重要性計算
    def compute_channel_importance(self, layer_name, images, boxes, class_images=None):
        """
        計算通道重要性
        
        Args:
            layer_name (str): 層名稱
            images (Tensor): 輸入圖像
            boxes (Tensor): 邊界框
            class_images (Tensor, optional): 類別圖像

        Returns:
            Tensor: 通道重要性分數
        """
        # 確保類別圖像有正確的形狀 (batch_size, channels, height, width)
        if class_images is not None:
            # Ensure class_images has 3 dimensions [channels, height, width]
            if len(class_images.shape) == 4:  # If it's [batch, channels, height, width]
                if class_images.size(0) == 1:  # If batch size is 1
                    class_images = class_images.squeeze(0)  # Convert to [channels, height, width]
                else:
                    self.logger.warning(f"批量大小為 {class_images.size(0)} > 1，只使用第一個樣本")
                    class_images = class_images[0]  # Use only the first sample
            
        # self.logger.info(f"類別圖像形狀: {class_images.shape}，圖像形狀: {images.shape}")
        # 使用 channel_selector 來計算重要性
        importance = self.channel_selector.compute_importance(
            layer_name, 
            images, 
            boxes,
            class_images=class_images
        )
        
        return importance

    def select_channels(self, layer_name, images, boxes, percentage=0.5, class_images=None):
        """
        選擇重要通道
        
        Args:
            layer_name (str): 層名稱
            images (Tensor): 輸入圖像
            boxes (Tensor): 邊界框
            percentage (float): 要保留的通道比例
            
        Returns:
            list: 選擇的通道索引
        """
        # 計算通道重要性
        importance = self.compute_importance(layer_name, images, boxes, class_images=class_images)
        
        # 根據重要性排序
        _, indices = torch.sort(importance, descending=True)
        
        # 選擇前N個通道
        target_layer = self.net_feature_maps
        for part in layer_name.split('.'):
            if part.isdigit():
                target_layer = target_layer[int(part)]
            else:
                target_layer = getattr(target_layer, part)
                
        num_channels = target_layer.out_channels
        num_to_keep = int(num_channels * percentage)
        selected_indices = indices[:num_to_keep].tolist()
        
        return selected_indices

    # 剪枝操作
    def should_skip_layer(self, layer_name):
        """
        判斷是否應該跳過某一層
        
        Args:
            layer_name (str): 層名稱
            
        Returns:
            bool: 是否應該跳過
        """
        # 跳過第一層卷積（輸入層）
        if layer_name == "conv1":
            return True
        
        # 跳過輸出層
        if "fc" in layer_name or "classifier" in layer_name or "pred" in layer_name:
            return True
        
        # 跳過每個殘差塊的最後一個卷積層
        if "conv3" in layer_name or (layer_name.endswith(".2") and "conv" in layer_name):
            return True
            
        # 跳過下採樣層
        if "downsample" in layer_name:
            return True
        
        # 獲取層
        try:
            layer = self.net_feature_maps
            for part in layer_name.split('.'):
                if part.isdigit():
                    part_idx = int(part)
                    # Check if the layer has a __getitem__ method (like list or nn.Sequential)
                    if hasattr(layer, '__getitem__'):
                        layer = layer[part_idx]
                    else:
                        # For Bottleneck objects that don't support indexing
                        # Try to get the conv layer directly
                        layer = getattr(layer, f"conv{part_idx+1}", None)
                        if layer is None:
                            return True  # Skip if we can't find the layer
                else:
                    layer = getattr(layer, part)
                    
            # 跳過輸入通道為3的層（通常是第一層）
            if isinstance(layer, nn.Conv2d) and layer.in_channels == 3:
                return True
        except (AttributeError, IndexError, TypeError):
            # 如果層不存在，則跳過
            return True
                
        return False

    def prune_layer(self, layer_name, images, boxes, pruneratio=0.3, class_images=None):
        if self.should_skip_layer(layer_name):
            self.logger.info(f"跳過層: {layer_name}")
            return self

        self.logger.info(f"計算層 {layer_name} 的通道重要性")
        importance = self.channel_selector.compute_importance(layer_name, images, boxes, class_images=class_images)
        self.logger.info(f"剪枝前參數量 {sum(p.numel() for p in self.net_feature_maps.parameters())}")

        # 1. 找到目標層與其父模組
        target_layer = self.net_feature_maps
        parent = None
        last_part = None
        for part in layer_name.split('.'):
            parent = target_layer
            last_part = part
            if part.isdigit():
                target_layer = target_layer[int(part)]
            else:
                target_layer = getattr(target_layer, part)

        num_channels = target_layer.out_channels
        num_to_keep = int(num_channels * (1 - pruneratio))
        _, indices = torch.topk(importance, num_to_keep)
        indices, _ = torch.sort(indices)

        # 2. 重建新 Conv2d 層
        from os2d.utils.utils import prune_conv_layer, prune_batchnorm_layer, prune_conv_in_channels
        new_conv = prune_conv_layer(target_layer, indices.tolist())

        # 3. 掛回 Conv2d
        if isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent[int(last_part)] = new_conv
        else:
            setattr(parent, last_part, new_conv)

        # 4. 同步剪 BatchNorm2d（假設 conv1->bn1, conv2->bn2, conv3->bn3）
        if "conv" in last_part:
            bn_part = last_part.replace("conv", "bn")
            if hasattr(parent, bn_part):
                bn_layer = getattr(parent, bn_part)
                new_bn = prune_batchnorm_layer(bn_layer, indices.tolist())
                setattr(parent, bn_part, new_bn)

        # 5. 同步剪下一層 Conv2d 的 in_channels
        # 假設下一層是同一個 block 的 conv2/conv3 或下一個 block 的 conv1
        # 這裡以同一個 block 的下一層為例
        next_conv_name = None
        if last_part == "conv1" and hasattr(parent, "conv2"):
            next_conv_name = "conv2"
        elif last_part == "conv2" and hasattr(parent, "conv3"):
            next_conv_name = "conv3"
        # 你可根據實際模型結構擴展
        if next_conv_name and hasattr(parent, next_conv_name):
            next_conv = getattr(parent, next_conv_name)
            new_next_conv = prune_conv_in_channels(next_conv, indices.tolist())
            setattr(parent, next_conv_name, new_next_conv)

        # 6. 剪枝後重新獲取該層
        target_layer_new = self.net_feature_maps
        for part in layer_name.split('.'):
            if part.isdigit():
                target_layer_new = target_layer_new[int(part)]
            else:
                target_layer_new = getattr(target_layer_new, part)
        self.logger.info(f"已剪枝層: {layer_name}，保留 {target_layer_new.out_channels}/{num_channels} 個通道（參數數量已減少）")
        self.logger.info(f"剪枝後參數量 {sum(p.numel() for p in self.net_feature_maps.parameters())}")
        return self


    def multi_layer_pruning(self, layer_names, images, boxes, pruneratio=0.3, class_images=None):
        """
        剪枝多個層
        
        Args:
            layer_names (list): 層名稱列表
            images (Tensor): 輸入圖像
            boxes (Tensor): 邊界框
            pruneratio (float): 剪枝比例，默認為0.3
            
        Returns:
            self: 剪枝後的模型
        """
        if pruneratio is None:
            pruneratio = self.pruneratio
            
        for layer_name in layer_names:
            self.prune_layer(layer_name, images, boxes, pruneratio, class_images=class_images)
            
        return self

    # 殘差連接處理
    def identify_residual_blocks(self):
        """
        識別模型中的所有殘差塊
        
        Returns:
            dict: 殘差塊字典，鍵為塊名稱，值為塊對象
        """
        residual_blocks = {}
        
        # 遍歷模型的所有模塊
        for name, module in self.net_feature_maps.named_modules():
            # 檢查是否為殘差塊（根據ResNet的結構）
            # 在ResNet中，殘差塊通常以 'layer' 開頭，且名稱格式為 'layerX.Y'
            if 'layer' in name and len(name.split('.')) == 2:
                residual_blocks[name] = module
                
        self.logger.info(f"找到 {len(residual_blocks)} 個殘差塊")
        return residual_blocks

    def group_residual_blocks_by_stage(self):
        """
        將殘差塊按階段分組，確保同一階段的殘差塊通道數一致
        
        Returns:
            dict: 按階段分組的殘差塊，鍵為階段名稱，值為該階段的殘差塊列表
        """
        residual_blocks = self.identify_residual_blocks()
        stages = {}
        
        for name, block in residual_blocks.items():
            stage_name = name.split('.')[0]
            if stage_name not in stages:
                stages[stage_name] = []
            stages[stage_name].append(name)
        
        self.logger.info(f"將殘差塊分組為 {len(stages)} 個階段")
        return stages

    def compute_kl_divergence(self, original_features, pruned_features):
        """
        計算原始特徵與剪枝後特徵的KL散度
        
        Args:
            original_features (Tensor): 原始特徵
            pruned_features (Tensor): 剪枝後特徵
            
        Returns:
            Tensor: KL散度
        """
        # 將特徵轉換為概率分佈
        original_probs = torch.nn.functional.softmax(original_features.view(-1), dim=0)
        pruned_probs = torch.nn.functional.softmax(pruned_features.view(-1), dim=0)
        
        # 計算KL散度
        kl_div = torch.nn.functional.kl_div(
            pruned_probs.log(), original_probs, reduction='sum')
        
        return kl_div

    def prune_residual_connection(self, stage_name, images, boxes, pruneratio=None, class_images=None):
        """
        剪枝殘差連接，同時處理內部和外部通道
        """
        if pruneratio is None:
            pruneratio = self.pruneratio
            
        # 獲取該階段的所有殘差塊
        stages = self.group_residual_blocks_by_stage()
        if stage_name not in stages:
            self.logger.warning("Stage {} not found in model".format(stage_name))
            return self
            
        residual_blocks = stages[stage_name]
        self.logger.info("Pruning residual connections in stage: {}".format(stage_name))
        
        # 計算該階段所有殘差塊的重要性
        all_importance = {}
        
        # 首先獲取所有層的通道數
        channel_sizes = {}
        for block_name in residual_blocks:
            block_idx = int(block_name.split('.')[1])
            block = getattr(self.net_feature_maps, stage_name)[block_idx]
            
            for layer_idx, layer in enumerate([block.conv1, block.conv2, block.conv3]):
                layer_name = "{}.{}.conv{}".format(stage_name, block_idx, layer_idx + 1)
                channel_size = layer.out_channels
                if channel_size not in channel_sizes:
                    channel_sizes[channel_size] = []
                channel_sizes[channel_size].append((layer_name, layer))
        
        # 為每種通道數計算重要性
        for channel_size, layers in channel_sizes.items():
            importance_scores = []
            for layer_name, _ in layers:
                # 使用記憶體管理計算重要性
                try:
                    importance = self.compute_channel_importance(layer_name, images, boxes, class_images=class_images)
                    importance_scores.append(importance)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        # 清理快取並重試
                        torch.cuda.empty_cache()
                        try:
                            importance = self.compute_channel_importance(layer_name, images, boxes, class_images=class_images)
                            importance_scores.append(importance)
                        except RuntimeError:
                            # 移至 CPU 計算
                            cpu_images = images.cpu()
                            cpu_boxes = boxes.cpu() if isinstance(boxes, torch.Tensor) else [b.cpu() for b in boxes if b is not None]
                            cpu_class_images = class_images.cpu() if class_images is not None else None
                            
                            importance = self.compute_channel_importance(layer_name, cpu_images, cpu_boxes, class_images=cpu_class_images)
                            importance_scores.append(importance)
                    else:
                        raise
            
            # 計算平均重要性
            avg_importance = torch.zeros_like(importance_scores[0])
            for imp in importance_scores:
                avg_importance += imp
            avg_importance /= len(importance_scores)
            
            # 選擇重要通道
            num_to_keep = int(channel_size * (1 - pruneratio))
            _, indices = torch.topk(avg_importance, num_to_keep)
            indices, _ = torch.sort(indices)
            
            # 創建掩碼
            mask = torch.zeros(channel_size, device=images.device)
            mask[indices] = 1
            
            # 保存掩碼
            all_importance[channel_size] = mask
        
        # 應用掩碼到該階段所有殘差塊的所有層
        for block_name in residual_blocks:
            block_idx = int(block_name.split('.')[1])
            block = getattr(self.net_feature_maps, stage_name)[block_idx]
            
            # 應用掩碼到殘差塊內部各層
            for layer_name, layer in [
                ("conv1", block.conv1),
                ("conv2", block.conv2),
                ("conv3", block.conv3)
            ]:
                channel_size = layer.out_channels
                if channel_size in all_importance:
                    mask = all_importance[channel_size]
                    layer.weight.data = layer.weight.data * mask.view(-1, 1, 1, 1)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        layer.bias.data = layer.bias.data * mask
                else:
                    self.logger.warning(f"No mask found for layer {layer_name} with {channel_size} channels")
            
            # 如果有下採樣層，也應用掩碼
            if hasattr(block, 'downsample') and block.downsample is not None:
                downsample_layer = block.downsample[0]
                channel_size = downsample_layer.out_channels
                if channel_size in all_importance:
                    mask = all_importance[channel_size]
                    downsample_layer.weight.data = downsample_layer.weight.data * mask.view(-1, 1, 1, 1)
                    if hasattr(downsample_layer, 'bias') and downsample_layer.bias is not None:
                        downsample_layer.bias.data = downsample_layer.bias.data * mask
        
        return self

    # 依賴關係處理
    def build_dependency_graph(self):
        """
        構建層依賴圖，包括殘差連接的依賴關係
        
        Returns:
            dict: 層依賴圖
        """
        dependency_graph = {}
        
        # 遍歷模型的所有模塊
        for name, module in self.net_feature_maps.named_modules():
            if isinstance(module, nn.Conv2d):
                dependency_graph[name] = []
                
                # 尋找依賴於此層的其他層
                for dep_name, dep_module in self.net_feature_maps.named_modules():
                    if isinstance(dep_module, nn.Conv2d) and dep_name != name:
                        # 檢查輸入通道是否匹配此層的輸出通道
                        if dep_module.in_channels == module.out_channels:
                            dependency_graph[name].append(dep_name)
        
        # 添加殘差連接的依賴關係
        stages = self.group_residual_blocks_by_stage()
        for stage_name, blocks in stages.items():
            for i in range(len(blocks) - 1):
                current_block = blocks[i]
                next_block = blocks[i + 1]
                
                # 添加當前塊的最後一層到下一個塊的第一層的依賴
                current_last_layer = "{}.conv3".format(current_block)
                next_first_layer = "{}.conv1".format(next_block)
                
                if current_last_layer in dependency_graph:
                    dependency_graph[current_last_layer].append(next_first_layer)
        
        # 輸出依賴圖的日誌
        # self.logger.info("Layer dependency graph constructed")
        # for layer, deps in dependency_graph.items():
        #     if deps:
        #         self.logger.info(f"Layer {layer} has dependencies: {deps}")
        
        return dependency_graph

    def find_residual_dependencies(self, layer_name):
        """
        找出與指定層有殘差連接的所有層
        
        Args:
            layer_name (str): 層名稱
            
        Returns:
            list: 與指定層有殘差連接的所有層名稱
        """
        # 解析層名稱
        parts = layer_name.split('.')
        if len(parts) < 3 or (parts[0] != 'layer1' and parts[0] != 'layer2' and parts[0] != 'layer3'):
            self.logger.info(f"Invalid layer name: {layer_name}")
            return []
            
        stage_name = parts[0]
        block_idx = int(parts[1])
        
        # 獲取該階段的所有殘差塊
        stage = getattr(self.net_feature_maps, stage_name)
        num_blocks = len(stage)
        
        dependencies = []
        
        # 檢查是否為殘差塊的第一層或最後一層
        if parts[2] == 'conv1':
            # 第一層依賴於前一個塊的最後一層
            if block_idx > 0:
                dependencies.append("{}.{}.conv3".format(stage_name, block_idx - 1))
                
        elif parts[2] == 'conv3':
            # 最後一層被下一個塊的第一層依賴
            if block_idx < num_blocks - 1:
                dependencies.append("{}.{}.conv1".format(stage_name, block_idx + 1))
                
        # 檢查下採樣層
        block = stage[block_idx]
        if hasattr(block, 'downsample') and block.downsample is not None:
            dependencies.append("{}.{}.downsample.0".format(stage_name, block_idx))
            
        return dependencies

    # 整體剪枝流程
    def cross_block_pruning(self, images, boxes, pruneratio=None, class_images=None):
        """
        跨塊剪枝，考慮殘差連接
        
        Args:
            images (Tensor): 輸入圖像
            boxes (Tensor): 邊界框
            pruneratio (float, optional): 剪枝比例，如果為None則使用self.pruneratio
            
        Returns:
            self: 剪枝後的模型
        """
        if pruneratio is None:
            pruneratio = self.pruneratio
            
        # 構建依賴圖
        dependency_graph = self.build_dependency_graph()
        
        # 按照依賴關係順序剪枝
        visited = set()
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            
            # 先處理依賴於此節點的所有節點
            for dep in dependency_graph.get(node, []):
                dfs(dep)
                
            # 然後剪枝此節點
            if not self.should_skip_layer(node):
                self.prune_layer(node, images, boxes, pruneratio, class_images=class_images)
                
        # 從所有沒有入邊的節點開始
        for node in dependency_graph:
            has_incoming = False
            for deps in dependency_graph.values():
                if node in deps:
                    has_incoming = True
                    break
                    
            if not has_incoming:
                dfs(node)
                    
        # 處理殘差連接
        stages = self.group_residual_blocks_by_stage()
        for stage_name in stages:
            self.prune_residual_connection(stage_name, images, boxes, pruneratio, class_images=class_images)
                
        return self

    def prune_model(self, dataloader, pruneratio=0.3, batch_indices=None, max_batches=3):
        """
        剪枝整個模型，實現 Algorithm 1
        
        Args:
            dataloader (DataLoader): 數據加載器
            pruneratio (float, optional): 剪枝比例，如果為None則使用self.pruneratio
            batch_indices (list, optional): 要使用的批次索引列表，如果為None則自動選擇
            max_batches (int, optional): 最大使用批次數量
            
        Returns:
            self: 剪枝後的模型
        """
        if pruneratio is None:
            pruneratio = self.pruneratio
            
        self.logger.info(f"開始模型剪枝，剪枝比例: {pruneratio}")
        self.logger.info(f"剪枝前的模型參數量: {sum(p.numel() for p in self.net_feature_maps.parameters())}")
        
        # 確定要使用的批次索引
        if batch_indices is None:
            total_batches = len(dataloader)
            batch_indices = list(range(min(total_batches, max_batches)))
            self.logger.info(f"自動選擇 {len(batch_indices)} 個批次進行剪枝: {batch_indices}")
        
        for batch_idx in batch_indices:
            try:
                self.logger.info(f"處理批次 {batch_idx}/{len(batch_indices)-1}")
                # 獲取指定批次的數據
                batch = dataloader.get_batch(batch_idx)
                
                # 解析批次數據
                images = batch[0]  # 輸入圖像，形狀為 [batch_size, channels, height, width]
                boxes = batch[1] if len(batch) > 1 else []  # 邊界框列表
                
                # 提取類別圖像
                class_images = None
                if len(batch) > 2:
                    class_images = batch[2]  # 獲取類別圖像
                    # self.logger.info(f"批次 {batch_idx} 類別圖像形狀: {class_images.shape}")
                    
                    # 處理5D格式: [batch_size, num_classes, channels, height, width]
                    if class_images.ndim == 5:
                        b, c, ch, h, w = class_images.shape
                        class_images = class_images.reshape(b*c, ch, h, w)
                        
                    # 處理4D格式: [batch_size, channels, height, width]
                    elif class_images.ndim == 4:
                        # 選擇第一個樣本，確保它是3D [channels, height, width]
                        class_images = class_images[0]
                    
                    # 確保class_images是3D [channels, height, width]
                    if class_images.ndim != 3:
                        self.logger.warning(f"將class_images從{class_images.shape}重塑為3D格式")
                        if class_images.ndim == 4:
                            class_images = class_images[0]
                else:
                    # 如果沒有類別圖像，則生成合成的類別圖像
                    channels = 3
                    class_images = torch.randn(channels, 64, 64)
                    self.logger.info(f"生成合成類別圖像，形狀: {class_images.shape}")

                self.logger.info(f"批次 {batch_idx} 最終類別圖像形狀: {class_images.shape}")
                
                # 如果需要，將數據移至CUDA
                if self.is_cuda:
                    images = images.cuda()
                    boxes = [b.cuda() if b is not None else None for b in boxes]
                    if class_images is not None:
                        class_images = class_images.cuda()
                
                # 使用記憶體管理進行跨塊剪枝
                try:
                    self.cross_block_pruning(images, boxes, pruneratio, class_images=class_images)
                    self.logger.info(f"批次 {batch_idx} 剪枝成功")
                    # 如果成功剪枝，可以提前退出循環
                    break
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        self.logger.warning(f"批次 {batch_idx} CUDA記憶體不足，嘗試使用CPU進行剪枝")
                        # 將數據移至CPU
                        images = images.cpu()
                        boxes = [b.cpu() if b is not None else None for b in boxes]
                        if class_images is not None:
                            class_images = class_images.cpu()
                        
                        # 在CPU上執行剪枝
                        try:
                            self.cross_block_pruning(images, boxes, pruneratio, class_images=class_images)
                            self.logger.info(f"批次 {batch_idx} 在CPU上剪枝成功")
                            # 如果成功剪枝，可以提前退出循環
                            break
                        except Exception as cpu_e:
                            self.logger.error(f"批次 {batch_idx} 在CPU上剪枝失敗: {cpu_e}")
                            # 繼續嘗試下一個批次
                            continue
                    else:
                        self.logger.error(f"批次 {batch_idx} 剪枝失敗: {e}")
                        # 繼續嘗試下一個批次
                        continue
            except Exception as e:
                self.logger.error(f"處理批次 {batch_idx} 時發生錯誤: {e}")
                # 繼續嘗試下一個批次
                continue
        
        self.logger.info(f"剪枝完成，保留比例: {1 - pruneratio}")
        self.logger.info(f"剪枝後的模型參數量: {sum(p.numel() for p in self.net_feature_maps.parameters())}")
        return self
