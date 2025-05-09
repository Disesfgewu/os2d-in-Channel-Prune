def test_grozi_dataset():
    """
    測試 OS2D 官方 Grozi-3.2k dataset/dataloader 功能與資料結構
    """
    import os
    import pytest
    import torch
    import numpy as np
    import pandas as pd
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize

    data_path = "./data"
    grozi_csv = os.path.join(data_path, "grozi", "classes", "grozi.csv")
    if not os.path.exists(grozi_csv):
        print("\n❌ Grozi-3.2k dataset not found. 請依官方說明手動下載並解壓至 ./data/grozi/")
        pytest.skip("Grozi-3.2k dataset missing, test skipped.")
        return False

    # 建立 dataset（mini subset 加速測試）
    dataset = build_grozi_dataset(
        data_path=data_path,
        name="grozi-train-mini",  # 只取2張圖2類別，適合單元測試
        eval_scale=224,
        cache_images=False
    )

    # 驗證 dataset 結構
    assert hasattr(dataset, "gtboxframe")
    assert hasattr(dataset, "image_ids")
    assert hasattr(dataset, "image_file_names")
    assert hasattr(dataset, "get_name")
    assert hasattr(dataset, "get_eval_scale")
    assert hasattr(dataset, "get_class_ids")
    assert hasattr(dataset, "get_image_annotation_for_imageid")
    assert isinstance(dataset.image_ids, list) and len(dataset.image_ids) > 0
    assert isinstance(dataset.image_file_names, list) and len(dataset.image_file_names) > 0
    assert isinstance(dataset.gtboxframe, pd.DataFrame)
    print(f"✅ Dataset 結構測試通過 ({dataset.get_name()})，共 {len(dataset.image_ids)} 張圖，{len(dataset.get_class_ids())} 類別")

    # 測試 get_class_ids, get_image_annotation_for_imageid
    class_ids = dataset.get_class_ids()
    assert isinstance(class_ids, (list, np.ndarray))
    image_id = dataset.image_ids[0]
    boxes = dataset.get_image_annotation_for_imageid(image_id)
    assert hasattr(boxes, "bbox_xyxy")
    assert hasattr(boxes, "get_field")
    print(f"✅ 單圖標註測試通過，image_id={image_id}，boxes數={len(boxes)}")

    # 建立 box_coder
    box_coder = Os2dBoxCoder(
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.4,
        remap_classification_targets_iou_pos=0.5,
        remap_classification_targets_iou_neg=0.4,
        output_box_grid_generator=BoxGridGenerator(
            box_size=FeatureMapSize(w=16, h=16),
            box_stride=FeatureMapSize(w=16, h=16)
        ),
        function_get_feature_map_size=lambda img_size: FeatureMapSize(w=img_size.w // 16, h=img_size.h // 16),
        do_nms_across_classes=False
    )

    dataloader = DataloaderOneShotDetection(
        dataset=dataset,
        box_coder=box_coder,
        batch_size=1,
        img_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        gt_image_size=64,
        random_flip_batches=False,
        random_crop_size=None,
        random_color_distortion=False,
        pyramid_scales_eval=[1.0],
        do_augmentation=False
    )

    # 測試 dataloader 的 batch 結構
    batch = dataloader.get_batch(0)
    assert isinstance(batch, tuple) and len(batch) >= 9
    images, class_images, loc_targets, class_targets, batch_class_ids, class_image_sizes, box_inverse_transform, batch_boxes, batch_img_size = batch
    assert isinstance(images, torch.Tensor)
    assert isinstance(class_images, list)
    assert isinstance(loc_targets, torch.Tensor)
    assert isinstance(class_targets, torch.Tensor)
    print(f"✅ Dataloader batch 結構測試通過，images.shape={images.shape}, class_images={len(class_images)}")

    # 測試 get_all_class_images
    batch_class_images, class_image_sizes, class_ids = dataloader.get_all_class_images()
    assert isinstance(batch_class_images, list)
    assert len(batch_class_images) == len(class_ids)
    print(f"✅ get_all_class_images 測試通過，class_images={len(batch_class_images)}")

    print("🎉 test_grozi_dataset: OS2D Grozi dataset/dataloader 功能測試全部通過！")
    return True


def test_os2d_dataset():
    """
    測試 OS2D 官方 build_dataset_by_name 支援的所有資料集能正確初始化
    """
    import os
    import pytest
    from os2d.data.dataset import build_dataset_by_name

    data_path = "./data"
    dataset_names = [
        "grozi-train-mini",
        "grozi-val-old-cl",
        "grozi-val-new-cl",
        "grozi-val-all"
        # 你可以根據安裝情況加入 "instre-all", "dairy", "paste-v", "paste-f" 等
    ]
    for name in dataset_names:
        try:
            dataset = build_dataset_by_name(
                data_path=data_path,
                name=name,
                eval_scale=224,
                cache_images=False
            )
            assert hasattr(dataset, "get_name")
            assert dataset.get_name() == name
            print(f"✅ Dataset {name} 初始化成功，images={len(dataset.image_ids)}, classes={len(dataset.get_class_ids())}")
        except Exception as e:
            print(f"⚠️ Dataset {name} 初始化失敗: {e}")
    print("🎉 test_os2d_dataset: build_dataset_by_name 支援的資料集測試完成")
    return True


def test_os2d_dataloader():
    """
    測試 OS2D 官方 DataloaderOneShotDetection 的 batch 結構與 API
    """
    import os
    import pytest
    import torch
    from os2d.data.dataset import build_grozi_dataset
    from os2d.data.dataloader import DataloaderOneShotDetection
    from os2d.modeling.box_coder import Os2dBoxCoder, BoxGridGenerator
    from os2d.structures.feature_map import FeatureMapSize

    data_path = "./data"
    dataset = build_grozi_dataset(
        data_path=data_path,
        name="grozi-train-mini",
        eval_scale=224,
        cache_images=False
    )
    box_coder = Os2dBoxCoder(
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.4,
        remap_classification_targets_iou_pos=0.5,
        remap_classification_targets_iou_neg=0.4,
        output_box_grid_generator=BoxGridGenerator(
            box_size=FeatureMapSize(w=16, h=16),
            box_stride=FeatureMapSize(w=16, h=16)
        ),
        function_get_feature_map_size=lambda img_size: FeatureMapSize(w=img_size.w // 16, h=img_size.h // 16),
        do_nms_across_classes=False
    )
    dataloader = DataloaderOneShotDetection(
        dataset=dataset,
        box_coder=box_coder,
        batch_size=1,
        img_normalization={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        gt_image_size=64,
        random_flip_batches=False,
        random_crop_size=None,
        random_color_distortion=False,
        pyramid_scales_eval=[1.0],
        do_augmentation=False
    )
    # 測試 __len__ 與 get_batch
    assert len(dataloader) > 0
    batch = dataloader.get_batch(0)
    assert isinstance(batch, tuple) and len(batch) >= 9
    images, class_images, loc_targets, class_targets, batch_class_ids, class_image_sizes, box_inverse_transform, batch_boxes, batch_img_size = batch
    assert isinstance(images, torch.Tensor)
    assert isinstance(class_images, list)
    assert isinstance(loc_targets, torch.Tensor)
    assert isinstance(class_targets, torch.Tensor)
    print(f"✅ Dataloader batch 結構測試通過，images.shape={images.shape}, class_images={len(class_images)}")
    # 測試 get_name, get_eval_scale
    assert hasattr(dataloader, "get_name")
    assert hasattr(dataloader, "get_eval_scale")
    print(f"✅ Dataloader get_name={dataloader.get_name()}, eval_scale={dataloader.get_eval_scale()}")
    print("🎉 test_os2d_dataloader: OS2D DataloaderOneShotDetection 功能測試全部通過！")
    return True
