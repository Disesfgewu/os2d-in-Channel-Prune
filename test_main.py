import sys
import traceback
import torchvision
from test.test_dataset import *
from test.test_os2d_model_in_prune_with_func import *
from test.test_contextual_roi_align import *
from test.test_auxiliary_network import *
from test.test_lcp_channel_selector import *
from test.test_os2d_model_in_prune import *
# from test.test_residual_connection import *
def test_empty():
    return True

def run_all_tests():
    test_in_dataset = [
        ("Grozi Dataset 測試", test_grozi_dataset),
        ("OS2D Dataset 測試", test_os2d_dataset),
        ("OS2D Dataloader 測試", test_os2d_dataloader),
    ]
    test_in_auxiliary_network = [
        ("AuxiliaryNetwork 測試", test_initialization),
        ("AuxiliaryNetwork 前向傳播測試", test_forward_pass),
        ("AuxiliaryNetwork 更新通道測試", test_update_input_channels),
    ]
    test_in_contextual_roi_ailgn = [
        ("ContextualRoIAlign 測試1", test_forward),
        ("ContextualRoIAlign 測試2", test_empty_boxes),
        ("ContextualRoIAlign All Case 測試" , test_contextual_roi_align_all_cases ),
    ]
    test_in_lcp_channel_selector = [
        ("LCP 通道選擇器測試", test_lcp_channel_selector),
        ("LCP 通道選擇器特徵圖測試", test_feature_map_extraction ),
    ]
    test_in_os2d_model_in_prune = [
        ("Init 測試" , test_in_init ),
        ("Set Layer Out Channels 測試" , test_set_layer_out_channels ),
        ("Should Skip 測試" , test_should_skip_pruning ),
        ("Handle Residual 測試" , test_in_handle_residual_connection ),
        ("Prune Conv Layer 測試" , test_in_prune_conv_layer ),
        ("Reset BatchNorm 測試" , test_in_reset_batchnorm_stats ),
        ("Prune Channel 測試" , test_in_prune_channel ),
        ("Prune Model 測試" , test_in_prune_model ),
        ("Visualize model Architecture 測試" , test_in_visualize_model_architecture ),
        ("Get Feature Map 測試" , test_in_get_feature_map ),
        ("Forward 測試" , test_in_forward ),
        ("Print Model Summary 測試" , test_in_print_model_summary ),
        ("Normalize Batch Images 測試" , test_in_normalize_batch_images ),
        ("Cat Boxes List 測試" , test_in_cat_boxes_list ),
        ("Analyze OS2D Output 測試" , test_in_analyze_os2d_output ),
        ("Convert 4D to 2D Scores 測試" , test_in_convert_4d_to_2d_scores ),
        ("Prepare Classification Targets 測試" , test_in_prepare_classification_targets ),
        ("Prepare Target Boxes 測試" , test_in_prepare_target_boxes ),
        ("Convert Dense Boxes to Standard 測試" , test_in_convert_dense_boxes_to_standard ),
        ("Match Box Counts 測試" , test_in_match_box_counts ),
        ("Select Boxes By Confidence 測試" , test_in_select_boxes_by_confidence ),
        ("Align Score Dimensions 測試" , test_in_align_score_dimensions ),
        ("Compute Classification Distillation Loss 測試" , test_in_compute_classification_distillation_loss ),
        ("Compute 4D Classification Distillation Loss 測試" , test_in_compute_4d_classification_distillation_loss ),
        ("Select Top Feature Positions 測試" , test_in_select_top_feature_positions ),
        ("Compute Box Distillation Loss 測試" , test_in_compute_box_distillation_loss ),
        ("Standardize Boxes 測試" , test_in_standardize_boxes ),
        ("Compute Classification Loss 測試" , test_in_compute_classification_loss ),
        ("Compute Box Regression Loss 測試" , test_in_compute_box_regression_loss ),
        ("Compute Teacher Distillation Loss 測試" , test_in_compute_teacher_distillation_loss ),
        ("Compute LCP Loss 測試" , test_in_compute_lcp_loss ),
        ("Get Feature Maps for LCP 測試" , test_in_get_feature_maps_for_lcp ),
        ("Get Boxes for LCP 測試" , test_in_get_boxes_for_lcp ),
        ("Prepare ROI Format Boxes 測試" , test_in_prepare_roi_format_boxes ),
        ("Enhance LCP Loss 測試" , test_in_enhance_lcp_loss ),
        ("Prepare GT Labels for LCP 測試" , test_in_prepare_gt_labels_for_lcp ),
        ("Scale Losses 測試" , test_in_scale_losses ),
        ("Compute Losses 測試" , test_in_compute_losses ),
        ("Update Classifier for Classes 測試" , test_in_update_classifier_for_classes ),
        ("Update Auxiliary Classifier 測試" , test_in_update_auxiliary_classifier ),
        ("Print Training Config 測試" , test_in_print_training_config ),
        ("Update Auxiliary Channels 測試" , test_in_update_auxiliarty_channels ),
        ("Run Standard Inference 測試" , test_in_run_standard_inference ),
        ("Run Teacher Feature Pyramid 測試" , test_in_run_teacher_feature_pyramid ),
        ("Run Student Feature Pyramid 測試" , test_in_run_student_feature_pyramid ),
        ("Scale Inputs 測試" , test_in_scale_inputs ),
        ("Merge Dense Outputs 測試" , test_in_merge_dense_outputs ),
        ("Merge Standard Outputs 測試" , test_in_merge_standard_outputs ),
        ("Apply NMS to Outputs 測試" , test_in_apply_nms_to_outputs ),
        ("Analyze Model Outputs 測試" , test_in_analyze_model_outputs ),
        ("Prepare Outputs Dict 測試" , test_in_prepare_outputs_dict ),
        ("Print Loss Info 測試" , test_in_print_loss_info ),
        ("Clip Gradients 測試" , test_in_clip_gradients ),
        ("Print Batch Summary 測試" , test_in_print_batch_summary ),
        ("Print Training Summary 測試" , test_in_print_training_summary ),
        ("Train One Epoch 測試" , test_in_train_one_epoch ),
    ]
    test_in_od2d_model_in_prune_with_func = [
        ( "OS2D 模型初始化測試", test_os2d_model_in_prune_initialization),
        ( "OS2D Set Channel 測試", test_set_layer_out_channels),
        ( "OS2D 模型前向傳播測試", test_os2d_model_in_prune_forward),
        
        ("OS2D 模型測試", test_load_os2d_weights),
        ("OS2D 特徵圖測試", test_get_feature_map),
        ("OS2D 殘差連接測試", test_residual_connection_protection),
        ("OS2D 殘差連接測試 CROSS BLOCK", test_cross_block_residual_connection),
        ("OS2D prune channel 測試",test_prune_channel),
        ("只剪 conv1", test_prune_conv1_only),
        ("只剪 conv2", test_prune_conv2_only),
        ("只剪 conv3", test_prune_conv3_only),
        ("連續剪枝測試" , test_continuous_block_pruning),
        ("連續剪多個 block", test_prune_multiple_blocks),
        ("跨 stage 剪枝", test_cross_stage_prune),
        ("ResNet18/34 BasicBlock 剪枝", test_resnet18_basicblock_prune),
        ("剪枝率 sweep 測試 layer2.0.conv1", lambda: test_pruning_ratios("layer2.0.conv1")),
        ("剪枝率 sweep 測試 layer2.0.conv2", lambda: test_pruning_ratios("layer2.0.conv2")),
        ("剪枝率 sweep 測試 layer3.0.conv1", lambda: test_pruning_ratios("layer3.0.conv1")),

        ("通道重要性計算測試", test_channel_importance_computation),
        ("Loss 測試", test_compute_losses),
        ("訓練函數 basic 測試" , test_train_one_epoch_basic ),
        ("訓練函數 多個 epoch 測試" , test_train_one_epoch_multiple_epochs ),
        # ("Eval 測試" , test_os2d_model_in_prune_eval),
        # ("Save Checkpoint 測試", test_save_checkpoint),
        # ("OS2D 框架相容性測試", test_os2d_compatibility_with_pruned_model),
        # ("Finetune 測試", test_lcp_finetune_pipeline ),
        # ("完整 LCP 剪枝 pippline+train 測試", test_lcp_prune_and_train_pipeline),
        # ("完整 pipeline 測試", test_full_lcp_pipeline_with_eval_and_checkpoint ),
    ]
    tests = [
        test_in_dataset,
        test_in_contextual_roi_ailgn,
        test_in_auxiliary_network,
        test_in_lcp_channel_selector,
        test_in_os2d_model_in_prune,
        test_in_od2d_model_in_prune_with_func
    ]
    
    results = []
    failed_tests = []
    print("\n======================= LCP/DCP 單元測試開始 =======================")
    for idx, (test_group) in enumerate(tests):
        group_name = f"Group {idx+1}"
        print(f"\n===== [{idx+1}/{len(tests)}] 執行 {group_name} =====")
        group_failed = False
        for _idx, (_name, _test_func) in enumerate(tests[idx]):
            print(f"\n===== [{_idx+1}/{len(tests[idx])}] 執行 {_name} =====")
            try:
                success = _test_func()
            except Exception as e:
                print(f"❌ {_name} 發生例外: {e}")
                traceback.print_exc()
                success = False
            
            results.append((_name, success))
            if not success:
                failed_tests.append((_name, group_name))
                group_failed = True
            
            print(f"===== {_name}: {'✅ 通過' if success else '❌ 失敗'} =====\n")
            import gc
            gc.collect()
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            if not success and not group_failed:
                print("⚠️ 測試失敗，請檢查上方錯誤訊息。")
                continue
        
        if not group_failed:
            print(f"===== {group_name}: ✅ 通過 =====\n")
        else:
            print(f"===== {group_name}: ❌ 部分測試失敗 =====\n")

    print("\n======================= 測試結果摘要 =======================")
    if not failed_tests:
        print("🎉🎉🎉 全部測試通過！")
    else:
        print(f"❌ 有 {len(failed_tests)} 個測試未通過:")
        for idx, (test_name, group_name) in enumerate(failed_tests):
            print(f"  {idx+1}. {test_name} (在 {group_name})")
        print("\n請檢查上述測試的錯誤訊息。")
if __name__ == "__main__":
    run_all_tests()
