import sys
import traceback
import torchvision
from test.test_os2d_model_in_prune import *
from test.test_contextual_roi_align import *
from test.test_auxiliary_network import *
from test.test_lcp_channel_selector import *
# from test.test_os2d_model_prune import *
# from test.test_residual_connection import *
def test_empty():
    return True

def run_all_tests():
    tests = [
        # ("Grozi Dataset 測試", test_grozi_dataset),
        # ("OS2D Dataset 測試", test_os2d_dataset),
        # ("OS2D Dataloader 測試", test_os2d_dataloader),

        # ( "OS2D 模型初始化測試", test_os2d_model_in_prune_initialization),
        # ( "OS2D Set Channel 測試", test_set_layer_out_channels),
        # ( "OS2D 模型前向傳播測試", test_os2d_model_in_prune_forward),
        
        # ("OS2D 模型測試", test_load_os2d_weights),
        # ("OS2D 特徵圖測試", test_get_feature_map),
        # ("OS2D 殘差連接測試", test_residual_connection_protection),
        # ("OS2D 殘差連接測試 CROSS BLOCK", test_cross_block_residual_connection),
        # ("OS2D prune channel 測試",test_prune_channel),
        # ("只剪 conv1", test_prune_conv1_only),
        # ("只剪 conv2", test_prune_conv2_only),
        # ("只剪 conv3", test_prune_conv3_only),
        # ("連續剪枝測試" , test_continuous_block_pruning),
        # ("連續剪多個 block", test_prune_multiple_blocks),
        # ("跨 stage 剪枝", test_cross_stage_prune),
        # ("ResNet18/34 BasicBlock 剪枝", test_resnet18_basicblock_prune),
        # ("剪枝率 sweep 測試 layer2.0.conv1", lambda: test_pruning_ratios("layer2.0.conv1")),
        # ("剪枝率 sweep 測試 layer2.0.conv2", lambda: test_pruning_ratios("layer2.0.conv2")),
        # ("剪枝率 sweep 測試 layer3.0.conv1", lambda: test_pruning_ratios("layer3.0.conv1")),
        
        # ("AuxiliaryNetwork 測試", test_initialization),
        # ("AuxiliaryNetwork 前向傳播測試", test_forward_pass),
        # ("AuxiliaryNetwork 更新通道測試", test_update_input_channels),
        
        # ("ContextualRoIAlign 測試1", test_forward),
        # ("ContextualRoIAlign 測試2", test_empty_boxes),
        # ("ContextualRoIAlign All Case 測試" , test_contextual_roi_align_all_cases ),

        # ("LCP 通道選擇器測試", test_lcp_channel_selector),
        # ("LCP 通道選擇器特徵圖測試", test_feature_map_extraction ),
        # ("通道重要性計算測試", test_channel_importance_computation),
        # ("Loss 測試", test_compute_losses),
        ("訓練函數 basic 測試" , test_train_one_epoch_basic ),
        # ("Eval 測試" , test_os2d_model_in_prune_eval),
        # ("Save Checkpoint 測試", test_save_checkpoint),
        # ("OS2D 框架相容性測試", test_os2d_compatibility_with_pruned_model),
        # ("Finetune 測試", test_lcp_finetune_pipeline ),
        # ("完整 LCP 剪枝 pippline+train 測試", test_lcp_prune_and_train_pipeline),
        # ("完整 pipeline 測試", test_full_lcp_pipeline_with_eval_and_checkpoint ),
    ]

    results = []
    print("\n======================= LCP/DCP 單元測試開始 =======================")
    for idx, (name, test_func) in enumerate(tests):
        print(f"\n===== [{idx+1}/{len(tests)}] 執行 {name} =====")
        try:
            success = test_func()
        except Exception as e:
            print(f"❌ {name} 發生例外: {e}")
            traceback.print_exc()
            success = False
        results.append((name, success))
        print(f"===== {name}: {'✅ 通過' if success else '❌ 失敗'} =====\n")
        import gc
        gc.collect()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        if success == False:
            print("⚠️ 測試失敗，請檢查上方錯誤訊息。")
            return
    print("\n======================= 測試結果摘要 =======================")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    for name, success in results:
        print(f"{name}: {'✅ 通過' if success else '❌ 失敗'}")
    print(f"\n總計: {passed}/{total} 通過 ({passed/total*100:.1f}%)")
    if passed == total:
        print("🎉🎉🎉 全部測試通過！")
    else:
        print("⚠️ 有部分測試未通過，請檢查上方錯誤訊息。")

if __name__ == "__main__":
    run_all_tests()
