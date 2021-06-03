from extract_features import extract_features, get_fine_tuned_model_base, get_train_test_dfs


fine_tuned_xception_base = get_fine_tuned_model_base(
    "data/xception_bpmll_fine_tuned.h5")

dfs = get_train_test_dfs()
subset_names = ["train", "test"]

for df, subset_name in zip(dfs, subset_names):
    extract_features(
        fine_tuned_xception_base,
        "xception_bpmll_fine_tuned",
        df,
        subset_name)
