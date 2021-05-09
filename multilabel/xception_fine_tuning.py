from extract_features import get_train_test_dfs
from fine_tune import fine_tune_pretrained_model

from tensorflow.keras.applications import Xception


base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(299,299,3),
        pooling="max")

linear_probe_model_save_path = "data/xception_linear_probe.h5"
fine_tuned_model_save_path = "data/xception_fine_tuned.h5"

df_train, _ = get_train_test_dfs()

fine_tune_pretrained_model(
    base_model,
    df_train,
    linear_probe_model_save_path,
    fine_tuned_model_save_path)
