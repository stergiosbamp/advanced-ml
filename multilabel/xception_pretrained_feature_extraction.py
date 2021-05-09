from extract_features import get_train_test_dfs, extract_features

from tensorflow.keras.applications import Xception


xception_avgpool = Xception(
    include_top=False,
    weights="imagenet",
    input_shape=(299,299,3),
    pooling="avg")

xception_maxpool = Xception(
    include_top=False,
    weights="imagenet",
    input_shape=(299,299,3),
    pooling="max")

models = [
    xception_avgpool,
    xception_maxpool]

model_names = [
    "pretrained_xception_avgpool",
    "pretrained_xception_maxpool"]

dfs = get_train_test_dfs()

subset_names = ["train", "test"]

for model, model_name in zip(models, model_names):
    for df, subset_name in zip(dfs, subset_names):
        extract_features(model, model_name, df, subset_name)
