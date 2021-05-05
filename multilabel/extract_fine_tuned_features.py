from extract_features import extract_features, get_train_test_dfs

from tensorflow.keras import Model
from tensorflow.keras.models import load_model


def get_fine_tuned_model_base(fine_tuned_model_file):
    
    # Load model from .h5 file
    fine_tuned_model = load_model(fine_tuned_model_file)

    # Get model base
    x_in = fine_tuned_model.input
    x_out = fine_tuned_model.layers[-2](x_in)
    fine_tuned_model_base = Model(inputs=x_in, outputs=x_out)

    return fine_tuned_model_base


if __name__ == "__main__":

    dfs = get_train_test_dfs()
    subset_names = ["train", "test"]

    fine_tuned_xception_base = get_fine_tuned_model_base(
        "data/xception_linear_probe_fine_tuned.h5")
    
    for df, subset_name in zip(dfs, subset_names):
        extract_features(
            fine_tuned_xception_base,
            "xception_fine_tuned",
            df,
            subset_name)
