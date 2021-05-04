import pickle

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_train_test_dfs(
    binarized_annotations_pkl="data/binarized_annotations.pkl",
    train_list_txt="data/train_val_list.txt",
    test_list_txt="data/test_list.txt"):

    # read all binarized annotations
    df_labels = pd.read_pickle(binarized_annotations_pkl)

    # read train filenames and merge corresponding annotations
    df_train = pd.read_csv(train_list_txt)
    df_train = df_train.merge(df_labels, how="left", on="filename")

    # read test filenames and merge corresponding annotations
    df_test = pd.read_csv(test_list_txt)
    df_test = df_test.merge(df_labels, how="left", on="filename")

    dfs = [df_train, df_test]
    return dfs


def extract_pretrained_features(
    model,
    model_name,
    df,
    subset_name,
    directory="data/chest_x_rays",
    x_col="filename",
    target_size=(299,299),
    batch_size=128,
    rescale=1./255.,
    dest_dir="data/extracted_features"):

    # just rescale pixel values, don't augment images
    datagen = ImageDataGenerator(rescale=rescale)

    # flow images from provided filenames
    generator = datagen.flow_from_dataframe(
        df,
        directory=directory,
        x_col=x_col,
        class_mode=None,
        shuffle=False,
        target_size=target_size,
        batch_size=batch_size)
    
    # extract features
    features = model.predict(generator, verbose=1)

    # get corresponding labels
    labels = np.stack(df["binarized_labels"].values, axis=0)

    # nest features and labels in dict
    data = {"X": features, "Y": labels}

    # pickle extracted features and labels
    filename = dest_dir + "/" + model_name + "_" + subset_name + ".pkl"
    with open(filename, "wb") as f:
            pickle.dump(data, f, protocol=4)


if __name__ == "__main__":

    from tensorflow.keras.applications import EfficientNetB3, EfficientNetB7, Xception

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

    efficientnetb3_avgpool = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(299,299,3),
        pooling="avg")

    efficientnetb3_maxpool = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(299,299,3),
        pooling="max")

    efficientnetb7_avgpool = EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_shape=(299,299,3),
        pooling="avg")

    efficientnetb7_maxpool = EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_shape=(299,299,3),
        pooling="max")

    models = [
        xception_avgpool,
        xception_maxpool,
        efficientnetb3_avgpool,
        efficientnetb3_maxpool,
        efficientnetb7_avgpool,
        efficientnetb7_maxpool]

    model_names = [
        "xception_avgpool",
        "xception_maxpool",
        "efficientnetb3_avgpool",
        "efficientnetb3_maxpool",
        "efficientnetb7_avgpool", 
        "efficientnetb7_maxpool"]

    dfs = get_train_test_dfs()
    subset_names = ["train", "test"]

    for model, model_name in zip(models, model_names):
        for df, subset_name in zip(dfs, subset_names):
            extract_pretrained_features(model, model_name, df, subset_name)
