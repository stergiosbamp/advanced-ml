import pickle

import numpy as np
import pandas as pd

from tensorflow.keras import Model
from tensorflow.keras.models import load_model
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


def extract_features(
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

def get_fine_tuned_model_base(fine_tuned_model_file):
    
    # Load model from .h5 file
    fine_tuned_model = load_model(fine_tuned_model_file)

    # Get model base
    x_in = fine_tuned_model.input
    x_out = fine_tuned_model.layers[-2](x_in)
    fine_tuned_model_base = Model(inputs=x_in, outputs=x_out)

    return fine_tuned_model_base
