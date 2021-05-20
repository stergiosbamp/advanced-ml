import numpy as np
import plotly.express as px


def print_num_examples(Y_train, Y_test):

    print(f"\nTotal examples: {Y_train.shape[0] + Y_test.shape[0]}")
    print(f"Train examples: {Y_train.shape[0]}")
    print(f"Test examples: {Y_test.shape[0]}")


def print_label_stats(Y_train, Y_test):

    print(f"\nLabel cardinality: {Y_train.shape[1]}")

    print(f"\nMin number of labels on train set examples: {Y_train.sum(axis=1).min()}")
    print(f"Avg number of labels on train set examples: {Y_train.sum(axis=1).mean()}")
    print(f"Max number of labels on train set examples: {Y_train.sum(axis=1).max()}")

    print(f"\nMin number of labels on test set examples: {Y_test.sum(axis=1).min()}")
    print(f"Avg number of labels on test set examples: {Y_test.sum(axis=1).mean()}")
    print(f"Max number of labels on test set examples: {Y_test.sum(axis=1).max()}")


def print_label_counts(Y_train, Y_test):

    print(f"\nNumber of examples per label on train set: {Y_train.sum(axis=0)}")
    print(f"Number of examples per label on test set: {Y_test.sum(axis=0)}")


def print_label_powerset(Y_train, Y_test):

    Y_train_unique = np.unique(Y_train, axis=0)
    Y_test_unique = np.unique(Y_test, axis=0)

    print(f"\nNumber of unique label combinations on train set: {Y_train_unique.shape[0]}")
    print(f"Number of unique label combinations on test set: {Y_test_unique.shape[0]}")

    overlapping = 0
    for test_row in Y_test_unique:
        if test_row in Y_train_unique:
            overlapping += 1
    
    print(f"\nNumber of unique test set label combination that are present in train set: {overlapping}")
    print(f"Number of unique test set label combination that are not present in train set: {Y_test_unique.shape[0] - overlapping}")
    print(f"Percentage of unique test set label combination that are present in train set: {overlapping / Y_test_unique.shape[0] * 100}%")


def plot_label_counts(Y_train, Y_test, class_mapping):

    label_counts_train = Y_train.sum(axis=0)
    label_counts_test = Y_test.sum(axis=0)

    fig = px.bar(
        x=class_mapping,
        y=label_counts_train,
        labels={"x":"Disease", "y":"Counts"},
        color_discrete_sequence=px.colors.qualitative.Safe,
        title='Label distribution (number of examples) on train set')
    fig.show()

    fig = px.bar(
        x=class_mapping,
        y=label_counts_test,
        labels={"x":"Disease", "y":"Counts"},
        color_discrete_sequence=px.colors.qualitative.Safe,
        title='Label distribution (number of examples) on test set')
    fig.show()


if __name__ == "__main__":

    from extract_features import get_train_test_dfs

    import pandas as pd


    df_train, df_test = get_train_test_dfs()

    class_mapping = pd.read_pickle("data/class_mapping.pkl")

    Y_train = np.stack(df_train["binarized_labels"].values, axis=0)
    Y_test = np.stack(df_test["binarized_labels"].values, axis=0)

    print_num_examples(Y_train, Y_test)
    print_label_stats(Y_train, Y_test)
    print_label_counts(Y_train, Y_test)
    print_label_powerset(Y_train, Y_test)
    plot_label_counts(Y_train, Y_test, class_mapping)