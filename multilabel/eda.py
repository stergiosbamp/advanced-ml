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
        title='Label distribution (number of examples) on train set',
        width=1200,
        height=600)
    fig.show()

    fig = px.bar(
        x=class_mapping,
        y=label_counts_test,
        labels={"x":"Disease", "y":"Counts"},
        color_discrete_sequence=px.colors.qualitative.Safe,
        title='Label distribution (number of examples) on test set',
        width=1200,
        height=600)
    fig.show()


def plot_unique_label_combinations(Y_train, Y_test):

    Y_train_unique = set()
    for label_comb in np.unique(Y_train, axis=0):
        Y_train_unique.add(str(label_comb))

    Y_test_unique = set()
    for label_comb in np.unique(Y_test, axis=0):
        Y_test_unique.add(str(label_comb))

    train_set = len(Y_train_unique)
    test_set = len(Y_test_unique)

    overlapping = 0
    for test_label_comb in Y_test_unique:
        if test_label_comb in Y_train_unique:
            overlapping += 1

    fig = px.bar(
        x=["Train set", "Test set", "Overlapping"],
        y=[train_set, test_set, overlapping],
        labels={"x":"Subset", "y":"Unique label combinations"},
        color_discrete_sequence=px.colors.qualitative.Safe,
        title='Number of unique label combinations in and overlap between train and test set.',
        width=800,
        height=400)
    fig.show()


def plot_examples_with_no_label(Y_train, Y_test):

    no_label_train = np.count_nonzero(Y_train.sum(axis=1))
    at_least_one_label_train = Y_train.shape[0] - no_label_train

    no_label_test = np.count_nonzero(Y_test.sum(axis=1))
    at_least_one_label_test = Y_test.shape[0] - no_label_test

    labels = {}
    labels["No label (train)"] = no_label_train
    labels["At least one label (train)"] = at_least_one_label_train
    labels["No label (test)"] = no_label_test
    labels["At least one label (test)"] = at_least_one_label_test

    df_labels = pd.DataFrame.from_dict(labels, orient="index").rename(columns={0: "Number of Examples"})

    fig = px.bar(
        data_frame=df_labels,
        labels={"index":"Labels", "value":"Number of Examples"},
        color=["Train set"] * 2 + ["Test set"] * 2,
        color_discrete_sequence=px.colors.qualitative.Safe,
        title='Number of examples with no label vs with at least one label.',
        width=800,
        height=400)
    fig.show()


if __name__ == "__main__":

    from extract_features import get_train_test_dfs

    import pandas as pd


    df_train, df_test = get_train_test_dfs()

    class_mapping = pd.read_pickle("data/class_mapping.pkl")

    Y_train = np.stack(df_train["binarized_labels"].values, axis=0)
    Y_test = np.stack(df_test["binarized_labels"].values, axis=0)

    """
    print_num_examples(Y_train, Y_test)
    print_label_stats(Y_train, Y_test)
    print_label_counts(Y_train, Y_test)
    print_label_powerset(Y_train, Y_test)
    plot_label_counts(Y_train, Y_test, class_mapping)
    plot_unique_label_combinations(Y_train, Y_test)
    """

    plot_examples_with_no_label(Y_train, Y_test)