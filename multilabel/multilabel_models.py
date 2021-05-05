import pickle


def load_extracted_features(train_pkl, test_pkl):

    # load train set
    with open(train_pkl, "rb") as f:
        data_train = pickle.load(f)

    # load test set
    with open(test_pkl, "rb") as f:
        data_test = pickle.load(f)

    # get X, Y
    X_train = data_train["X"]
    Y_train = data_train["Y"]

    X_test = data_test["X"]
    Y_test = data_test["Y"]

    return X_train, Y_train, X_test, Y_test


def train_predict_save(clf, clf_name, X_train, Y_train, X_test, Y_test, dest_dir="data"):

    print("\nFitting: ", clf_name)

    # fit
    clf.fit(X_train, Y_train)

    # predict train and test set
    Y_train_pred = clf.predict(X_train)
    Y_test_pred = clf.predict(X_test)

    # save fitted model
    with open(dest_dir + "/" + clf_name + "_clf.pkl", "wb") as f:
        pickle.dump(clf, f, protocol=4)
    
    # save predictions
    with open(dest_dir + "/" + clf_name + "_y_train_pred.pkl", "wb") as f:
        pickle.dump(Y_train_pred, f, protocol=4)
        
    with open(dest_dir + "/" + clf_name + "_y_test_pred.pkl", "wb") as f:
        pickle.dump(Y_test_pred, f, protocol=4)


if __name__ == "__main__":

    from sklearn.linear_model import SGDClassifier
    from skmultilearn.ensemble import RakelD
    from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset

    X_train, Y_train, X_test, Y_test = load_extracted_features(
        "data/xception_fine_tuned_train.pkl",
        "data/xception_fine_tuned_test.pkl")

    clfs = [
        BinaryRelevance(SGDClassifier()),
        ClassifierChain(SGDClassifier()),
        LabelPowerset(SGDClassifier()),
        RakelD(SGDClassifier())]

    clf_names = [
        "binaryrelevance_sgdclassifier",
        "classifierchain_sgdclassifier",
        "labelpowerset_sgdclassifier",
        "rakeld_sgdclassifier"]

    for clf, clf_name in zip(clfs, clf_names):
        train_predict_save(clf, clf_name, X_train, Y_train, X_test, Y_test)
