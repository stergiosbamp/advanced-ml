import json
import os
import pickle

from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score, precision_score, recall_score, roc_auc_score


def train_evaluate_save(clf, clf_name, X_train, Y_train, X_test, Y_test, dest_dir="data"):

    # model pickle file
    model_path = dest_dir + "/" + clf_name + "_clf.pkl"

    # check if model exists, else train
    if os.path.exists(model_path):
        print(f"\nModel already trained: {clf_name}")
        with open(model_path, "rb") as f:
            clf = pickle.load(f)    
    else:
        print(f"\nFitting model: {clf_name}")
        clf.fit(X_train, Y_train)

    # evaluation json file
    evaluation_path = dest_dir + "/" + clf_name + "_eval.json"

    Y_pred = clf.predict(X_test)
    Y_pred_proba = clf.predict_proba(X_test)
    Y_pred_proba = Y_pred_proba.toarray()

    acc = accuracy_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, Y_pred_proba)
    f1_macro = f1_score(Y_test, Y_pred, average="macro")
    f1_micro = f1_score(Y_test, Y_pred, average="micro")
    prec_macro = precision_score(Y_test, Y_pred, average="macro")
    prec_micro = precision_score(Y_test, Y_pred, average="micro")
    rec_macro = recall_score(Y_test, Y_pred, average="macro")
    rec_micro = recall_score(Y_test, Y_pred, average="micro")
    f1_labels = f1_score(Y_test, Y_pred, average=None)
    prec_labels = precision_score(Y_test, Y_pred, average=None)
    rec_labels = recall_score(Y_test, Y_pred, average=None)
    hamm_loss = hamming_loss(Y_test, Y_pred)
    jaccard_samples = jaccard_score(Y_test, Y_pred, average="samples")

    results = {
        "acc": acc,
        "roc_auc": roc_auc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "prec_macro": prec_macro,
        "prec_micro": prec_micro,
        "rec_macro": rec_macro,
        "rec_micro": rec_micro,
        "f1_labels": f1_labels,
        "prec_labels": prec_labels,
        "rec_labels": rec_labels,
        "hamm_loss": hamm_loss,
        "jaccard_samples": jaccard_samples
    }

    # save fitted model
    with open(dest_dir + "/" + clf_name + "_evaluation.json", "w") as f:
        json.dump(results, f)
