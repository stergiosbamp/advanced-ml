from extract_features import load_extracted_features
from model_multilabel import train_evaluate_save

from sklearn.linear_model import SGDClassifier
from skmultilearn.ensemble import RakelD
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset


X_train, Y_train, X_test, Y_test = load_extracted_features(
    "data/xception_bpmll_fine_tuned_train.pkl",
    "data/xception_bpmll_fine_tuned_test.pkl")

clfs = [
    BinaryRelevance(SGDClassifier(loss="log")),
    ClassifierChain(SGDClassifier(loss="log")),
    LabelPowerset(SGDClassifier(loss="log")),
    RakelD(SGDClassifier(loss="log"))]

clf_names = [
    "binaryrelevance_sgdclassifier",
    "classifierchain_sgdclassifier", 
    "labelpowerset_sgdclassifier",
    "rakeld_sgdclassifier"]

for clf, clf_name in zip(clfs, clf_names):
    train_evaluate_save(
        clf,
        clf_name,
        X_train, 
        Y_train,
        X_test,
        Y_test,
        dest_dir="data/xception_bpmll_fine_tuned")
