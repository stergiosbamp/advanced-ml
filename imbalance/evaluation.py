from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, plot_precision_recall_curve
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from sklearn.multiclass import OneVsRestClassifier

from dataloader import DataLoader


class Evaluator:

    def __init__(self, y_true, y_predicted):
        self.y_true = y_true
        self.y_predicted = y_predicted

    def roc_auc(self, y_pred_prob, multi_class='ovr'):
        # For roc in multi-class problems we need the probabilities of each class
        auc = roc_auc_score(self.y_true, y_pred_prob, multi_class=multi_class)
        print("AUC-ROC: {}".format(auc))
        return auc

    def report_imbalance(self):
        print(classification_report_imbalanced(self.y_true, self.y_predicted))

    @staticmethod
    def plot_pr_curve(clf, x_test, y_test):
        # Needs adjustments for multi-class classification
        pass


# Example use
if __name__ == '__main__':
    xtr, xte, ytr, yte = DataLoader().get_train_test()

    clf = AdaBoostClassifier(n_estimators=100, random_state=4)
    clf.fit(xtr, ytr)

    y_pred = clf.predict(xte)
    y_pred_prob = clf.predict_proba(xte)

    ev = Evaluator(yte, y_pred)
    ev.report_imbalance()
    ev.roc_auc(y_pred_prob)
