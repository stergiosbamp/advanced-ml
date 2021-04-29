from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, precision_recall_curve, plot_precision_recall_curve
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from sklearn.multiclass import OneVsRestClassifier

from dataloader import DataLoader


class Evaluator:

    def __init__(self):
        self.roc_aucs = []
        self.f1_scores = []
        self.g_means = []
        self.b_accs = []

    def log_roc_auc(self, y_true, y_predicted_prob, multi_class='ovr'):
        # For roc in multi-class problems we need the probabilities of each class
        auc = roc_auc_score(y_true, y_predicted_prob, multi_class=multi_class)
        self.roc_aucs.append(auc)

    @staticmethod
    def report_imbalance(y_true, y_predicted):
        print(classification_report_imbalanced(y_true, y_predicted))

    def log_balanced_accuracy(self, y_true, y_predicted):
        b_acc = balanced_accuracy_score(y_true, y_predicted)
        self.b_accs.append(b_acc)

    def log_geometric_mean(self, y_true, y_predicted):
        g_mean = geometric_mean_score(y_true, y_predicted)
        self.g_means.append(g_mean)

    def log_f1_score(self, y_true, y_predicted, average='macro'):
        f1 = f1_score(y_true, y_predicted, average=average)
        self.f1_scores.append(f1)

    def log_metrics(self, y_true, y_predicted, y_predicted_prob):
        self.log_balanced_accuracy(y_true, y_predicted)
        self.log_f1_score(y_true, y_predicted)
        self.log_geometric_mean(y_true, y_predicted)
        self.log_roc_auc(y_true, y_predicted_prob)

    def get_avg_metrics(self):
        avg_auc_roc = sum(self.roc_aucs) / len(self.roc_aucs)
        avg_b_acc = sum(self.b_accs) / len(self.b_accs)
        avg_g_mean = sum(self.g_means) / len(self.g_means)
        avg_f1_score = sum(self.f1_scores) / len(self.f1_scores)

        return {
            'avg_auc_roc': avg_auc_roc,
            'avg_b_acc': avg_b_acc,
            'avg_g_mean': avg_g_mean,
            'avg_f1_score': avg_f1_score
        }

    @staticmethod
    def plot_pr_curve(clf, x_test, y_test):
        # Needs adjustments for multi-class classification
        pass
