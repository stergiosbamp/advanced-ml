from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, make_scorer, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from itertools import combinations, product, permutations

from dataloader import DataLoader


class Modelling:

    def __init__(self, classifier, resampler):
        self.x, self.y = DataLoader().get_x_y()
        # self.x_train, self.x_test, self.y_train, self.y_test = DataLoader().get_train_test()

        self.classifier = classifier
        self.resampler = resampler

    def run(self, scoring):
        # Note the Pipeline from "imblearn" to avoid data leakage!
        
        # To obtain metrics such as roc_auc we might must to manually iterate on CV splits
        # since it can't be applied directly on multi-class problems
        
        model = Pipeline([
            ('r', self.resampler),
            ('s', StandardScaler()),
            ('c', self.classifier)
        ])
        results = cross_validate(estimator=model,
                                 X=self.x,
                                 y=self.y,
                                 # cv=cv,
                                 scoring=scoring,
                                 n_jobs=-1)

        print("For classifier: {} with resampler: {}".format(self.classifier, self.resampler))
        for score in scoring.keys():
            cv_key = 'test_' + score  # construct the key of the cv to obtain avg
            print("\tAverage {} is: {}".format(score, results[cv_key].mean()))
        

if __name__ == '__main__':
    UPSAMPLERS = [SMOTE(random_state=4), BorderlineSMOTE(random_state=4), ADASYN(random_state=4)]
    random_forest = RandomForestClassifier(n_estimators=200, random_state=4)

    for upsampler in UPSAMPLERS:
        modelling = Modelling(random_forest, upsampler)
        modelling.run(scoring={
            'b_acc': make_scorer(balanced_accuracy_score),
            'g_mean': make_scorer(geometric_mean_score)
        })
