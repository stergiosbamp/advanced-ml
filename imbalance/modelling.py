import pathlib
import json

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.pipeline import Pipeline
from imblearn.metrics import geometric_mean_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.metrics import AUC
from keras.wrappers.scikit_learn import KerasClassifier

from dataloader import DataLoader
from evaluation import Evaluator


class Modelling:
    RESULTS_PATH = 'results/'

    def __init__(self, classifier):
        self.x, self.y = DataLoader().get_x_y()  # Whole dataset for cross-validation
        # self.x_train, self.x_test, self.y_train, self.y_test = DataLoader().get_train_test()

        self.classifier = classifier
        self.evaluator = Evaluator()

    def run(self):
        raise NotImplemented('Implement concrete class of Modelling')

    def save_results(self, results, dist_type, filename):
        dest = pathlib.Path(self.RESULTS_PATH, dist_type, filename)
        dest = dest.with_suffix('.json')
        with dest.open('w', encoding='utf-8') as f:
            json.dump(results, f)

    def model_has_run(self):
        raise NotImplemented('Implement concrete class of Modelling')


class ImbalancedModelling(Modelling):

    def __init__(self, classifier):
        super().__init__(classifier)
        if isinstance(self.classifier, KerasClassifier):
            self.model_id = 'MLP'
        else:
            self.model_id = str(self.classifier)

    def run(self):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', self.classifier)
        ])

        if self.model_has_run():
            print("\tModel has already run, skipping...")
            return

        cross_validator = StratifiedKFold(n_splits=5)

        for train_idx, test_idx in cross_validator.split(self.x, self.y):
            x_train, x_test = self.x.iloc[train_idx], self.x.iloc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred_prob = model.predict_proba(x_test)

            self.evaluator.log_metrics(y_test, y_pred, y_pred_prob)

        results = self.evaluator.get_avg_metrics()

        for key, value in results.items():
            print("\t\tAverage {} is: {}".format(key, value))

        self.save_results(results, 'imbalance', self.model_id)

    def model_has_run(self):
        dest = pathlib.Path(self.RESULTS_PATH, 'imbalance', self.model_id)
        dest = dest.with_suffix('.json')
        if dest.exists():
            return True
        return False

    def __str__(self):
        return 'ImbalancedModelling({})'.format(self.classifier)


class BalancedModelling(Modelling):

    def __init__(self, classifier, resampler):
        super().__init__(classifier)
        self.resampler = resampler
        if isinstance(self.classifier, KerasClassifier):
            self.model_id = str(self.resampler) + "-" + 'MLP'
        else:
            self.model_id = str(self.resampler) + "-" + str(self.classifier)

    def run(self):
        # Note the Pipeline from "imblearn" to avoid data leakage!

        model = Pipeline([
            ('resampler', self.resampler),
            ('scaler', StandardScaler()),
            ('classifier', self.classifier)
        ])

        if self.model_has_run():
            print("\tModel has already run, skipping...")
            return

        cross_validator = StratifiedKFold(n_splits=5)

        for train_idx, test_idx in cross_validator.split(self.x, self.y):
            x_train, x_test = self.x.iloc[train_idx], self.x.iloc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred_prob = model.predict_proba(x_test)

            self.evaluator.log_metrics(y_test, y_pred, y_pred_prob)

        results = self.evaluator.get_avg_metrics()

        for key, value in results.items():
            print("\t\tAverage {} is: {}".format(key, value))

        self.save_results(results, 'balance', self.model_id)

    def model_has_run(self):
        dest = pathlib.Path(self.RESULTS_PATH, 'balance', self.model_id)
        dest = dest.with_suffix('.json')
        if dest.exists():
            return True
        return False

    def custom_run(self, model: Pipeline):
        if self.model_has_run():
            print("\tModel has already run, skipping...")
            return

        cross_validator = StratifiedKFold(n_splits=5)

        for train_idx, test_idx in cross_validator.split(self.x, self.y):
            x_train, x_test = self.x.iloc[train_idx], self.x.iloc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred_prob = model.predict_proba(x_test)

            self.evaluator.log_metrics(y_test, y_pred, y_pred_prob)

        results = self.evaluator.get_avg_metrics()

        for key, value in results.items():
            print("\t\tAverage {} is: {}".format(key, value))

        self.save_results(results, 'balance', self.model_id)
    
    def __str__(self):
        return 'BalancedModelling({}, {})'.format(self.classifier, self.resampler)


class DeepModelling:

    @staticmethod
    def init_model(input_dim=None):
        model = Sequential()

        model.add(Input(shape=(input_dim,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer='adam', metrics=[AUC()], loss='categorical_crossentropy')

        return model


if __name__ == '__main__':

    UPSAMPLERS = [SMOTE(random_state=4), RandomOverSampler(random_state=0), BorderlineSMOTE(random_state=4), ADASYN(random_state=4)]
    DOWNSAMPLERS = [TomekLinks(), RandomUnderSampler(random_state=4), NearMiss(version=1)]
    CLASSIFIERS = [
        RandomForestClassifier(n_estimators=200, random_state=4),
        GradientBoostingClassifier(random_state=4),
        KerasClassifier(build_fn=DeepModelling.init_model, input_dim=21, batch_size=64, epochs=100),
        LogisticRegression(max_iter=500, random_state=0)
    ]

    # Experiment with no balancing method
    for clf in CLASSIFIERS:
        imb_model = ImbalancedModelling(clf)
        print('\n{}'.format(imb_model))
        imb_model.run()

    # Experiment with over-sampling methods
    for upsampler in UPSAMPLERS:
        print("Upsampler: {}".format(upsampler))
        for clf in CLASSIFIERS:
            bal_model = BalancedModelling(clf, upsampler)
            print('\t{}'.format(bal_model))
            bal_model.run()

    # Experiment with down-sampling methods
    for downsampler in DOWNSAMPLERS:
        print("Downsampler: {}".format(downsampler))
        for clf in CLASSIFIERS:
            bal_model = BalancedModelling(clf, downsampler)
            print('\t{}'.format(bal_model))
            bal_model.run()

    # Experiment with hybrid method via SMOTE + TomekLinks for every classifier
    for clf in CLASSIFIERS:
        b_modelling = BalancedModelling(clf, None)

        # Setup ids for logging results
        if isinstance(clf, KerasClassifier):
            b_modelling.model_id = "SMOTE-TomekLinks-" + "MLP"
        else:
            b_modelling.model_id = "SMOTE-TomekLinks-" + str(clf)
        
        pipe = Pipeline([
            ('smote', SMOTE(random_state=0)),
            ('tomek', TomekLinks()),
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        b_modelling.custom_run(model=pipe)
