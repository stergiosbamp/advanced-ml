from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.pipeline import Pipeline
from imblearn.metrics import geometric_mean_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, make_scorer, f1_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.metrics import AUC
from keras.wrappers.scikit_learn import KerasClassifier

from dataloader import DataLoader


class Modelling:

    def __init__(self, classifier):
        self.x, self.y = DataLoader().get_x_y()  # Whole dataset for cross-validation
        # self.x_train, self.x_test, self.y_train, self.y_test = DataLoader().get_train_test()

        self.classifier = classifier

    def run(self, scoring):
        raise NotImplemented('Implement concrete class of Modelling')


class ImbalancedModelling(Modelling):

    def __init__(self, classifier):
        super().__init__(classifier)

    def run(self, scoring):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', self.classifier)
        ])
        results = cross_validate(estimator=model,
                                 X=self.x,
                                 y=self.y,
                                 # cv=cv,
                                 scoring=scoring,
                                 n_jobs=-1)

        for score in scoring.keys():
            cv_key = 'test_' + score  # construct the key of the cv to obtain avg
            print("\t\tAverage {} is: {}".format(score, results[cv_key].mean()))

    def __str__(self):
        return 'ImbalancedModelling({})'.format(self.classifier)


class BalancedModelling(Modelling):

    def __init__(self, classifier, resampler):
        super().__init__(classifier)
        self.resampler = resampler

    def run(self, scoring):
        # Note the Pipeline from "imblearn" to avoid data leakage!

        # To obtain metrics such as roc_auc we might have to manually iterate on CV splits
        # since it can't be applied directly on multi-class problems

        model = Pipeline([
            ('resampler', self.resampler),
            ('scaler', StandardScaler()),
            ('classifier', self.classifier)
        ])
        results = cross_validate(estimator=model,
                                 X=self.x,
                                 y=self.y,
                                 # cv=cv,
                                 scoring=scoring,
                                 n_jobs=-1)

        for score in scoring.keys():
            cv_key = 'test_' + score  # construct the key of the cv to obtain avg
            print("\t\tAverage {} is: {}".format(score, results[cv_key].mean()))

    def __str__(self):
        return 'BalancedModelling({}, {})'.format(self.classifier, self.resampler)


class DeepModelling:

    @staticmethod
    def init_model(input_dim=None):
        model = Sequential()

        model.add(Input(shape=(input_dim,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer='adam', metrics=[AUC()], loss='categorical_crossentropy')

        return model


if __name__ == '__main__':

    # Define constants for the experiments
    SCORING = {
        'b_acc': make_scorer(balanced_accuracy_score),
        'g_mean': make_scorer(geometric_mean_score),
        'f1': make_scorer(f1_score, average='macro')
    }
    UPSAMPLERS = [SMOTE(random_state=4), BorderlineSMOTE(random_state=4), ADASYN(random_state=4)]
    DOWNSAMPLERS = [TomekLinks(), RandomUnderSampler(random_state=4)]
    CLASSIFIERS = [
        RandomForestClassifier(n_estimators=200, random_state=4),
        GradientBoostingClassifier(random_state=4),
        KerasClassifier(build_fn=DeepModelling.init_model, input_dim=21, batch_size=64, epochs=100)
    ]

    for upsampler in UPSAMPLERS:
        print("Upsampler: {}".format(upsampler))
        for clf in CLASSIFIERS:
            imb_model = ImbalancedModelling(clf)
            print('\t{}'.format(imb_model))
            imb_model.run(scoring=SCORING)

            bal_model = BalancedModelling(clf, upsampler)
            print('\t{}'.format(bal_model))
            bal_model.run(scoring=SCORING)

    # TODO: Do the same loop for DOWNSAMPLERS
