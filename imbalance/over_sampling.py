from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE


class OverSampling:

    def __init__(self):
        self.smote = SMOTE(random_state=4)
        self.adasyn = ADASYN(random_state=4)
        self.borderline_smote = BorderlineSMOTE(random_state=4)

    def over_sample_smote(self, X, y):
        x_over, y_over = self.smote.fit_resample(X, y)
        return x_over, y_over

    def over_sample_adasyn(self, X, y):
        x_over, y_over = self.adasyn.fit_resample(X, y)
        return x_over, y_over

    def over_sample_border_smote(self, X, y):
        x_over, y_over = self.borderline_smote.fit_resample(X, y)
        return x_over, y_over
