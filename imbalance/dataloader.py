import pandas as pd

from sklearn.model_selection import train_test_split


class DataLoader:

    def __init__(self, file='../datasets/fetal_health.csv', target='fetal_health'):
        self.df = pd.read_csv(file)
        self.y = self.df[target]
        self.x = self.df.drop(labels=target, axis=1)

    def get_x_y(self):
        return self.x, self.y

    def get_train_test(self, split_size=0.3, random_state=4):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y,
                                                            random_state=random_state,
                                                            test_size=split_size)
        return x_train, x_test, y_train, y_test
