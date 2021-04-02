from imblearn.under_sampling import TomekLinks, NearMiss

from imbalance.dataloader import DataLoader


class UnderSampling:

    def __init__(self):
        self.tomek_links = TomekLinks()
        self.near_miss = NearMiss()

    def under_sample_tomek_links(self, x, y):
        x_under, y_under = self.tomek_links.fit_resample(x, y)
        return x_under, y_under

    def under_sample_nearmiss(self, x, y, version=1):
        self.near_miss.version = version

        x_under, y_under = self.near_miss.fit_resample(x, y)
        return x_under, y_under


# Example use
if __name__ == '__main__':
    dataloader = DataLoader()

    under_sampler = UnderSampling()
    x_train, x_test, y_train, y_test = dataloader.get_train_test()

    under_sampler.under_sample_nearmiss(x_train, y_train, 3)
