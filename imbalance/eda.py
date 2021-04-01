import pandas as pd
import plotly.express as px


class EDA:

    def __init__(self, df):
        self.df = df

    def pie_chart(self, name):
        title = 'Class distribution of fetal Cardiotocograms'

        fig = px.pie(self.df, names=name, hole=0.2, title=title)
        fig.show()


if __name__ == '__main__':
    df = pd.read_csv('../datasets/fetal_health.csv')

    eda = EDA(df=df)
    eda.pie_chart('fetal_health')

