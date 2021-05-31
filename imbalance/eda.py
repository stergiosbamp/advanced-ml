import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

from dataloader import DataLoader


class EDA:

    def __init__(self, df, target):
        self.df = df
        self.target = target

    def pie_chart_class_distr(self):
        # Replace numbers with their meaning for visualization reasons
        fetal_health_classes = self.df[self.target]
        fetal_health_classes = fetal_health_classes.replace({
            1: 'Normal',
            2: 'Suspect',
            3: 'Pathological'
        })

        title = 'Class distribution (percentage) of fetal Cardiotocograms'

        fig = px.pie(fetal_health_classes, names=self.target, hole=0.2, title=title, width=650, height=650,
                     color_discrete_sequence=px.colors.qualitative.Safe)
        fig.show()

    def bar_chart_class_distr(self):
        # Replace numbers with their meaning for visualization reasons
        fetal_health_classes = self.df[self.target]
        fetal_health_classes = fetal_health_classes.replace({
            1: 'Normal',
            2: 'Suspect',
            3: 'Pathological'
        })
        counts = fetal_health_classes.value_counts()
        fig = px.bar(data_frame=counts, color_discrete_sequence=px.colors.qualitative.Safe,
                     title='Class distribution (number of examples) of fetal Cardiotocograms',
                     labels={'index': 'Classes', 'value': 'Examples'},
                     width=650, height=650)
        fig.show()

    def box_plot(self):
        df_unscaled = self.df
        fig = px.box(data_frame=df_unscaled, color_discrete_sequence=px.colors.sequential.Aggrnyl)
        fig.show()

    def corr_heatmap(self):
        corr = df.corr()
        fig = px.imshow(img=corr, color_continuous_scale=px.colors.sequential.YlGnBu, width=700, height=700)
        fig.show()


if __name__ == '__main__':
    df = pd.read_csv('../datasets/fetal_health.csv')

    eda = EDA(df=df, target='fetal_health')
    eda.pie_chart_class_distr()
    eda.bar_chart_class_distr()
    eda.box_plot()
    eda.corr_heatmap()
