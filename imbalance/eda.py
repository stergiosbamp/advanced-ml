import pandas as pd
import plotly.express as px


class EDA:

    def __init__(self, df, target):
        self.df = df
        self.target = target

    def pie_chart_class_distr(self):
        # Replace numbers with their meaning for visualization reasons
        fetal_health_classes = self.df[self.target]
        fetal_health_classes.replace({
            1: 'Normal',
            2: 'Suspect',
            3: 'Pathological'
        }, inplace=True)

        title = 'Class distribution of fetal Cardiotocograms'

        fig = px.pie(fetal_health_classes, names=self.target, hole=0.2, title=title, width=800, height=800,
                     color_discrete_sequence=px.colors.qualitative.Safe)
        fig.show()


if __name__ == '__main__':
    df = pd.read_csv('../datasets/fetal_health.csv')

    eda = EDA(df=df, target='fetal_health')
    eda.pie_chart_class_distr()
