import pandas as pd
import seaborn as sns
import plotly.express as px


from report_best_combo import populated_metrics_df


def plot_imbalance(df):
    fig = px.bar(df, barmode='group', color_discrete_sequence=px.colors.sequential.Aggrnyl)
    fig.show()


if __name__ == '__main__':
    df_imbal = populated_metrics_df(directory='./imbalance')
    plot_imbalance(df_imbal)
