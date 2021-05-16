import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


from report_best_combo import populated_metrics_df


def plot_imbalance(df):
    fig = px.bar(df, barmode='group', color_discrete_sequence=px.colors.sequential.Aggrnyl)
    fig.show()


def plot_best_combo_per_metric(df):
    metrics = df.max().index.to_list()
    values_per_metric = df.max().to_list()
    models_per_metric = df.idxmax().to_list()

    labels = {}
    for metric, model in zip(metrics, models_per_metric):
        labels[metric] = model

    fig = px.bar(x=values_per_metric, y=metrics, orientation='h', color=labels,
                 text=values_per_metric, color_discrete_sequence=px.colors.qualitative.Safe,
                 labels={'x': 'value', 'y': 'metric'})
    fig.show()


if __name__ == '__main__':
    df_imbal = populated_metrics_df(directory='./imbalance')
    plot_imbalance(df_imbal)

    df_bal = populated_metrics_df(directory='./balance')
    plot_best_combo_per_metric(df_bal)
