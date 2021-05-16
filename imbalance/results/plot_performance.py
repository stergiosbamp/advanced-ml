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

    fig = go.Figure()

    for metric, value, model in zip(metrics, values_per_metric, models_per_metric):
        fig.add_trace(go.Bar(name=model, x=[metric], y=[value], ))

    fig.show()


if __name__ == '__main__':
    df_imbal = populated_metrics_df(directory='./imbalance')
    plot_imbalance(df_imbal)

    df_bal = populated_metrics_df(directory='./balance')
    plot_best_combo_per_metric(df_bal)
