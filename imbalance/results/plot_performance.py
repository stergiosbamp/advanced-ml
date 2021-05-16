import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from report_best_combo import populated_metrics_df


def plot_imbalance(df):
    fig = px.bar(df, barmode='group', color_discrete_sequence=px.colors.qualitative.Safe)
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


def plot_before_after(df_imb, df_bal, model_idx):
    for idx in df_imb.index:
        if model_idx in idx:
            metrics = df_imbal.loc[idx].index.to_list()
            value_metrics_imb = df_imbal.loc[idx].to_list()

    all_metrics_combos_bal = pd.DataFrame()
    for idx in df_bal.index:
        if model_idx in idx:
            temp_metrics = df_bal.loc[idx]
            all_metrics_combos_bal = all_metrics_combos_bal.append(temp_metrics)

    value_metrics_bal = all_metrics_combos_bal.max().to_list()

    data = {}
    for metric, value_before, value_after in zip(metrics, value_metrics_imb, value_metrics_bal):
        data[metric] = [value_before, value_after]
    display_df = pd.DataFrame(data=data, index=['Before', 'After'])
    fig = px.bar(display_df, orientation='h', barmode='group', color_discrete_sequence=px.colors.qualitative.Safe)
    fig.show()


if __name__ == '__main__':
    df_imbal = populated_metrics_df(directory='./imbalance')
    # plot_imbalance(df_imbal)

    df_bal = populated_metrics_df(directory='./balance')
    # plot_best_combo_per_metric(df_bal)

    plot_before_after(df_imbal, df_bal, 'Boosting')
