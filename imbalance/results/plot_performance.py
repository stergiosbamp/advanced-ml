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


def plot_before_after(df_imb, df_bal, models):
    sub_titles = [ 'Before/After for {} model'.format(model) for model in models]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, shared_yaxes=True, subplot_titles=sub_titles)

    metrics = df_imb.columns.to_list()
    for i, model in enumerate(models):
        for idx in df_imb.index:
            if model in idx:
                value_metrics_imb = df_imbal.loc[idx].to_list()

        all_metrics_combos_bal = pd.DataFrame()
        for idx in df_bal.index:
            if model in idx:
                temp_metrics = df_bal.loc[idx]
                all_metrics_combos_bal = all_metrics_combos_bal.append(temp_metrics)

        value_metrics_bal = all_metrics_combos_bal.max().to_list()

        fig.add_trace(go.Bar(name='Before sampling', x=metrics, y=value_metrics_imb),
                      row=i+1, col=1)
        fig.add_trace(go.Bar(name='After (best) sampling', x=metrics, y=value_metrics_bal),
                      row=i+1, col=1)
    fig.show()


if __name__ == '__main__':
    df_imbal = populated_metrics_df(directory='./imbalance')
    # plot_imbalance(df_imbal)

    df_bal = populated_metrics_df(directory='./balance')
    # plot_best_combo_per_metric(df_bal)

    plot_before_after(df_imbal, df_bal, ['MLP', 'LogisticRegression', 'GradientBoosting', 'RandomForest'])
