import os
import re
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from report_best_combo import populated_metrics_df


CLF_RENAMER = [
    (r'(.+)?GradientBoosting', "Gradient Boosting"),
    (r'(.+)?MLP', "MLP"),
    (r'(.+)?LogisticRegression', "Logistic Regression"),
    (r'(.+)?RandomForest', "Random Forest")
]

RESAMPLER_RENAMER = [
    (r'(.+)?ADASYN', "ADASYN"),
    (r'(.+)?BorderlineSMOTE', "Borderline SMOTE"),
    (r'(.+)?NearMiss', "Nearmiss (v1)"),
    (r'(.+)?RandomOverSampler', "Random oversampling"),
    (r'(.+)?RandomUnderSampler', "Random undersampling"),
    (r'(.+)?SMOTE', "SMOTE"),
    (r'(.+)?SMOTE-TomekLinks', "SMOTE - Tomek links"),
    (r'(.+)?TomekLinks', "Tomek links"),
]


def metrics_renamer():
    return {
        'avg_auc_roc': 'ROC AUC (avg)',
        'avg_b_acc': 'Balanced accuracy (avg)',
        'avg_g_mean': 'Geometric mean (avg)',
        'avg_f1_score': 'F1-score (avg)'
    }


def plot_imbalance(df):
    # Rename classifiers from json files
    old_indices = df.index.to_list()
    new_indices = []

    for ind in old_indices:
        for renamer in CLF_RENAMER:
            regex, new_name = renamer
            if re.match(regex, ind):
                new_indices.append(new_name)
    df.index = new_indices

    # Rename metrics
    df.rename(columns=metrics_renamer(), inplace=True)
    fig = px.bar(df, barmode='group', color_discrete_sequence=px.colors.qualitative.Safe,
                 title='Performance without handling imbalance',
                 width=1000, height=1000)
    fig.show()


def plot_best_combo_per_metric(df):
    metrics = df.max().index.to_list()
    values_per_metric = df.max().to_list()
    models_per_metric = df.idxmax().to_list()

    labels = {}
    for metric, model in zip(metrics, models_per_metric):
        labels[metric] = model

    # Start renaming for nice visualizations
    new_labels = {}
    metrics_mapper = metrics_renamer()
    for metric, combo in labels.items():
        new_metric = metrics_mapper[metric]
        resampler, clf = combo.split('-')
        for renamer in CLF_RENAMER:
            regex, new_name = renamer
            if re.match(regex, clf):
                new_clf = new_name
        for renamer in RESAMPLER_RENAMER:
            regex, new_name = renamer
            if re.match(regex, resampler):
                new_resampler = new_name

        new_combo = "{} - {}".format(new_resampler, new_clf)
        new_labels[new_metric] = new_combo

    fig = px.bar(x=values_per_metric, y=new_labels.keys(), orientation='h', color=new_labels,
                 text=values_per_metric, color_discrete_sequence=px.colors.qualitative.Safe,
                 labels={'x': 'value', 'y': 'metric'}, height=800,
                 title='Best model and sampling technique per evaluation metric')
    fig.show()


def plot_before_after(df_imb, df_bal, models):
    sub_titles = ['Before/After for {} model'.format(model) for model in models]
    fig = make_subplots(rows=4, cols=1, subplot_titles=sub_titles)

    metrics = list(metrics_renamer().values())

    for i, model in enumerate(models):
        for idx in df_imb.index:
            if model in idx:
                value_metrics_imb = df_imb.loc[idx].to_list()

        all_metrics_combos_bal = pd.DataFrame()
        for idx in df_bal.index:
            if model in idx:
                temp_metrics = df_bal.loc[idx]
                all_metrics_combos_bal = all_metrics_combos_bal.append(temp_metrics)

        value_metrics_bal = all_metrics_combos_bal.max().to_list()

        fig.add_trace(go.Bar(name='Before sampling', x=metrics, y=value_metrics_imb),
                      row=i + 1, col=1)
        fig.add_trace(go.Bar(name='After (best) sampling', x=metrics, y=value_metrics_bal),
                      row=i + 1, col=1)
    fig.show()


def plot_confusion_matrix(cf_path, title):
    cf = json.load(open(cf_path, 'rb'))
    x_labels = ['Predicted Normal', 'Predicted Suspect', 'Predicted Pathological']
    y_labels = ['True Normal', 'True Suspect', 'True Pathological']
    fig = ff.create_annotated_heatmap(cf, x=x_labels, y=y_labels, annotation_text=cf,
                                      colorscale=px.colors.sequential.Aggrnyl)

    fig.update_layout(title=title, width=650, height=650)
    fig.show()


if __name__ == '__main__':
    df_imbal = populated_metrics_df(directory='./results/imbalance')
    # plot_imbalance(df_imbal)

    df_bal = populated_metrics_df(directory='./results/balance')
    plot_best_combo_per_metric(df_bal)

    # plot_before_after(df_imbal, df_bal, ['MLP', 'LogisticRegression', 'GradientBoosting', 'RandomForest'])

    # # Confusion matrix before/after GD
    # plot_confusion_matrix('./results/confusion-matrices/GradientBoostingClassifier(random_state=4).json', 'Gradient Boosting')
    # plot_confusion_matrix('./results/confusion-matrices/SMOTE(random_state=4)-GradientBoostingClassifier(random_state=4).json',
    #                       'SMOTE - Gradient Boosting')
    #
    # # Confusion matrix before/after MLP
    # plot_confusion_matrix('./results/confusion-matrices/MLP.json', 'MLP')
    # plot_confusion_matrix('./results/confusion-matrices/RandomUnderSampler(random_state=4)-MLP.json',
    #                       'Random undersampling - MLP')
