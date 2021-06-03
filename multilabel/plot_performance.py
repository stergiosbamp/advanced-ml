import glob
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots


def plot_fine_tuned_models_perf():

    # read performance metrics for binary cross entropy loss function
    perf = {}
    for fname in glob.glob("results/chest-x-rays-xception-fine-tuned-evaluation/*.json"):
        with open(fname, "rb") as f:
            model = fname.split("/")[-1].split("_")[0]
            perf[model] = json.load(f)
    
    # plot roc auc for fine-tuned features with binary cross entropy
    rocauc_per_model = {}
    for key, value in perf.items():
        rocauc_per_model[key] = value["roc_auc"]
    rocauc_per_model["Wang et al."] = 0.738143
    rocauc_per_model["Yao et al."] = 0.802714
    rocauc_per_model["CheXNet"] = 0.841378

    df_rocauc_per_model = pd.DataFrame.from_dict(rocauc_per_model, orient="index").rename(columns={0: "ROC AUC"})

    fig = px.bar(
        data_frame=df_rocauc_per_model,
        labels={"index":"Model", "value":"ROC AUC"},
        color=["Our multi-label models"] * 4 + ["Other models"] * 3,
        color_discrete_sequence=px.colors.qualitative.Safe,
        title='ROC AUC Performance',
        width=1200,
        height=600)
    fig.show()

    # plot macro and micro f1, prec, rec for fine-tuned features with binary cross entropy
    f1_prec_rec = {}
    for key, value in perf.items():
        f1_prec_rec[key] = [value["f1_macro"], value["f1_micro"], value["prec_macro"], value["prec_micro"], value["rec_macro"], value["rec_micro"]]
    print(f1_prec_rec)

    df_f1_prec_rec = pd.DataFrame.from_dict(f1_prec_rec, orient="index").rename(columns={0:"F1-macro", 1:"F1-micro", 2:"Prec-macro", 3:"Prec-micro", 4:"Rec-macro", 5:"Rec-micro"})
    print(df_f1_prec_rec)

    fig = px.bar(
        data_frame=df_f1_prec_rec,
        barmode="group",
        labels={"index":"Model", "value":"Metrics"},
        color_discrete_sequence=px.colors.qualitative.Safe,
        title='Performance Metrics',
        width=1200,
        height=600)
    fig.show()

    # read performance metrics for bp-mll loss function
    perf_bpmll = {}
    for fname in glob.glob("results/chest-x-rays-xception-bpmll-fine-tuned-evaluation/*.json"):
        with open(fname, "rb") as f:
            model = fname.split("/")[-1].split("_")[0]
            perf_bpmll[model] = json.load(f)
    
    print("\n\n", perf_bpmll)
  
    # plot rocauc for bp-mll vs binary cross entropy
    rocauc_comparison = {}
    for key, value in perf.items():
        rocauc_comparison[key] = [value["roc_auc"], perf_bpmll[key]["roc_auc"]]
    print(rocauc_comparison)

    df_rocauc_comparison = pd.DataFrame.from_dict(rocauc_comparison, orient="index").rename(columns={0:"Binary Crossentropy Loss", 1:"BP-MLL Loss"})
    print(df_rocauc_comparison)

    fig = px.bar(
        data_frame=df_rocauc_comparison,
        barmode="group",
        labels={"index":"Model", "value":"ROC AUC"},
        color_discrete_sequence=px.colors.qualitative.Safe,
        title='ROC AUC Performance',
        width=1200,
        height=600)
    fig.show()

    # plot macro- and micro-averaged f1, precision, recall for bp-mll vs binary cross entropy
    sub_titles = ["F1-macro", "F1-micro", "Prec-macro", "Prec-micro", "Rec-macro", "Rec-micro"]
    fig = make_subplots(rows=6, cols=1, subplot_titles=sub_titles)

    metrics = ["f1_macro", "f1_macro", "prec_macro", "prec_micro", "rec_macro", "rec_micro"]
    for i, metric in enumerate(metrics):

        metrics_bce = []
        metrics_bpmll = []
        models = []
        for key, value in perf.items():
            metrics_bce.append(value[metric])
            metrics_bpmll.append(perf_bpmll[key][metric])
            models.append(key)

        fig.add_trace(go.Bar(name='BCE Loss', x=models, y=metrics_bce),
                      row=i + 1, col=1)
        fig.add_trace(go.Bar(name='BP-MLL Loss', x=models, y=metrics_bpmll),
                      row=i + 1, col=1)

    fig.show()


if __name__ == "__main__":

    import pandas as pd


    plot_fine_tuned_models_perf()