import glob
import json

import numpy as np
import pandas as pd
import plotly.express as px


def plot_fine_tuned_models_perf():

    # read performance metrics
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


if __name__ == "__main__":

    import pandas as pd


    plot_fine_tuned_models_perf()