import pandas as pd

from model.evaluation import metrics, averages

def label_scores(df:pd.DataFrame, label:str) -> list:
    tp = int(((df["Prediction"] == label) & (df["Label"] == label)).sum())  ## Correctly predicted as label
    fp = int(((df["Prediction"] == label) & (df["Label"] != label)).sum())  ## Uncorrectly predicted as label
    tn = int(((df["Prediction"] != label) & (df["Label"] != label)).sum())  ## Correctly predicted as different
    fn = int(((df["Prediction"] != label) & (df["Label"] == label)).sum())  ## Uncorrectly predicted as different

    precision = metrics.precision(tp=tp, fp=fp)
    recall = metrics.recall(tp=tp, fn=fn)
    f1 = metrics.f1(precision, recall)
    total = int(sum([tp, fp, fn])) ## Amount of predictions for label

    return {
        "tp":tp, "fp":fp,"tn":tn,"fn":fn,
        "precision":precision,
        "recall":recall,
        "f1":f1,
        "total":total
        }

def average_scores(scores:dict) -> dict:
    return {
        "Metric": ["Precision", "Recall", "F1"],
        "Macro" : [*averages.macro_averages(scores)], ## * splat operator unpacks recieved tuple
        "Micro" : [*averages.micro_averages(scores)],
        "Weighted" : [*averages.weighted_averages(scores)]
    }

def cf_matrix(df:pd.DataFrame):
    df = df[["Label", "Prediction"]]
    labels = list(df["Label"].unique())
    cf_matrix = {true: {pred: 0 for pred in labels} for true in labels} ## sqaure matrix on true and pred

    for _, row in df.iterrows():
        if row["Prediction"] not in labels:
            continue
        cf_matrix[row["Label"]][row["Prediction"]] += 1

    ## Transpose beacause "If data is a dict, column order follows insertion-order.".
    ## Therefore true's would be columns (vertical) instead of rows (horizontal).
    cf_matrix_df = pd.DataFrame(data=cf_matrix).T

    return cf_matrix_df