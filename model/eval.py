import os
import json
import pandas as pd

def calculate_accuracy(correct_predictions:int, total_predictions:int) -> int:
    return float(round(correct_predictions / total_predictions, 2))

def __metric_precision(tp:int, fp:int) -> float:
    if (denominator := tp + fp) == 0: ## Check for division with 0
        return 0.0
    return float(round(tp / denominator, 2))

def __metric_recall(tp:int, fn:int) -> float:
    if (denominator := tp + fn) == 0:
        return 0.0
    return float(round(tp / denominator, 2))

def __metric_f1(precision:float, recall:float) -> float:
    if (denominator := precision + recall) == 0:
        return 0.0
    return float(round((2 * (precision * recall)) / denominator, 2))

def calculate_label_scores(df:pd.DataFrame, label:str) -> list:
    tp = int(((df["Prediction"] == label) & (df["Label"] == label)).sum())  ## Correctly predicted as label
    fp = int(((df["Prediction"] == label) & (df["Label"] != label)).sum())  ## Uncorrectly predicted as label
    tn = int(((df["Prediction"] != label) & (df["Label"] != label)).sum())  ## Correctly predicted as different
    fn = int(((df["Prediction"] != label) & (df["Label"] == label)).sum())  ## Uncorrectly predicted as different

    precision = __metric_precision(tp=tp, fp=fp)
    recall = __metric_recall(tp=tp, fn=fn)
    f1 = __metric_f1(precision, recall)
    total = int(sum([tp, fp, fn])) ## Amount of predictions for label

    return {
        "tp":tp, "fp":fp,"tn":tn,"fn":fn,
        "precision":precision,
        "recall":recall,
        "f1":f1,
        "total":total
        }

def __macro_average(values:list) -> float:
    return float(round(sum(values) / len(values),2))

def __calculate_macro_averages(scores:dict) -> tuple:
    macro_precision = __macro_average([scores[key]["precision"] for key in scores.keys()])
    macro_recall = __macro_average([scores[key]["recall"] for key in scores.keys()])
    macro_f1 = __macro_average([scores[key]["f1"] for key in scores.keys()])

    return macro_precision, macro_recall, macro_f1

def __calculate_micro_averages(scores:dict) -> tuple:
    print(scores)
    total_tp = sum([scores[key]["tp"] for key in scores.keys()])
    total_fp = sum([scores[key]["fp"] for key in scores.keys()])
    total_fn = sum([scores[key]["fn"] for key in scores.keys()])

    micro_precision = __metric_precision(total_tp, total_fp)
    micro_recall = __metric_recall(total_tp, total_fn)
    micro_f1 = __metric_f1(micro_precision, micro_recall)

    return micro_precision, micro_recall, micro_f1

def __weighted_average(values:list, weights:list) -> float:
    acum = 0
    for value, weight in zip(values, weights):
        acum += value * weight
    return float(round(acum / sum(weights),2))

def __calculate_weighted_averages(scores:dict) -> tuple:
    weights = [scores[key]["total"] for key in scores.keys()]

    weighted_precision = __weighted_average([scores[key]["precision"] for key in scores.keys()], weights)
    weighted_recall = __weighted_average([scores[key]["recall"] for key in scores.keys()], weights)
    weighted_f1 = __weighted_average([scores[key]["f1"] for key in scores.keys()], weights)

    return weighted_precision, weighted_recall, weighted_f1

def calculate_averages(scores:dict) -> dict:
    return {
        "Metric": ["Precision", "Recall", "F1"],
        "Macro" : [*__calculate_macro_averages(scores)], ## * splat operator unpacks recieved tuple
        "Micro" : [*__calculate_micro_averages(scores)],
        "Weighted" : [*__calculate_weighted_averages(scores)]
    }