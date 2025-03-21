from model.evaluation import metrics

def macro(values: list) -> float:
    """Calculates the average of a list, rounded to two decimal places.

    Args:
        values (list): List of numerical values.

    Returns:
        float: Rounded average.
    """
    return float(round(sum(values) / len(values),2))

def macro_averages(scores: dict) -> tuple:
    """Calculates macro-averaged precision, recall, and F1-score.

    Args:
        scores (dict): Dictionary containing precision, recall, and F1-score for each key.

    Returns:
        tuple: Macro-averaged (precision, recall, F1-score).
    """
    macro_precision = macro([scores[key]["precision"] for key in scores.keys()])
    macro_recall = macro([scores[key]["recall"] for key in scores.keys()])
    macro_f1 = macro([scores[key]["f1"] for key in scores.keys()])
    return macro_precision, macro_recall, macro_f1

def micro_averages(scores: dict) -> tuple:
    """Calculates micro-averaged precision, recall, and F1-score.

    Args:
        scores (dict): Dictionary containing true positives (tp), false positives (fp), and false negatives (fn) for each key.

    Returns:
        tuple: Micro-averaged (precision, recall, F1-score).
    """
    total_tp = sum([scores[key]["tp"] for key in scores.keys()])
    total_fp = sum([scores[key]["fp"] for key in scores.keys()])
    total_fn = sum([scores[key]["fn"] for key in scores.keys()])

    micro_precision = metrics.precision(total_tp, total_fp)
    micro_recall = metrics.recall(total_tp, total_fn)
    micro_f1 = metrics.f1(micro_precision, micro_recall)
    return micro_precision, micro_recall, micro_f1


def weighted(values: list, weights: list) -> float:
    """Calculates the weighted average of a list of values.

    Args:
        values (list): List of numerical values.
        weights (list): Corresponding weights for each value.

    Returns:
        float: Weighted average rounded to two decimal places.
    """
    acum = 0
    for value, weight in zip(values, weights):
        acum += value * weight
    return float(round(acum / sum(weights),2))

def weighted_averages(scores: dict) -> tuple:
    """Calculates weighted-averaged precision, recall, and F1-score.

    Args:
        scores (dict): Dictionary containing precision, recall, F1-score, and total count for each key.

    Returns:
        tuple: Weighted-averaged (precision, recall, F1-score).
    """
    weights = [scores[key]["total"] for key in scores.keys()]

    weighted_precision = weighted([scores[key]["precision"] for key in scores.keys()], weights)
    weighted_recall = weighted([scores[key]["recall"] for key in scores.keys()], weights)
    weighted_f1 = weighted([scores[key]["f1"] for key in scores.keys()], weights)
    return weighted_precision, weighted_recall, weighted_f1