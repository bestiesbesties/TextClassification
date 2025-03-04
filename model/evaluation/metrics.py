def accuracy(correct_predictions:int, total_predictions:int) -> int:
    if total_predictions == 0: ## Check for division with 0
        return 0.0
    return float(round(correct_predictions / total_predictions, 2))

def precision(tp:int, fp:int) -> float:
    if (denominator := tp + fp) == 0:
        return 0.0
    return float(round(tp / denominator, 2))

def recall(tp:int, fn:int) -> float:
    if (denominator := tp + fn) == 0:
        return 0.0
    return float(round(tp / denominator, 2))

def f1(precision:float, recall:float) -> float:
    if (denominator := precision + recall) == 0:
        return 0.0
    return float(round((2 * (precision * recall)) / denominator, 2))