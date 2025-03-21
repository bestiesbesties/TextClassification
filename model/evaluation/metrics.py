def accuracy(correct_predictions:int, total_predictions:int) -> int:
    """
    Calculates the accuracy given the number of correct and total predictions.

    Args:
        correct_predictions (int): The number of correctly predicted values.
        total_predictions (int): The total number of predictions made.

    Returns:
        float: The accuracy as a decimal value.
    """
    if total_predictions == 0: ## Check for division with 0
        return 0.0
    return float(round(correct_predictions / total_predictions, 2))

def precision(tp:int, fp:int) -> float:
    """
    Calculates precision given the number of true positives and false positives.

    Args:
        tp (int): The number of true positive predictions.
        fp (int): The number of false positive predictions.

    Returns:
        float: The precision value, rounded to two decimal places.
    """
    if (denominator := tp + fp) == 0:
        return 0.0
    return float(round(tp / denominator, 2))

def recall(tp:int, fn:int) -> float:
    """
    Calculates recall given the number of true positives and false negatives.

    Args:
        tp (int): The number of true positive predictions.
        fn (int): The number of false negative predictions.

    Returns:
        float: The recall value, rounded to two decimal places.
    """
    if (denominator := tp + fn) == 0:
        return 0.0
    return float(round(tp / denominator, 2))

def f1(precision:float, recall:float) -> float:
    """
    Calculates the F1 score given precision and recall values.

    Args:
        precision (float): The precision value.
        recall (float): The recall value.

    Returns:
        float: The F1 score, rounded to two decimal places.
    """
    if (denominator := precision + recall) == 0:
        return 0.0
    return float(round((2 * (precision * recall)) / denominator, 2))