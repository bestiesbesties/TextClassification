import os
import time
import json
import pandas as pd

from model import Model
from model.evaluation import calculate, metrics

def predict_df(classification_model: Model, df: pd.DataFrame, size: int | None = None) -> tuple[pd.DataFrame, str]:
    """Predicts classifications for a DataFrame and measures execution time.

    Args:
        classification_model (Model): The classification model to use for predictions.
        df (pd.DataFrame): DataFrame containing text data for prediction.
        size (int | None, optional): Number of samples to predict. Defaults to None (uses full DataFrame).

    Returns:
        tuple[pd.DataFrame, str]: DataFrame with predictions and execution time as a string.
    """
    if size:
        df = df.sample(size, ignore_index=True)

    start = time.time()
    df["Prediction"] = df["Resume_str"].apply(lambda x: ( result := classification_model.predict(x), print(result))[0]) ## := Lets us assign variable inside 
    end = time.time()

    duration = end - start
    minutes = int(duration // 60) ## Amount of entire divisions
    seconds = int(duration % 60) ## Remainder of entire divisions
    duration_str = f"{minutes} minutes and {seconds} seconds"
    return df, duration_str

def get_correct(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a correctness column to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing actual labels and predictions.

    Returns:
        pd.DataFrame: DataFrame with an additional 'Correct' column indicating if the prediction matches the label.
    """
    df["Correct"] = df["Label"] == df["Prediction"]
    return df

def evaluate(df:pd.DataFrame, title:str, duration:str="No data") -> None:
    """Evaluates classification performance and saves results to a file.

    Args:
        df (pd.DataFrame): DataFrame containing actual labels, predictions, and correctness.
        title (str): Title for the evaluation results.
        duration (str, optional): Time taken for predictions. Defaults to "No data".
    """
    df = df[["Label", "Prediction", "Correct"]]
    accuracy = metrics.accuracy(int(df["Correct"].sum()), len(df))
    label_scores = {label : calculate.label_scores(df, label) for label in list(df["Label"].unique())}
    evaluation = pd.DataFrame(data=calculate.average_scores(label_scores)).to_string(index=False)
    cf_matrix = calculate.cf_matrix(df).to_string()

    with open(os.path.join("eval", "preloads", f"{title}_evaluation.txt"), "w") as file:
        file.write(f"Evaluation of model '{title}'")
        file.write("\n\n-- General evalutation --\n")
        file.write(f"Duration: {duration}\nAccuracy: {accuracy}")
        file.write("\n\n-- General scores --\n")
        file.write(evaluation)
        file.write("\n\n-- Confusion matrix --\n")
        file.write(cf_matrix)
        file.write("\n\n-- Source data--\n")
        file.write(json.dumps(label_scores, indent=4))