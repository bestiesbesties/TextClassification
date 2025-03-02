import os
import time
import json
import pandas as pd

from model import Model, eval

def predict_df(df:pd.DataFrame, size:int=None) -> pd.DataFrame:
    if size:
        df = df.sample(size, ignore_index=True)

    start = time.time()
    df["Prediction"] = df["Resume_str"].apply(lambda x: ( result := classification_model.predict(x), print(result))[0]) ## := Lets us assign variable inside 
    end = time.time()

    duration = end - start
    minutes = int(duration // 60) ## Amount of entire divisions
    seconds = int(duration % 60) ## Remainder of entire divisions
    print(f"duration: {minutes} minutes and {seconds} seconds")

    return df

def get_correct(df:pd.DataFrame) -> pd.DataFrame:
    df["Correct"] = df["Label"] == df["Prediction"]
    return df

def evaluate(df:pd.DataFrame, title:str):
    df = df[["Label", "Prediction", "Correct"]]
    accuracy = eval.calculate_accuracy(int(df["Correct"].sum()), len(df))
    label_scores = {label : eval.calculate_label_scores(df, label) for label in list(df["Label"].unique())}
    evaluation = pd.DataFrame(data=eval.calculate_averages(label_scores)).to_string()

    with open(os.path.join("eval", "preloads", f"{title}_evaluation.txt"), "w") as file:
        file.write(f"Evaluation of model '{title}' \n\n\n")
        file.write(f"Accuracy: {accuracy} \n\n")
        file.write(evaluation)
        file.write("\n\n")
        file.write(json.dumps(label_scores, indent=4))

## Configurate model variables
config = json.load(open("config.json", "r"))["config"]
embedding_model_name = "bert-base-uncased"
embedding_model_path = config["model_mapping"][embedding_model_name]
run = "at_2025-03-01_17-25"
preloads = json.load(open(os.path.join("eval", "preloads", f"{run}_preloads.json"), "r"))["preloads"]

## Instantiate model
classification_model = Model(
    model_path=embedding_model_path,
    config=config,
    preloads=preloads
    )

## Load cleaned test data
test_data = pd.read_csv(os.path.join("eval", "data","test_data.csv"))[["Resume_str", "Label"]]
## Inference on batch & save results to .csv
df_predict = get_correct(predict_df(test_data, None))
df_predict.to_csv(os.path.join("eval", "preloads", f"{run}_data.csv"))

## Evaluate & save results to .txt
evaluate(df_predict, run)
