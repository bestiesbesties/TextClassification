import os
import json
import pandas as pd

from model import Model
from model.evaluation import batch

## Configurate model variables
duration = None
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
df_predict, duration = batch.predict_df(classification_model, test_data, size=None)
df_predict = batch.get_correct(df_predict)
df_predict.to_csv(os.path.join("eval", "preloads", f"{run}_data.csv"))

## Evaluate & save results to .txt
batch.evaluate(df_predict, run, duration)
