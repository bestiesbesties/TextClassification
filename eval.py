import os
import json
from importlib import reload
import pandas as pd

from model import Model

config = json.load(open("config.json", "r"))["config"]
embedding_model_name = "bert-base-uncased"
embedding_model_path = config["model_mapping"][embedding_model_name]
preloads = json.load(open(os.path.join("eval", "preloads", "at_2025-03-01_14-57_preloads.json"), "r"))["preloads"]

classification_model = Model(
    model_path=embedding_model_path,
    config=config,
    preloads=preloads
    )

test_data = pd.read_csv(os.path.join("eval", "data","test_data.csv"))[["Resume_str", "Category"]]

size = 25
sampled_test_data = test_data.sample(size, ignore_index=True)
sampled_test_data["Prediction"] = sampled_test_data["Resume_str"].apply(lambda x: classification_model.predict(x))
sampled_test_data