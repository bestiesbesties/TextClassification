import os
import json

from model import model

config = json.load(open("config.json", "r"))["config"]
embedding_model_name = "bert-base-uncased"
embedding_model_path = config["model_mapping"][embedding_model_name]

classification_model = model.Model(
    model_path=embedding_model_path,
    config=config["sectors"],
    sectors=["tech", "healthcare", "construction"], 
    )

classification_model.fit(os.path.join("files"))