import os
import json

from model import Model

config = json.load(open("config.json", "r"))["config"]
embedding_model_name = "bert-base-uncased"
embedding_model_path = config["model_mapping"][embedding_model_name]

classification_model = Model(
    model_path=embedding_model_path,
    config=config, 
    )

classification_model.fit(os.path.join("eval", "sector_descs"), save_to=os.path.join("eval", "preloads"))