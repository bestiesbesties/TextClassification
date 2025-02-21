import os
import json

from lib import calculate
with open("config.json", "r") as file:
    config = json.load(file)["config"]

sectors = config["sectors"]
models = list(config["model_mapping"].values())
dict_to_save = {}

for model in models:
    model_dict = {}
    for sector in sectors:
        
        filepath = os.path.join("files", sector+".pdf")
        sector_embeddings, sector_keywords = calculate.calculate_pdf(filepath, model)

        model_dict[sector] = {
            "embeddings" : sector_embeddings.tolist(),
            "keywords" : list(sector_keywords)
            }
        
        dict_to_save[model] = model_dict

with open(os.path.join("app", "data", "preloads.json"), "w") as file:
    json.dump({"preloads" : dict_to_save}, file)
