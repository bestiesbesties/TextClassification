import os
import json

from cvmatching import calculate
with open("config.json", "r") as file:
    config = json.load(file)["config"]

sectors = config["sectors"]
dict_to_save = {}

for sector in sectors:
    filepath = os.path.join("files", sector+".pdf")
    sector_embeddings, sector_keywords = calculate.calculate_pdf(filepath)

    dict_to_save[sector] = {
        "embeddings" : sector_embeddings.tolist(),
        "keywords" : list(sector_keywords)
        }

with open(os.path.join("app", "data", "preloads.json"), "w") as file:
    json.dump({"preloads" : dict_to_save}, file)
