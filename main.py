import os
import logging
import argparse
from importlib import reload
import json

## TODO sentence transformers gebruiken
from sentence_transformers import SentenceTransformer, util

from cvmatching import calculate
with open("config.json", "r") as file:
    config = json.load(file)["config"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "filepath", 
    type=str, 
    help="The rich words document filepath to process and stdout the calculations for."
)
namespace = parser.parse_args()


pdf_embedding, pdf_keywords = calculate.calculate_pdf(namespace.filepath)

with open(os.path.join("app", "data", "preloads.json"), "r") as file:
    preloads = json.load(file)["preloads"]

scores = {}
for sector in config["sectors"]:
    sector_embeddings = preloads[sector]["embeddings"]
    sector_keywords = preloads[sector]["keywords"]

    # Berekent de cos() van de angle tussen beide vectoren in de vectorruimte
    cosine_similarity = util.cos_sim(pdf_embedding.tolist(), sector_embeddings).item()

    # Aantal woorden in pdf die belangrijk zijn voor sector
    keyword_overlap = len(pdf_keywords.intersection(sector_keywords))
    
    # print(sector, "cosine_similarity: ", cosine_similarity, "keyword_score: ", keyword_overlap / (len(sector_keywords) / 2))
    score = 0.6 * cosine_similarity + 0.4 * (keyword_overlap / (len(sector_keywords) / 3)) ## / 3 weghalen als lemmitization
    scores[sector] = str(round(score * 100, 0))

print(json.dumps(scores))
