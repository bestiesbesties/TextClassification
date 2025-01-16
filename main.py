import os
import logging
import argparse
from importlib import reload
import json
from sentence_transformers import util

from cvmatching import calculate
with open("config.json", "r") as file:
    config = json.load(file)["config"]

parser = argparse.ArgumentParser()

parser.add_argument(
    "file_path", 
    type=str, 
    help="The rich words document filepath to process and stdout the calculations for."
)
parser.add_argument(
    "embedding_model_name", 
    type=str, 
    help="The name of the embedding model of which the embeddings have to be generated with."
)
namespace = parser.parse_args()
file_path = namespace.file_path
embedding_model_path = config["model_mapping"][namespace.embedding_model_name]

pdf_embedding, pdf_keywords = calculate.calculate_pdf(file_path, embedding_model_path)

with open(os.path.join("app", "data", "preloads.json"), "r") as file:
    preloads = json.load(file)["preloads"]

scores = {}
for sector in config["sectors"]:

    ## Preloaded embeddings van aangegeven model inladen
    sector_embeddings = preloads[embedding_model_path][sector]["embeddings"]
    sector_keywords = preloads[embedding_model_path][sector]["keywords"]

    # Berekent de cos() van de angle tussen beide vectoren in de vectorruimte
    cosine_similarity = util.cos_sim(pdf_embedding.tolist(), sector_embeddings).item()

    # Aantal woorden in pdf die belangrijk zijn voor sector
    keyword_overlap = len(pdf_keywords.intersection(sector_keywords))
    
    # print(sector, "cosine_similarity: ", cosine_similarity, "keyword_score: ", keyword_overlap / (len(sector_keywords) / 2))
    score = 0.5 * cosine_similarity + 0.5 * (keyword_overlap / (len(sector_keywords) / 3)) ## / 3 weghalen als lemmitization
    scores[sector] = str(round(score * 100, 0))

print(json.dumps(scores))
