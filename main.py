import os
import logging
import argparse
from importlib import reload
import json
import numpy as np
from sentence_transformers import util
import faiss

from cvmatching import calculate

config = json.load(open("config.json", "r"))["config"]

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

# file_path = os.path.join("files", "cvs", "cv1.pdf")
# model = "bert-base-uncased"
# embedding_model_path = config["model_mapping"][model]

pdf_embedding, pdf_keywords = calculate.calculate_pdf(file_path, embedding_model_path)

preloads = json.load(open(os.path.join("app", "data", "preloads.json"), "r"))["preloads"]

scores = {}
all_emb = []
for sector in config["sectors"]:

    ## Preloaded embeddings van aangegeven model inladen
    sector_embeddings = preloads[embedding_model_path][sector]["embeddings"]
    all_emb.append(sector_embeddings)
    sector_keywords = preloads[embedding_model_path][sector]["keywords"]

    # Berekent de cos() van de angle tussen beide vectoren in de vectorruimte
    cosine_similarity = util.cos_sim(pdf_embedding.tolist(), sector_embeddings).item()

    # Aantal woorden in pdf die belangrijk zijn voor sector
    keyword_overlap = len(pdf_keywords.intersection(sector_keywords))
    adjust = int(len(sector_keywords) / 3)
    keyword_overlap_adjusted = keyword_overlap / adjust
    
    scores[sector] = {}
    scores[sector]["cosine_similarity"] = round(cosine_similarity, 2)
    scores[sector]["keyword_overlap_adjusted"] = round(keyword_overlap_adjusted, 2)

all_emb_np = np.array(all_emb)

model_embedding_dimmensions = config["model_embedding_dimmensions_mapping"][model]
index = faiss.IndexFlatL2(model_embedding_dimmensions)
index.add(all_emb_np)
distances, indeces = index.search(np.array([pdf_embedding]), len(config["sectors"]))

distances_list = distances[0].tolist()
indeces_list = indeces[0].tolist()
for x in zip(distances_list, indeces_list):
    current = [x[0], x[1]] 
    current.append(config["sectors"][x[1]])
    print(current)
    scores[current[2]]["inverse_faiss_similarity"] = round(1 - current[0] * 0.01 ,2)

## cosine_similairy higher better
## keyword_overlap higher better
## inverse_faiss_similarity higher better

end_scores = {}
for sector in scores:
    accum = 0 
    for y in scores[sector].values():
        accum += 0.33 * y
    end_scores[sector] = round(accum * 100)


print(json.dumps(end_scores))