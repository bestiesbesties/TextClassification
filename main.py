import os
import logging
from importlib import reload
import json

from cvmatching import calculate, arguments, scores

config = json.load(open("config.json", "r"))["config"]
preloads = json.load(open(os.path.join("app", "data", "preloads.json"), "r"))["preloads"]

parser = arguments.create_parser()
namespace = parser.parse_args()
file_path = namespace.file_path
embedding_model_name = namespace.embedding_model_name
embedding_model_path = config["model_mapping"][embedding_model_name]
use_faiss = True if namespace.use_faiss == "True" else False

pdf_embedding, pdf_keywords = calculate.calculate_pdf(file_path, embedding_model_path)

scores_dict = {}
for sector in config["sectors"]:

    ## Preloaded embeddings van aangegeven model inladen
    sector_embeddings = preloads[embedding_model_path][sector]["embeddings"]
    sector_keywords = preloads[embedding_model_path][sector]["keywords"]

    # Berekent de cos() van de angle tussen beide vectoren in de vectorruimte
    cosine_similarity = scores.cosine_similarity(pdf_embedding.tolist(), sector_embeddings)

    # Aantal woorden in pdf die belangrijk zijn voor sector
    keyword_overlap = len(pdf_keywords.intersection(sector_keywords))
    adjust = int(len(sector_keywords) / 3)
    keyword_overlap_adjusted = keyword_overlap / adjust
    
    scores_dict[sector] = {}
    scores_dict[sector]["cosine_similarity"] = round(cosine_similarity, 2)
    scores_dict[sector]["keyword_overlap_adjusted"] = round(keyword_overlap_adjusted, 2)

print(json.dumps(scores.export_scores(scores_dict)))