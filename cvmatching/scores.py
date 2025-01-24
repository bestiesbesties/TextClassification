import math
import numpy as np
import faiss

def cosine_similarity(v1:list, v2:list) -> float:
    dot_product = sum(x * y for x, y in zip(v1, v2))

    magnitude_v1 = math.sqrt(sum(x**2 for x in v1))
    magnitude_v2 = math.sqrt(sum(x**2 for x in v2))

    return dot_product / (magnitude_v1 * magnitude_v2)

def __get_all_embeddings(config:dict, preloads:dict, embedding_model_path) -> np.array:
    all_emb = []
    for sector in config["sectors"]:
        all_emb.append(preloads[embedding_model_path][sector]["embeddings"])
    return np.array(all_emb)

def export_scores(scores_dict:dict, faiss_scores:dict=None) -> dict:
    end_scores = {}

    for key in scores_dict:
        cs = scores_dict[key]["cosine_similarity"]
        kwa = scores_dict[key]["keyword_overlap_adjusted"]
        if faiss_scores:
            ifs = (100 - faiss_scores[key]) * 0.01
            score = (0.33 * cs) + (0.33 * kwa) + (0.33 * ifs)
        else: 
            score = (0.5 * cs) + (0.5 * kwa)

        end_scores[key] = round(score * 100)
    return end_scores

def faiss_similarity(pdf_embedding:np.array, config:dict, preloads:dict, embedding_model_path:str) -> dict:
    all_embeddings = __get_all_embeddings(config, preloads, embedding_model_path)
    
    model_embedding_dimmensions = config["model_embedding_dimmensions_mapping"][embedding_model_path]
    index = faiss.IndexFlatL2(model_embedding_dimmensions)
    index.add(all_embeddings)
    distances, indeces = index.search(np.array([pdf_embedding]), len(config["sectors"]))

    faiss_scores = {}
    for distance, indence in zip(distances[0].tolist(), indeces[0].tolist()):
        faiss_scores[config["sectors"][indence]] = distance

    return faiss_scores

