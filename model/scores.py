import math
import numpy as np
import faiss

def cosine_similarity(v1:list, v2:list) -> float:
    """Computes a cosine similarity score between two vectors.

    Args:
        v1 (list): First vector containing word embeddings.
        v2 (list): Second vector containing word embeddings.

    Returns:
        float: Cosine similarity score.
    """

    dot_product = sum(x * y for x, y in zip(v1, v2))

    magnitude_v1 = math.sqrt(sum(x**2 for x in v1))
    magnitude_v2 = math.sqrt(sum(x**2 for x in v2))

    return dot_product / (magnitude_v1 * magnitude_v2)

def __get_all_embeddings(preloads:dict) -> np.array:
    """Loads all preloaded embeddings for specific sector.

    Args:
        config (dict): Config file as a dict.
        preloads (dict): Preloaded embeddings as a dict.
        embedding_model_path(str): Path to the embedding model.

    Returns:
        np.array: Array of embeddings.
    """
    all_emb = []
    for sector in preloads["sectors"]:
        all_emb.append(preloads["data"][sector]["embeddings"])
    return np.array(all_emb)

def export_score(scores_dict:dict, faiss_scores:dict=None) -> str:
    """Creates dict with final scores by conditionally combining scores.

    Args:
        scores_dict (dict): Dictionary with standard cosine similarity and keyword overlap scores.
        faiss_scores (dict, optional): Dictionary with FAISS scores.

    Returns:
        dict: Final integer score per sector.
    """
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

    best_sector_name = ""
    best_sector_score = 0
    for sector_name, sector_score in end_scores.items():
        if sector_score > best_sector_score:
            best_sector_name = sector_name
            best_sector_score = sector_score
    # return best_sector_name, best_sector_score
    return best_sector_name

def faiss_similarity(pdf_embedding:np.array, config:dict, preloads:dict, embedding_model_path:str) -> dict:
    """Creates FAISS scores from embedding.

    Args:
        pdf_embedding (np.array): Vector containing word embeddings.
        config (dict): Config file as a dict.
        preloads (dict): Preloaded embeddings as a dict.
        embedding_model_path(str): Path to the embedding model.

    Returns:
        dict: FAISS score mapped to sector name.
    """
    all_embeddings = __get_all_embeddings(preloads)
    
    output_dimmensions = config["model_output_dimmensions_mapping"][embedding_model_path]

    index = faiss.IndexFlatL2(output_dimmensions)
    index.add(all_embeddings)
    distances, indeces = index.search(np.array([pdf_embedding]), len(preloads["sectors"]))

    faiss_scores = {}
    for distance, indence in zip(distances[0].tolist(), indeces[0].tolist()):
        faiss_scores[preloads["sectors"][indence]] = distance

    return faiss_scores

