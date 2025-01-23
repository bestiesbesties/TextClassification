import math
import numpy as np

def cosine_similarity(v1:list, v2:list) -> float:
    dot_product = sum(x * y for x, y in zip(v1, v2))

    magnitude_v1 = math.sqrt(sum(x**2 for x in v1))
    magnitude_v2 = math.sqrt(sum(x**2 for x in v2))

    return dot_product / (magnitude_v1 * magnitude_v2)

# def __get_all_embeddings(config:dict, preloads:dict, embedding_model_path) -> np.array:
#     all_emb = []
#     for sector in config["sectors"]:
#         all_emb.append(preloads[embedding_model_path][sector]["embeddings"])
#     return np.array(all_emb)

def export_scores(scores_dict:dict, fais_output:tuple[list, list]=None) -> dict:
    if fais_output:
        print("FUNCTION NOT WRITTEN")
    else: 
        end_scores = {}
        for key in scores_dict:
            accum = 0 
            for y in scores_dict[key].values():
                accum += 0.5 * y
            end_scores[key] = round(accum * 100)
    return end_scores

# def faiss_similarity(pdf_embedding, config:dict, preloads:dict, embedding_model_path) -> tuple[list, list]:
#     import faiss
#     all_embeddings = __get_all_embeddings(config, preloads, embedding_model_path)
    
#     model_embedding_dimmensions = config["model_embedding_dimmensions_mapping"][embedding_model_path]
#     index = faiss.IndexFlatL2(model_embedding_dimmensions)
#     index.add(all_embeddings)
#     distances, indeces = index.search(np.array([pdf_embedding]), len(config["sectors"]))

#     return (distances[0].tolist(), indeces[0].tolist())

