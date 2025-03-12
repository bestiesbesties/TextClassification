import os
from datetime import datetime
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel

from lib import parser, filehandler
from model import chars, words, scores

class Model:
    def __init__(self, model_path:str, config:dict, preloads:dict|None=None):
        self.config = config
        self.model_path = model_path
        self.tokenizer_model = AutoTokenizer.from_pretrained(model_path)
        self.embedding_model = AutoModel.from_pretrained(model_path)
        ## Load trained data
        self.preloads = preloads

    def __tokenize(self, text:str) -> any:
        return self.tokenizer_model(
            text=text,
            return_tensors = "pt",
            padding = "max_length",
            truncation = True, 
            max_length = 512
        )
    
    def __embed(self, tokens:dict) -> dict:
        return self.embedding_model.forward(
            input_ids=tokens["input_ids"], 
            attention_mask=tokens["attention_mask"]
        )

    def __get_embbeding(self, text:str) -> np.array:
        tokens = self.__tokenize(text)
        embeddings = self.__embed(tokens)
        ## Make 1 dimensional numpdisy from embedding_model output
        last_hidden_state = embeddings.last_hidden_state[:,0,:]
        return last_hidden_state.detach().numpy().squeeze()
    
    def __calculate(self, text:str):
        ## .1 clean text
        text_cleaned = chars.clean(text)
        ## .2 filter for stopwords
        text_filtered = words.remove_stopwords(words.make_doc(text_cleaned))
        ## .3 stemming
        text_filtered_stemmed = words.extract_lemmas(words.make_doc(text_filtered))
        ## .4 create embeddings & keywords
        doc_embedding = self.__get_embbeding(text_filtered)
        doc_keywords = words.extract_keywords(words.make_doc(text_filtered_stemmed))
        return doc_embedding, doc_keywords

    def __classification(self, doc_embedding:np.array, doc_keywords:set) -> dict:
        scores_dict = {}
        for sector in self.preloads["sectors"]:

            ## Preloaded data respective to requested model
            sector_embeddings = self.preloads["data"][sector]["embeddings"]
            sector_keywords = self.preloads["data"][sector]["keywords"]

            # Calculate cos() of the angle between both vectors in N dimensional space
            cosine_similarity = scores.cosine_similarity(doc_embedding.tolist(), sector_embeddings)

            # Amount of words in document which are important for sector
            keyword_overlap = len(doc_keywords.intersection(sector_keywords))
            adjust = int(len(sector_keywords) / 3)
            keyword_overlap_adjusted = keyword_overlap / adjust
            
            scores_dict[sector] = {}
            scores_dict[sector]["cosine_similarity"] = round(cosine_similarity, 2)
            scores_dict[sector]["keyword_overlap_adjusted"] = round(keyword_overlap_adjusted, 2)

        faiss_scores = scores.faiss_similarity(doc_embedding, self.config, self.preloads, self.model_path)

        best_combined_score = scores.export_score(scores_dict, faiss_scores)
        return best_combined_score
    
    def fit(self, folder_path:str, save_to:str=None):
        preloads = {}

        files = filehandler.find_files_with_extension(folder_path, ".pdf")
        preloads["sectors"] = [file.split(f"{os.sep}")[-1].replace(".pdf", "").replace("_", " ") for file in files]
        print(f"Fitting model for: {preloads["sectors"]}")

        preloads["data"] = {}
        for file, sector in zip(files, preloads["sectors"]):   
            sector_text = parser.pdf(file)
            sector_embeddings, sector_keywords = self.__calculate(sector_text)

            preloads["data"][sector] = {
                "embeddings" : sector_embeddings.tolist(),
                "keywords" : list(sector_keywords)
                }
        self.preloads = preloads

        if save_to:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M")
            save_to_loc = os.path.join(save_to, f"at_{now}_preloads.json")
            print(f"Saving preloads to {save_to_loc}")
            with open(save_to_loc, "w") as file:
                json.dump({"preloads" : preloads}, file)
         

    def predict(self, text:str) -> str:
        if not self.preloads:
            print("No preloads available. Please fit the model using the .fit() method provided with documents")
            return

        doc_embedding, doc_keywords = self.__calculate(text)
        best_combined_score = self.__classification(doc_embedding, doc_keywords)
        return best_combined_score



    