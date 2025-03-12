import os
from datetime import datetime
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

from lib import parser, filehandler
from model import chars, words, scores

class Model:
    def __init__(self, model_path:str, config:dict, preloads:dict|None=None):
        self.config = config
        self.model_path = str(model_path)
        self.tokenizer_model = AutoTokenizer.from_pretrained(model_path)
        self.embedding_model = AutoModel.from_pretrained(model_path)
        self.pad_token_id = self.tokenizer_model.pad_token_id
        self.MID = int(config["model_input_dimmensions_mapping"][self.model_path])
        ## Load trained data
        self.preloads = preloads

    
    def __get_padded_chunk(self, chunk:list) -> torch.tensor:
        padding_length = self.MID - len(chunk)
        # Pad to specific lenght (Model Input Dimmension), if the chunk is smaller than MID
        chunk_padded = chunk + [self.pad_token_id] * padding_length
        return torch.tensor(chunk_padded)

    def __tokenize(self, text:str) -> tuple[torch.tensor, torch.tensor]:
        tokenizer_output = self.tokenizer_model(text=text)

        ## Split tokenizer output into tokens and attentions
        tokens = tokenizer_output["input_ids"]
        attentions = tokenizer_output["attention_mask"]

        ## Conditionally split tokens & attentions into smaller lists
        # if len(tokens) < self.MID:
        #     chunks_tokens = tokens
        #     chunks_attentions = attentions

        if len(tokens) == len(attentions):
            chunks_tokens = [tokens[i:i+self.MID] for i in range(0, len(tokens), self.MID)]
            chunks_attentions = [attentions[i:i+self.MID] for i in range(0, len(attentions), self.MID)]
        
        else:
            raise ValueError("tokens and attentions are not of same size")

        ## Proper batch tensor using torch.stack()
        padded_chunks_tokens = torch.stack([self.__get_padded_chunk(chunk) for chunk in chunks_tokens])
        padded_chunks_attentions = torch.stack([self.__get_padded_chunk(chunk) for chunk in chunks_attentions])
        
        return padded_chunks_tokens, padded_chunks_attentions
    
    def __embed(self, tokens:torch.tensor, attentions:torch.tensor) -> dict:
        return self.embedding_model(
            input_ids=tokens, 
            attention_mask=attentions
        )

    def __get_embbeding(self, text:str) -> np.array:
        tokens, attentions = self.__tokenize(text)
        embeddings = self.__embed(tokens, attentions)
        ## Using mean to 1 dim (combined) instead of last_hidden_state (individual dim)
        last_hidden_state_mean  = embeddings.last_hidden_state.mean(dim=(0,1))
        ## Required detaching to stop gradient tracking, saves memory
        last_hidden_state_mean_list = last_hidden_state_mean.detach().cpu().numpy().flatten().tolist()
        return last_hidden_state_mean_list


    def __calculate(self, text:str):
        ## .1 clean text
        text_cleaned = chars.clean(text)
        ## .2 filter for stopwords
        text_filtered = words.remove_stopwords(words.make_doc(text_cleaned))
        ## .3 create embeddings & keywords
        doc_embedding = self.__get_embbeding(text_filtered)
        doc_keywords = words.extract_keywords(words.make_doc(text_filtered))
        return doc_embedding, doc_keywords

    def __classification(self, doc_embedding:list, doc_keywords:set) -> dict:
        scores_dict = {}
        # print("len(doc_embedding): ", len(doc_embedding))
        for sector in self.preloads["sectors"]:

            ## Preloaded data respective to requested model
            sector_embeddings = self.preloads["data"][sector]["embeddings"]
            sector_keywords = self.preloads["data"][sector]["keywords"]

            # Calculate cos() of the angle between both vectors in N dimensional space
            cosine_similarity = scores.cosine_similarity(doc_embedding, sector_embeddings)

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
                "embeddings" : sector_embeddings,
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
        # print("len(doc_embedding): ", len(doc_embedding))
        best_combined_score = self.__classification(doc_embedding, doc_keywords)
        return best_combined_score



    