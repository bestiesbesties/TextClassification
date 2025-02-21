import numpy as np
from transformers import AutoTokenizer, AutoModel

class Model:
    def __init__(self, model_name):
        self.tokenizer_model = AutoTokenizer.from_pretrained(model_name)
        self.embedding_model = AutoModel.from_pretrained(model_name)

    def tokenize(self, text:str) -> any:
        return self.tokenizer_model(
            text=text,
            return_tensors = "pt",
            padding = "max_length",
            truncation = True, 
            max_length = 512
        )
    
    def embed(self, tokens:dict) -> dict:
        return self.embedding_model.forward(
            input_ids=tokens["input_ids"], 
            attention_mask=tokens["attention_mask"]
        )

    def run(self, text:str) -> np.array:
        tokens = self.tokenize(text)
        embeddings = self.embed(tokens)
        ## Make 1 dimensional numpdisy from embedding_model output
        last_hidden_state = embeddings.last_hidden_state[:,0,:]
        return last_hidden_state.detach().numpy().squeeze()
    