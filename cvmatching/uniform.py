import numpy as np
from transformers import AutoTokenizer, AutoModel

def uniform_run(text:str, model_name:str) -> np.array:

    tokenizer_model = AutoTokenizer.from_pretrained(model_name)
    embedding_model = AutoModel.from_pretrained(model_name)

    tokens = __tokenize(tokenizer_model, text)
    embeddings = __embed(embedding_model, tokens)

    ## Make 1 dimensional numpdisy from embedding_model output
    last_hidden_state = embeddings.last_hidden_state[:,0,:]
    return last_hidden_state.detach().numpy().squeeze()

def __tokenize(tokenizer_model, text:str) -> any:
    return tokenizer_model(
        text=text,
        return_tensors = "pt",
        padding = "max_length",
        truncation = True, 
        max_length = 512
        )

def __embed(embedding_model, tokens:dict) -> any:
    return embedding_model.forward(
        input_ids=tokens["input_ids"], 
        attention_mask=tokens["attention_mask"]
        )
