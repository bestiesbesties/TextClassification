import torch
from umap import UMAP


def handle_batch(documents, tokenizer_model, embedding_model, reducer):
    tokens = __tokenize_batch(documents, tokenizer_model)
    embeddings = __embed_batch(tokens, embedding_model)
    reduced = __reduce_batch(embeddings, reducer)
    return reduced

def __tokenize_batch(documents, tokenizer_model) -> torch.Tensor:
    return tokenizer_model(
        documents, 
        return_tensors = "pt",
        padding = "max_length",
        truncation = True, 
        max_length = 512
        )

def __embed_batch(tokens, embedding_model) -> torch.Tensor:
    embeddings = embedding_model(
        input_ids = tokens["input_ids"], 
        attention_mask = tokens["attention_mask"]
        )
    return embeddings.last_hidden_state[:,0,:]

def __reduce_batch(embeddings, reducer) -> torch.Tensor:
    return reducer.fit_transform(embeddings)