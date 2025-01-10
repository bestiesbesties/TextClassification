from transformers import BertModel

def embed(embedding_model:BertModel, tokens:dict) -> None:
    return embedding_model.forward(
        input_ids=tokens["input_ids"], 
        attention_mask=tokens["attention_mask"]
        )