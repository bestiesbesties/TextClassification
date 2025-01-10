from transformers import BertTokenizer

def tokenize(tokenizer_model:BertTokenizer, document:str) -> str :
    return tokenizer_model(
        text=document,
        return_tensors = "pt",
        padding = "max_length",
        truncation = True, 
        max_length = 512
        )

