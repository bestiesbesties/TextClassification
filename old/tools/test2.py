from transformers import BertTokenizer, BertModel
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
import torch

def iterate(document):
    ## Tokenizer step
    resume_tokens = tokenize(
        tokenizer_model=bert_tokenizer,
        document=document["resume_text"]
        )
    
    desc_tokens = tokenize(
        tokenizer_model=bert_tokenizer,
        document=document["job_description_text"]
        )

    ## Embedding step
    resume_embedding = embed(
        embedding_model=bert_model,
        input_ids=resume_tokens["input_ids"],
        attention_mask=resume_tokens["attention_mask"]
        )

    desc_embedding = embed(
        embedding_model=bert_model,
        input_ids=desc_tokens["input_ids"][:512],
        attention_mask=desc_tokens["attention_mask"][:512]
        )
    
    return resume_embedding, desc_embedding

def tokenize(tokenizer_model, document) -> torch.Tensor:
    tokens =
    ## de waardes zitten nu in een 2 dimensionale tensort met slechts 1 row
    ## .squeeze(0) Haalt dimensie '0' weg van de tensor, blijft 1 dimensie over
    # return(tokens["input_ids"].squeeze(0),tokens["attention_mask"].squeeze(0))
    return tokens

def embed(embedding_model, input_ids, attention_mask) -> torch.Tensor:
    embeddings = embedding_model(input_ids=input_ids, attention_mask=attention_mask)
    return embeddings.last_hidden_state[:,0,:]


def reduce_dimensions(embeddings) -> UMAP:
    amt_reduced_dims = int(embeddings.shape(1)/3)
    reducer = UMAP(n_neighbors=15,n_components=amt_reduced_dims)
    reduceds = reducer.fit_transform(embeddings)
    return reduceds, reducer

