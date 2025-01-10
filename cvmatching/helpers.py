import numpy as np

from cvmatching import summarize, embed, tokenize

def state_document(tokenizer_model, embedding_model, text) -> np.array:
    summ = summarize.summarize_MiniLM(text, 20)
    tokens = tokenize.tokenize(tokenizer_model=tokenizer_model, document=summ)
    embeddings = embed.embed(embedding_model=embedding_model, tokens=tokens)

    ## Make 1 dimensional numpy from embedding_model output
    last_hidden_state = embeddings.last_hidden_state[:,0,:]
    return last_hidden_state.detach().numpy().squeeze()