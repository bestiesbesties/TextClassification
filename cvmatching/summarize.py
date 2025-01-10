import os
import re
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sentence_transformers import SentenceTransformer, util
model_MiniLM = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def summarize_MiniLM(text:str|list, output_len:int=6) -> str:
    """
    Before summarizing, this function splits any text into sentences. It creates a 
    shorter text by selecting the most representative sentences from the origin text.
    Cosine-similarity is used to calculate how much an embedded sentence is the same as
    the mean embedding of all embedded sentences from the origin text. If the text is 'too' 
    big the function does not try to fragment it.

    Args:
        text (str): A text to summarize.
        output_len (sintr): The amount of sentences the summary will have.

    Returns:
        str: A summary of the input text.

    """
    if type(text) == list:
        text = __aggragate_for_summary(text)

    ## Split zinnen op generieke syntax met behoud van de leestekens
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    if output_len > len(sentences):
        print("len(sentences): ", len(sentences), "output_len: ", output_len)
        return ""
    
    ## Cosine-similarity van elke embedding tegenover het gemiddelde van de embeddings 
    embeddings = model_MiniLM.encode(sentences, convert_to_tensor=True)
    sentence_scores = util.cos_sim(embeddings, embeddings.mean(axis=0))

    ## Sorteren embeddings die het meest lijken op de gemiddelde embedding
    top_sentence_indices = sentence_scores.flatten().argsort(descending=True)[:output_len * 2].tolist()

    ## Selecteer zinnen voor de samenvatting en voorkom dubbele zinnen
    in_sum = 0
    seen_sentences = set()
    summary = "'"
    for id in top_sentence_indices:
        sentence = sentences[id]
        if sentence not in seen_sentences and in_sum <= output_len:
            summary += sentence + " "
            in_sum += 1
            seen_sentences.add(sentence)

    return summary.strip()


def __aggragate_for_summary(full_texts:list[str]) -> str:
    return " ".join(full_texts)