import numpy as np

from lib import text, words
from model import uniform

def calculate_pdf(filepath:str, model_name:str) -> tuple[np.array, set]:
    """Runs pipeline to create embeddings and combine extracted filtered keywords.

    Args:
        filepath (str): Path to the PDF file.
        model_name (str): Name of the embedding model.

    Returns:
        tuple[np.array, set]: Word embeddings and keywords.
    """
    sector_text = text.parse_pdf(filepath)
    sector_text_clean = text.clean_text(sector_text)

    ## Make Spacy Doc from cleaned text 
    sector_text_clean_filter = words.remove_stopwords(words.make_doc(sector_text_clean))

    model = uniform.Model(model_name)
    sector_embeddings = model.run(sector_text_clean_filter)

    ## Make Spacy Doc from cleaned & filtered text
    sector_keywords = words.extract_keywords(words.make_doc(sector_text_clean_filter))

    return(sector_embeddings, sector_keywords)
    