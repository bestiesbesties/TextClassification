import numpy as np
from transformers import BertTokenizer, BertModel

from cvmatching import text, helpers, words
tokenizer_model = BertTokenizer.from_pretrained("bert-base-uncased")
embedding_model = BertModel.from_pretrained("bert-base-uncased")

def calculate_pdf(filepath:str) -> tuple[np.array, set]:
    sector_text = text.parse_pdf(filepath)
    sector_text_clean = text.clean_text(sector_text)

    ## Maak van de gecleande text een spacy doc class en sla het filter resultaat op als str
    sector_text_clean_filter = words.remove_stopwords(words.make_doc(sector_text_clean))

    sector_embeddings = helpers.state_document(tokenizer_model, embedding_model, sector_text_clean_filter)

    ## Maak van de gecleande & gefilterde text en nieuwe spacy doc class
    sector_keywords = words.extract_keywords(words.make_doc(sector_text_clean_filter))

    return(sector_embeddings, sector_keywords)


    