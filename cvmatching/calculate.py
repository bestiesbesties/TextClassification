import numpy as np

from cvmatching import text, helpers, words, uniform

def calculate_pdf(filepath:str, model_name:str) -> tuple[np.array, set]:
    sector_text = text.parse_pdf(filepath)
    sector_text_clean = text.clean_text(sector_text)

    ## Maak van de gecleande text een spacy doc class en sla het filter resultaat op als str
    sector_text_clean_filter = words.remove_stopwords(words.make_doc(sector_text_clean))

    sector_embeddings = uniform.uniform_run(text=sector_text_clean_filter, model_name=model_name)

    ## Maak van de gecleande & gefilterde text en nieuwe spacy doc class
    sector_keywords = words.extract_keywords(words.make_doc(sector_text_clean_filter))

    return(sector_embeddings, sector_keywords)


    