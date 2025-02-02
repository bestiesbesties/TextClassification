import spacy
from spacy.cli import download
from spacy.util import get_package_path

def check_install_model() -> bool:
    try:
        get_package_path("en_core_web_sm")
        return True
    except:
        download("en_core_web_sm")
        return False

check_install_model()
nlp = spacy.load("en_core_web_sm")

def make_doc(text:str) -> spacy.tokens.doc.Doc:
    return nlp(text)
    
def extract_keywords(doc:spacy.tokens.doc.Doc) -> set:
    keywords = {chunk.text.lower() for chunk in doc.noun_chunks}
    return __burn_keywords(keywords)

def remove_stopwords(doc:spacy.tokens.doc.Doc) -> str:
    filtered_text = " ".join([token.text for token in doc if not token.is_stop])
    return filtered_text.strip()

def __burn_keywords(keywords:set) -> set:
    prohibited_words = ["eg", "etc", "specific" "relevant", "work", "time"]
    all = []
    for sentence in keywords:
        for word in sentence.split(" "):
            if word not in prohibited_words:
                all.append(word)
    return(set(all))