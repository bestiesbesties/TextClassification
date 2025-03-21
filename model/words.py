import spacy
from spacy.cli import download
from spacy.util import get_package_path

def check_install_model() -> bool:
    """Checks if the specific 'en_core_web_sm' model for Spacy is available.

    Returns:
        bool: True if the model is installed, False if the model had to be downloaded.
    """
    try:
        get_package_path("en_core_web_sm")
        return True
    except:
        download("en_core_web_sm")
        return False

check_install_model()
nlp = spacy.load("en_core_web_sm")

def make_doc(text:str) -> spacy.tokens.doc.Doc:
    """Transforms a string into a Spacy doc object.

    Args:
        text (str): Respectively complete body of text. 

    Returns:
        spacy.tokens.doc.Doc: A feature rich text object for NLP.
    """
    return nlp(text)
    
def extract_keywords(doc:spacy.tokens.doc.Doc) -> set:
    """Creates a (filtered) set containing different type of subjects.
    The filter condition is applied by the '__burn_keywords' function.

    Args:
        doc (spacy.tokens.doc.Doc): A feature rich text object for NLP.

    Returns:
        set: Contains subjects as individual strings.
    """
    keywords = {chunk.text.lower() for chunk in doc.noun_chunks}
    return __burn_keywords(keywords)

def remove_stopwords(doc:spacy.tokens.doc.Doc) -> str:
    """Cleans the text of a Doc object from a specific set of English words which lack context.

    Args:
        doc (spacy.tokens.doc.Doc): A feature rich text object for NLP.

    Returns:
        str: Cleaned input from provided Doc.
    """
    text = " ".join([token.text for token in doc if not token.is_stop])
    return text.strip()

def extract_lemmas(doc: spacy.tokens.doc.Doc):
    """Extracts the source of the words in a Doc object as a string.
    Args:
        doc (spacy.tokens.doc.Doc): A spaCy Doc object containing tokenized text.
    Returns:
        str: A string of space-separated lemmatized words from the input document.
    """
    text = " ".join([token.lemma_ for token in doc])
    return text.strip()

def __burn_keywords(keywords:set) -> set:
    """Filters out a hardcoded list of English stopwords from a provided set.

    Args:
        keywords (set): Set to filter out on.

    Returns:
        set: Filtered input set
    """
    prohibited_words = ["eg", "etc", "specific" "relevant", "work", "time"]
    all = []
    for sentence in keywords:
        for word in sentence.split(" "):
            if word not in prohibited_words:
                all.append(word)
    return(set(all))