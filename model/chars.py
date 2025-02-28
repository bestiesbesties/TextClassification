import re

def clean(text:str) -> str:
    """Cleans text by removing any special and unneeded characters.

    Args:
        text (str): Input text to clean.

    Returns:
        str: Lowercase cleaned text.
    """
    text_cleared = re.sub(r"[^a-z\s]", "", text.lower())
    text_cleaned = re.sub(r"\s+"," ", text_cleared).strip()
    return text_cleaned