import re
import logging
import PyPDF2

def clean_text(text:str) -> str:
    text_cleared = re.sub(r"[^a-z\s]", "", text.lower())
    text_cleaned = re.sub(r"\s+"," ", text_cleared).strip()
    return text_cleaned


def parse_pdf(filepath:str) -> str:
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.replace("\n", " ")