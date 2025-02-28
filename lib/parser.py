import PyPDF2

def pdf(filepath:str) -> str:
    """Extracts the text from a PDF file.

    Args:
        filepath (str): Path to the PDF file.

    Returns:
        str: Extracted text as flat string.
    """
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.replace("\n", " ")