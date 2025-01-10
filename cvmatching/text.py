import re
import logging
import pandas as pd
import PyPDF2


def clean_text(text:str) -> pd.DataFrame:
    """This function cleans 2 specific columns in the provided dataframe. 

    Args:
        df (pd.DataFrame): Data holding both resume_text and job_description_text columns.

    Returns:
        pd.DataFrame: The input data, where each string is cleaned.
    """
    ## Categoriale labels omzetten naar kans
    # mapping = {"Good Fit" : float(1),
    #             "Potential Fit": float(0.5),
    #             "No Fit" : float(0)}
    # df["label"] = df["label"].map(mapping)

    ## Tekst cleaning
    ## unieke karakters verwijderen
    ## woord 'Summary' verwijderen
    ## linebrakes verwijderen
    ## cijfers verwijderen
    ## hoofletters omzetten in kleineletters
    
    forbidden = r'[@$%^&*()_+\-=\[\]{}|;:\<>/\\]|summary|Summary|\n|[0-9]'  ## | pipe or operator in regex
    df["resume_text"] = df["resume_text"].apply(lambda x: re.sub(string=x, pattern=forbidden, repl="").lower())
    df["job_description_text"] = df["job_description_text"].apply(lambda x: re.sub(string=x, pattern=forbidden, repl="").lower())

    ## drop nan values
    df = df.dropna()
    return df.reset_index(drop=True)




def parse_pdf(file_path:str) -> str:
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.replace("\n", " ")