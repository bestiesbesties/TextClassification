from torch.utils.data import DataLoader, Dataset
import pandas as pd
import re
from umap import UMAP

# from transformers import BertTokenizer, BertModel
# bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# bert_model = BertModel.from_pretrained("bert-base-uncased")

# from tools import embedding_handler

def load_data(file_path:str, nrows:int=None) -> pd.DataFrame|None:
    print("Loading data...")
    df = pd.read_csv(filepath_or_buffer=file_path)
    ## Random samples pakken van 100% van de data
    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    ## Return wel of geen specifiek aantal rows.
    if nrows is not None:
        return df_shuffled[:nrows]
    else:
        return df_shuffled
    

    print(f"Amount of rows loaded: {len(df)}")
    return 

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    ## categoriale label omzetten naar kans
    mapping = {"Good Fit" : float(1),
                "Potential Fit": float(0.5),
                "No Fit" : float(0)}
    df["label"] = df["label"].map(mapping)

    ## tekst cleaning
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

class DocumentDataset(Dataset):  
    ## Maximaal aantal tokens waarvan contextual embeddings gemaakt kunnen worden in bert
    ## TODO uitzoeken of dit hoger kan
    # MAX_LENGTH = bert_model.config["max_position_embeddings"]
        
    ## Laad de custom dataset met de feature(s), labels 
    def __init__(self, documents, tokenizer_model, embedding_model):
        self.documents = documents
        self.tokenizer_model = tokenizer_model
        self.embedding_model = embedding_model

    def __len__(self):
        return len(self.documents)

    ## Inplement de functie uit de inhereted Dataset class, welke gebruikt wordt door de dataloader
    ## TODO uitzoeken waarom er niet getruncat wordt / geknipt
    ## voor nu zelf want: RuntimeError: stack expects each tensor to be equal size, but got [512] at entry 0 and [1036] at entry 2
    def __getitem__(self, index):
        print("Getting item...")
        document = self.documents.iloc[index]

        ## Tokenizer step
        resume_tokens = tokenize(
            tokenizer_model=self.tokenizer_model,
            document=document["resume_text"]
            )
        desc_tokens = tokenize(
            tokenizer_model=self.tokenizer_model,
            document=document["job_description_text"]
            )

        ## Embedding step
        resume_embedding = embed(
            embedding_model=self.embedding_model,
            input_ids=resume_tokens["input_ids"][:512],
            attention_mask=resume_tokens["attention_mask"][:512]
            )

        desc_embedding = embed(
            embedding_model=self.embedding_model,
            input_ids=desc_tokens["input_ids"][:512],
            attention_mask=desc_tokens["attention_mask"][:512]
            )
        
        return resume_embedding, desc_embedding, document["label"]

def tokenize(tokenizer_model, document):
    tokens = tokenizer_model(
        document, 
        return_tensors = "pt",
        padding = "max_length",
        trucation = True,
        max_length = 512
        )
    ## de waardes zitten nu in een 2 dimensionale tensort met slechts 1 row
    ## .squeeze(0) Haalt dimensie '0' weg van de tensor, blijft 1 dimensie over
    # return(tokens["input_ids"].squeeze(0),tokens["attention_mask"].squeeze(0))
    return tokens

def embed(embedding_model, input_ids, attention_mask):
    print("input_ids.size(): ",input_ids.size())
    embeddings = embedding_model(input_ids=input_ids, attention_mask=attention_mask)
    print(type(embeddings.last_hidden_state[:,0,:]))
    return embeddings.last_hidden_state[:,0,:]

def reduce_dimensions(embeddings) -> UMAP:
    amt_reduced_dims = int(embeddings.shape(1)/3)
    reducer = UMAP(n_neighbors=15,n_components=amt_reduced_dims)
    reduceds = reducer.fit_transform(embeddings)
    return reduceds, reducer
