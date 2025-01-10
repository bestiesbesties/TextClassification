from transformers import BertTokenizer, BertModel
import pandas as pd
import json
from importlib import reload
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch import optim
from umap import UMAP

from tools import data_handler
from tools import bertswag
from tools import train_handler
from tools import test2
from tools import embedding_handler

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
reducer = UMAP(n_neighbors=15,n_components=30)

## Inladen config en instellen randomstate
with open("config.json", "r") as file:
    config = json.load(fp=file)

torch.manual_seed(config['random_state'])

df = data_handler.load_data(file_path=config['data_filepath'], nrows=1000)
df = data_handler.preprocess_data(df=df) ## TODO weghalen alle leestekens

x = df.apply(lambda x: embedding_handler.handle_batch(x, bert_tokenizer, bert_model , ), axis=1)

## train_test_split (shuffle=false) houdt stratified split aan drs eerlijke verdeling
## (random_state) overbodig tenzij wel willekeurige verdeling
## TODO stratify defineren
target = "label"
train, test = train_test_split(df, test_size=config["test_size"], shuffle=False)
train[target].value_counts()
test[target].value_counts()

train_dataset = data_handler.DocumentDataset(documents=train, tokenizer_model=bert_tokenizer, embedding_model=bert_model)
test_dataset = data_handler.DocumentDataset(documents=test, tokenizer_model=bert_tokenizer, embedding_model=bert_model)
train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True)

y = train_dataset.__getitem__(5)

compute_device = torch.device(type="cpu")
model = bertswag.BERTSwag( 
    device=compute_device,
    dropout=0.1
    ).to(compute_device)

new_folder_path = train_handler.trainer(
    model=model,
    train_loader=train_loader,
    loss_func=nn.MSELoss(), ## Regressie
    optimizer=optim.Adam(model.parameters(), lr=config["learning_rate"]),
    epochs=1
)

all_labels, all_predictions = train_handler.evaluation(
    model=model,
    test_loader=test_loader,
    loss_func=nn.MSELoss(), ## Regressie
    new_folder_path = new_folder_path,
    config = config
)

# print("bert_model.config:",bert_model.config)
# print("BERTSwag.parameters: ", SWAGmodel.parameters)
# ##TODO uitzoeken speciale getter/generator yield voor .parameters en .parameters() (call versus variable)
## CrossEntropyLoss neemt -log van de TRUE(label) predicted probability (hoe lager hoe accurater) 

