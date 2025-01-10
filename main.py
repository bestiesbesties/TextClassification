import os
import logging
import numpy as np
import json
import pandas as pd 
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from importlib import reload

from cvmatching import helpers

with open("config.json", "r") as file:
    config = json.load(file)["config"]

bert_tokenizer = BertTokenizer.from_pretrained(config["model_name"])
bert_model = BertModel.from_pretrained(config["model_name"])

filepath = os.path.join("groep8", "data", "data.csv")
df = pd.read_csv(filepath_or_buffer=filepath)

label = df.iloc[0]['label']

index = 400

jd = df.iloc[index]['job_description_text']
cv = df.iloc[index]['resume_text']
label = df.iloc[index]['label']

jd_embedding = helpers.state_document(bert_tokenizer, bert_model, jd)
cv_embedding = helpers.state_document(bert_tokenizer, bert_model, cv)

similarity = cosine_similarity([jd_embedding], [cv_embedding])
print(f"similarity: {similarity[0][0]}")
print(f"label: {label}")
