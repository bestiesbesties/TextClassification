import os
import logging
import numpy as np
import json
import pandas as pd 
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from importlib import reload

from cvmatching import helpers, text

with open("config.json", "r") as file:
    config = json.load(file)["config"]

bert_tokenizer = BertTokenizer.from_pretrained(config["model_name"])
bert_model = BertModel.from_pretrained(config["model_name"])

jd = text.parse_pdf(config["jd"])
cv = text.parse_pdf(config["cv"])

jd_embedding = helpers.state_document(bert_tokenizer, bert_model, jd)
cv_embedding = helpers.state_document(bert_tokenizer, bert_model, cv)

similarity = cosine_similarity([jd_embedding], [cv_embedding])
print("similarity", similarity[0][0])
