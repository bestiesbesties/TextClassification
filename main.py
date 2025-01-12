import logging
import argparse
from importlib import reload
import json
from transformers import BertTokenizer, BertModel

from cvmatching import helpers, text

parser = argparse.ArgumentParser()
parser.add_argument(
    "filepath", 
    type=str, 
    help="The rich words document to process and save embeddings for."
)
namespace = parser.parse_args()

with open("config.json", "r") as file:
    config = json.load(file)["config"]

bert_tokenizer = BertTokenizer.from_pretrained(config["model_name"])
bert_model = BertModel.from_pretrained(config["model_name"])

pdf_text = text.parse_pdf(namespace.filepath)

pdf_embedding = helpers.state_document(bert_tokenizer, bert_model, pdf_text)

print(pdf_embedding)
