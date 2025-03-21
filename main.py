import os
from importlib import reload
import json

from lib import parser, arguments
from model import Model

config = json.load(open("config.json", "r"))["config"]
preloads = json.load(open(os.path.join("eval", "preloads", "preloads.json"), "r"))["preloads"]

parser = arguments.create_parser()
namespace = parser.parse_args()
file_path = namespace.file_path
embedding_model_name = namespace.embedding_model_name
embedding_model_path = config["model_mapping"][embedding_model_name]

classification_model = Model(
    model_path=embedding_model_path,
    config=config,
    preloads=preloads
    )

parsed_pdf_txt = parser.pdf(file_path)
sector = classification_model.predict(parsed_pdf_txt)
print(sector)
