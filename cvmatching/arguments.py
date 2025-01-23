import argparse

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "file_path", 
    type=str, 
    help="The rich words document filepath to process and stdout the calculations for."
    )
    parser.add_argument(
        "embedding_model_name", 
        type=str, 
        help="The name of the embedding model of which the embeddings have to be generated with."
    )
    parser.add_argument(
        "use_faiss",
        type=str,
        choices=["True", "False"],
        help="Include ranking calculated with fais in scores"
    )

    return parser