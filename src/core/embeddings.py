import torch
from utils import load_yaml_file
from langchain_huggingface import HuggingFaceEmbeddings


def embedding_function():
    config_data = load_yaml_file("config.yaml")

    # Force CPU usage to avoid CUDA out of memory errors
    device = torch.device("cpu")

    model_name = config_data["embedding_model_name"]
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    return embeddings
