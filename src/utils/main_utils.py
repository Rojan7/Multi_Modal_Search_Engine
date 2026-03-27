import os
from src.exception import MyException
from src.logger import logger
import json
import torch
from transformers import CLIPModel, CLIPProcessor


def save_json(text):
    with open("URL_mapping","w") as f:
        json.dump(text,f)

def load_json(URL_Mapping):
    with open(URL_Mapping, "r") as f:
        img_to_url=json.load(f)
        



def load_clip_model(model_path: str, device: str = None):
    """
    Loads CLIP model and processor from given path

    Args:
        model_path (str): Path to fine-tuned or pretrained CLIP model
        device (str): 'cpu', 'cuda', or 'mps' (optional)

    Returns:
        model, processor, device
    """

    # Auto-detect device if not provided
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device = torch.device(device)

    # Load model + processor
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)

    # Move model to device
    model = model.to(device)

    # Set evaluation mode (VERY IMPORTANT)
    model.eval()

    return model, processor, device