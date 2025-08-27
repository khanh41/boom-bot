import torch, os

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
