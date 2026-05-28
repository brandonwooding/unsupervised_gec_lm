from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch
from pathlib import Path

def login_to_huggingface(env_file: Path) -> str | None:
    if env_file.exists():
        load_dotenv(env_file)
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("huggingface_login=skipped token_env_missing")
        return None
    try:
        login(token=token, add_to_git_credential=False)
    except Exception as exc:
        print(f"huggingface_login=failed {type(exc).__name__}: {exc}")
        return token
    print("huggingface_login=ok")
    return token


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device
