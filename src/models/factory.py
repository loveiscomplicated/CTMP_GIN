# models/factory.py
from src.models.ctmp_gin import CTMPGIN

MODEL_REGISTRY = {
    "ctmp_gin": CTMPGIN,
}

def build_model(model_name: str, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name](**kwargs)
