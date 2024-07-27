from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parents[2]
SAVED_MODELS_PATH = ROOT_DIR / "saved_models"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SMALL_DATA_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
SMALL_DATA_FILEPATH = RAW_DATA_DIR / "small_instruct_data.json"

BASE_CONFIG = {
    "vocab_size": 50257,  # vocab size
    "context_length": 1024,  # context length
    "dropout": 0.1,  # dropout rate
    "qkv_bias": True,  # qkv bias
    "activation": "gelu",
    "norm_first": False,
    "pre_norm": True,
}

MODEL_CONFIGS = {
    "gpt2": {"d_model": 768, "num_layers": 12, "num_heads": 12},  # 124M
    "gpt2-medium": {
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
    },  # 355M
    "gpt2-large": {"d_model": 1280, "num_layers": 36, "num_heads": 20},  # 774M
    "gpt2-xl": {"d_model": 1600, "num_layers": 48, "num_heads": 25},  # 1558M
}
