# config.py
import random
from pathlib import Path

class Config:
    SEED = 42
    MODEL_NAME      = "qwen3:1.7b"
    TEMPERATURE     = 0.5
    CONTEXT_WINDOW  = 4096

    class Path:
        APP_HOME        = Path(__file__).parent.parent
        DATA_DIR        = APP_HOME
        DATABASE_PATH   = APP_HOME / "northwind.db"
        VECTORS_DIR     = APP_HOME / "vector_store"        # <-- direktori Chroma

    def seed_everything(seed: int = SEED):
        random.seed(seed)