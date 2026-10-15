from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QA_MODEL = os.getenv("QA_MODEL", "deepset/roberta-base-squad2")
CANONICALIZER_MODEL = os.getenv("CANONICALIZER_MODEL", "google/flan-t5-small")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "2400"))
OVERLAP_CHARS = int(os.getenv("OVERLAP_CHARS", "300"))
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
