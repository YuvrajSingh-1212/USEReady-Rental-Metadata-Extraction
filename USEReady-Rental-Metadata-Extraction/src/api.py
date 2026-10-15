from pathlib import Path
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from .extractor import SemanticExtractor
from .canonicalizer import Canonicalizer

app = FastAPI(title="Rental Agreement Metadata Extraction API", version="1.0.0")
extractor = None
canonicalizer = None

@app.get("/")
def root():
    return {"service": "Rental Agreement Metadata Extraction", "status": "ok", "docs": "/docs"}

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    global extractor, canonicalizer
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Only DOCX and image files are supported")
    if extractor is None:
        extractor = SemanticExtractor()
        canonicalizer = Canonicalizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text, raw, _ = extractor.extract_file(tmp_path)
        result = canonicalizer.normalize_all(raw)
        return {"file_name": file.filename, "metadata": result}
    finally:
        Path(tmp_path).unlink(missing_ok=True)
