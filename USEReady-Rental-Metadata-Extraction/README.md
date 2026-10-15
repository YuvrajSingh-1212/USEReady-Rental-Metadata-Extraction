# USEReady AI/ML Internship — Rental Agreement Metadata Extraction

## 1. Problem
Build an AI/ML system that extracts metadata from rental-agreement documents irrespective of template. Input documents may be scanned PNG images or DOCX files.

Required fields:
- Agreement Value
- Agreement Start Date
- Agreement End Date
- Renewal Notice (Days)
- Party One
- Party Two

The solution deliberately avoids Regex/static field-location rules for selecting values. It uses OCR for scanned images and pretrained transformer models for semantic question answering and canonicalization.

## 2. Solution architecture

```text
PNG/DOCX
   |
   +--> DOCX text extraction (python-docx)
   |
   +--> PNG OCR (Tesseract)
            |
            v
       Clean document text
            |
            v
  Transformer Question Answering
     (deepset/roberta-base-squad2)
            |
            v
       Candidate answers
            |
            v
 Transformer canonicalizer
     (google/flan-t5-small)
            |
            v
 Structured JSON/CSV
            |
            v
 Per-field exact recall
```

### Why this approach?
The assignment explicitly says not to use a rule-based approach such as Regex/static conditions. The extraction stage therefore asks a pretrained language model semantic questions such as *“What is the monthly rent amount?”* instead of looking for a fixed phrase or document position. A second pretrained model converts the semantic answer into the exact canonical format required by the evaluation CSV.

## 3. Requirements

- Python 3.10 or 3.11 recommended
- Tesseract OCR installed and available on PATH
- Internet connection on first run so Hugging Face models can be downloaded
- 4 GB+ RAM recommended; 8 GB+ is preferable

## 4. Installation

### Windows

1. Install Python from https://www.python.org/downloads/
2. Install Tesseract OCR. The default Windows installation is usually:
   `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`
3. Open a new terminal in this project folder.
4. Create a virtual environment:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

5. Install Python dependencies:

```bash
pip install -r requirements.txt
```

If Tesseract is not on PATH, set its location before running:

```bash
set TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
```

### macOS

```bash
brew install tesseract
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y tesseract-ocr
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Run the complete test prediction pipeline

From the project root:

```bash
python -m src.predict --input-dir data/test --output outputs/test_predictions.csv
```

This will:
1. read each DOCX/PNG in `data/test/`;
2. extract text directly from DOCX or OCR the PNG;
3. semantically extract the six required fields;
4. canonicalize the answers;
5. save predictions to `outputs/test_predictions.csv`.

## 6. Evaluate predictions

```bash
python -m src.evaluate --truth data/test.csv --predictions outputs/test_predictions.csv
```

The evaluator prints recall for every field and average recall.

## 7. Run on one document

```bash
python -m src.predict_one --file data/test/24158401-Rental-Agreement.png
```

## 8. REST API (optional requirement)

Start the API:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open Swagger UI:

`http://127.0.0.1:8000/docs`

Or call the endpoint with curl:

```bash
curl -X POST "http://127.0.0.1:8000/extract" -F "file=@data/test/24158401-Rental-Agreement.png"
```

The API returns JSON with the six extracted fields.

## 9. Output format

Example:

```json
{
  "File Name": "24158401-Rental-Agreement",
  "Agreement Value": "12000",
  "Agreement Start Date": "01.04.2008",
  "Agreement End Date": "31.03.2009",
  "Renewal Notice (Days)": "60",
  "Party One": "Hanumaiah",
  "Party Two": "Vishal Bhardwaj"
}
```

## 10. Important note on dates and notice periods

Some source documents express dates or durations indirectly (for example, an agreement can state an 11-month term without printing an explicit end date). The semantic model identifies the relevant term, and the canonicalizer converts the answer to the requested output format. Date arithmetic is used only when the extracted agreement term requires it; it is not used to identify fields by fixed document locations.

## 11. Project structure

```text
USEReady_Rental_Metadata_Extraction/
├── assignment-details.pdf
├── data/
│   ├── train/
│   ├── test/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── document_loader.py
│   ├── extractor.py
│   ├── canonicalizer.py
│   ├── predict.py
│   ├── predict_one.py
│   ├── evaluate.py
│   └── api.py
├── outputs/
├── notebooks/
├── requirements.txt
├── .gitignore
└── README.md
```

## 12. Reproducibility

The first execution downloads the Hugging Face models:
- `deepset/roberta-base-squad2`
- `google/flan-t5-small`

After downloading, Transformers caches them locally. The same code can then be rerun without downloading the models again.
