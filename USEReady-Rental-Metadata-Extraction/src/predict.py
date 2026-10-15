import argparse
from pathlib import Path
import pandas as pd
from .extractor import SemanticExtractor
from .canonicalizer import Canonicalizer, FIELDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    extractor = SemanticExtractor()
    canonicalizer = Canonicalizer()
    rows = []
    for path in sorted(Path(args.input_dir).iterdir()):
        if path.suffix.lower() not in {".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        print(f"Processing: {path.name}")
        text, raw, candidates = extractor.extract_file(path)
        normalized = canonicalizer.normalize_all(raw)
        row = {"File Name": path.stem}
        row.update(normalized)
        rows.append(row)
        print(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["File Name"] + FIELDS).to_csv(out, index=False)
    print(f"Saved predictions to: {out}")

if __name__ == "__main__":
    main()
