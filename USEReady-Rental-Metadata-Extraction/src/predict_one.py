import argparse
import json
from .extractor import SemanticExtractor
from .canonicalizer import Canonicalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    extractor = SemanticExtractor()
    canonicalizer = Canonicalizer()
    text, raw, candidates = extractor.extract_file(args.file)
    result = canonicalizer.normalize_all(raw)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nRaw semantic answers:")
    print(json.dumps(raw, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
