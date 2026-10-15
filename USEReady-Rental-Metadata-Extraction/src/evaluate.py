import argparse
import pandas as pd

FIELDS = [
    "Aggrement Value",
    "Aggrement Start Date",
    "Aggrement End Date",
    "Renewal Notice (Days)",
    "Party One",
    "Party Two",
]

PRED_MAP = {
    "Aggrement Value": "Agreement Value",
    "Aggrement Start Date": "Agreement Start Date",
    "Aggrement End Date": "Agreement End Date",
    "Renewal Notice (Days)": "Renewal Notice (Days)",
    "Party One": "Party One",
    "Party Two": "Party Two",
}


def norm(x):
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().split()).casefold()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    truth = pd.read_csv(args.truth, dtype=str).fillna("")
    pred = pd.read_csv(args.predictions, dtype=str).fillna("")
    merged = truth.merge(pred, on="File Name", how="left", suffixes=("_truth", "_pred"))
    recalls = {}
    for field in FIELDS:
        pred_field = PRED_MAP[field]
        matches = (merged[field + "_truth"].map(norm) == merged[pred_field + "_pred"].map(norm))
        recall = float(matches.mean()) if len(matches) else 0.0
        recalls[field] = recall
        print(f"{field}: {matches.sum()}/{len(matches)} = {recall:.2%}")
    print(f"Overall average recall: {sum(recalls.values())/len(recalls):.2%}")

if __name__ == "__main__":
    main()
