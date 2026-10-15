from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import re
import torch
from transformers import pipeline
from .config import QA_MODEL, MAX_CONTEXT_CHARS, OVERLAP_CHARS
from .document_loader import load_document

# Regex is intentionally NOT used to locate metadata in the document.
# The semantic QA model decides which span answers each question.

QUESTIONS: Dict[str, List[str]] = {
    "Agreement Value": [
        "What is the monthly rent amount in this agreement?",
        "What amount is agreed as the rent payable for the property?",
    ],
    "Agreement Start Date": [
        "What date does the lease or rental agreement start or commence?",
        "What is the commencement date of the rental term?",
    ],
    "Agreement End Date": [
        "What date does the lease or rental agreement end or expire?",
        "What is the end date of the rental term?",
    ],
    "Renewal Notice (Days)": [
        "How much notice is required to terminate or vacate the rental agreement?",
        "What is the notice period required by either party?",
    ],
    "Party One": [
        "Who is the first party, owner, lessor, landlord, or house owner?",
        "Who is the owner or lessor in this agreement?",
    ],
    "Party Two": [
        "Who is the second party, tenant, lessee, resident, or occupier?",
        "Who is the tenant or lessee in this agreement?",
    ],
}


def chunks(text: str) -> List[str]:
    if len(text) <= MAX_CONTEXT_CHARS:
        return [text]
    out = []
    start = 0
    while start < len(text):
        end = min(len(text), start + MAX_CONTEXT_CHARS)
        out.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - OVERLAP_CHARS)
    return out


@dataclass
class Candidate:
    answer: str
    score: float
    question: str
    context: str


class SemanticExtractor:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.qa = pipeline(
            "question-answering",
            model=QA_MODEL,
            tokenizer=QA_MODEL,
            device=device,
        )

    def _ask(self, question: str, context: str) -> Candidate:
        result = self.qa(question=question, context=context, handle_impossible_answer=True)
        return Candidate(
            answer=result.get("answer", "").strip(),
            score=float(result.get("score", 0.0)),
            question=question,
            context=context,
        )

    def extract_candidates(self, text: str) -> Dict[str, List[Candidate]]:
        result: Dict[str, List[Candidate]] = {}
        ctxs = chunks(text)
        for field, questions in QUESTIONS.items():
            candidates = []
            for q in questions:
                for ctx in ctxs:
                    c = self._ask(q, ctx)
                    if c.answer:
                        candidates.append(c)
            candidates.sort(key=lambda x: x.score, reverse=True)
            result[field] = candidates[:8]
        return result

    def extract(self, text: str) -> Tuple[Dict[str, str], Dict[str, List[Candidate]]]:
        candidates = self.extract_candidates(text)
        raw = {}
        for field, vals in candidates.items():
            raw[field] = vals[0].answer if vals else ""
        return raw, candidates

    def extract_file(self, path: str | Path):
        text = load_document(path)
        raw, candidates = self.extract(text)
        return text, raw, candidates
