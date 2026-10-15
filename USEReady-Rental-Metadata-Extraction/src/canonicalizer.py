from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .config import CANONICALIZER_MODEL

FIELDS = [
    "Agreement Value",
    "Agreement Start Date",
    "Agreement End Date",
    "Renewal Notice (Days)",
    "Party One",
    "Party Two",
]

PROMPTS = {
    "Agreement Value": "Convert the extracted rent answer to only the numeric monthly rent amount. Return digits only. Extracted answer: {answer}",
    "Agreement Start Date": "Convert the extracted lease start date to DD.MM.YYYY. Return only the date. Extracted answer: {answer}",
    "Agreement End Date": "Convert the extracted lease end date to DD.MM.YYYY. Return only the date. Extracted answer: {answer}",
    "Renewal Notice (Days)": "Convert the extracted termination notice period to the number of days. Return digits only. If it is expressed in months, convert one month to 30 days. Extracted answer: {answer}",
    "Party One": "Return only the name of the owner/lessor/landlord/first party. Remove role labels and address details. Extracted answer: {answer}",
    "Party Two": "Return only the name of the tenant/lessee/resident/second party. Remove role labels and address details. Extracted answer: {answer}",
}

class Canonicalizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CANONICALIZER_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(CANONICALIZER_MODEL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def normalize(self, field: str, answer: str) -> str:
        if not answer:
            return ""
        prompt = PROMPTS[field].format(answer=answer)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=48, num_beams=4, do_sample=False)
        value = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return self._light_cleanup(field, value)

    def _light_cleanup(self, field: str, value: str) -> str:
        # Formatting only; field identification is performed by the transformer.
        value = value.strip().strip('"\'')
        if field in {"Agreement Start Date", "Agreement End Date"}:
            value = value.replace("/", ".").replace("-", ".")
        return value

    def normalize_all(self, raw: Dict[str, str]) -> Dict[str, str]:
        return {field: self.normalize(field, raw.get(field, "")) for field in FIELDS}
