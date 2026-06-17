"""
DeBERTa Step Verifier: wraps the custom DebertaV2ForTokenClassification model
for holistic solution-level scoring.

Key design (matches run_ner.py):
  - Model: token classification (trained), but inference uses sequence
    classification — only the [CLS] position matters.
  - Score: softmax(logits[:, 0, :])[SOLUTION-CORRECT] × 10
    (position 0 = BOS [CLS]; inference uses sequence classification,
     not token-level predictions — STEP labels are training-only aux).
  - STEP labels (loss weight 0.1) are training-only auxiliary signals;
    inference entirely ignores per-token predictions.
  - Input format: "[CLS] {solution} && {question} ####{answer}"
    where solution steps are separated by "%%".

Usage:
    verifier = DebertaStepVerifier(model_path="data/checkpoint")
    score = verifier.score_step(question, options, previous_steps, step)
    # or holistic:
    score = verifier.score_full_path(question, full_solution_text)
"""
import os
import re
from typing import Optional

import torch

# ── Use HF mirror for users in China ──
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ── Import the custom model from models/ ──
from models.deberta_model import DebertaV2ForTokenClassification
from transformers import AutoTokenizer


class DebertaStepVerifier:
    """DeBERTa-v3-large verifier for AQuA step verification.

    Scoring strictly follows ``run_ner.py:get_solution_logits()``:
    softmax at **position 1** (the ``[CLS]`` token in the training format),
    take the ``SOLUTION-CORRECT`` probability, map to 0-10.
    """

    def __init__(
        self,
        model_path: str = "k1r1same/aqua-verifier",
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_length = max_length

        if tokenizer_name is None:
            tokenizer_name = "microsoft/deberta-v3-large"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = DebertaV2ForTokenClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    # ──────────────────────────────────────────────────────
    #  Public interfaces
    # ──────────────────────────────────────────────────────

    def score_step(
        self,
        question: str,
        options: str,
        previous_steps: str,
        step: str,
    ) -> float:
        """
        Score a reasoning step with full accumulated context.

        Although the signature matches ``_verify_step()`` (called per-step
        by ``StepAwareVerifierStrategy``), this method internally
        reconstructs the **full solution text** and scores it holistically,
        because the model does step-level work only as an auxiliary loss.

        Returns a float 0-10 (10 = perfectly correct solution).
        """
        clean_prev = self._clean_context(previous_steps)
        clean_step = step.strip()

        if clean_prev:
            full = clean_prev + "%%" + clean_step
        else:
            full = clean_step

        return self._score_text(question, options, full)

    def score_full_path(
        self, question: str, solution: str, options: str = ""
    ) -> float:
        """Score an entire reasoning path in one call.  Prefer this."""
        return self._score_text(question, options, solution)

    # ──────────────────────────────────────────────────────
    #  Internal
    # ──────────────────────────────────────────────────────

    def _score_text(self, question: str, options: str, solution: str) -> float:
        """Tokenise, forward, extract [CLS] score (position 0)."""
        solution = self._preprocess(solution)
        q_part = f"Q: {question} {options}".strip()
        text = f"[CLS] {solution} && {q_part}"

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return self._logits_to_score(outputs.logits.cpu())

    def _logits_to_score(self, logits: torch.Tensor) -> float:
        """
        Convert model logits to a 0-10 score.

        Uses **position 0** (BOS ``[CLS]`` token), which is where the
        SOLUTION label lands after the tokenizer adds the automatic BOS.

        ⚠️  ``run_ner.py:get_solution_logits()`` uses position 1, but that
        is because the training data pipeline strips the BOS ``[CLS]``
        before alignment.  With our inference format (``[CLS] ... && ...``
        as raw text), position 0 is the correct one.
        """
        # logits shape: (batch, seq_len, num_labels) = (1, seq_len, 5)
        probs = torch.softmax(logits[0, 0], dim=-1)  # position 0 = BOS [CLS]
        return max(0.0, min(10.0, probs[0].item() * 10.0))

    # ──────────────────────────────────────────────────────
    #  Text processing
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(text: str) -> str:
        """
        Normalise solution text to match the training-data format.

        1. Remove ``- `` / ``* `` list prefixes (from step_verifier.py).
        2. Replace newlines with ``%%`` (step separator).
        3. Append ``####{answer}`` if a final answer choice is detected.
        """
        text = text.strip()
        # Strip list prefixes like "- Step 1: ..." or "* text"
        text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)
        # Normalise line endings
        text = text.replace("\r\n", "\n")
        # Collapse blank lines
        text = re.sub(r"\n{2,}", "\n", text)
        # Replace \n with %% (training-data step separator)
        text = text.replace("\n", "%%")

        # Append ####{answer} if not already present
        if "####" not in text:
            m = re.search(r"answer\s*[:\s]+([A-E])", text, re.IGNORECASE)
            if m:
                text += f" ####{m.group(1)}"
        return text

    @staticmethod
    def _clean_context(previous_steps: str) -> str:
        """Strip the ``- {step}\\n`` accumulation format used by the strategy."""
        text = previous_steps.strip()
        text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)
        return text

    # ──────────────────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        return {
            "verifier": "DebertaStepVerifier",
            "model_type": "DebertaV2ForTokenClassification (custom)",
            "scoring": "softmax(logits[:,1,:])[SOLUTION-CORRECT] × 10",
            "device": self.device,
        }

    def __repr__(self):
        return f"DebertaStepVerifier(device={self.device}, scoring=[CLS]@pos0)"
