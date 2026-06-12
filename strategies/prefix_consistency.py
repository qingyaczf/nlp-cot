"""
Prefix Consistency strategy.
Based on: https://arxiv.org/abs/2605.07654

Sample multiple Chain-of-Thought traces, truncate each partway through,
regenerate the remainder, and weight votes by how often the original answer
reappears under regeneration (prefix consistency).

This reaches standard Self-Consistency plateau accuracy with up to 21x
fewer tokens (median 4.6x).
"""
import os
from collections import Counter
from typing import Dict, Any, List

from .base import BaseStrategy


class PrefixConsistencyStrategy(BaseStrategy):
    """
    Prefix-Consistency Weighted Majority Voting (PC-WMV).

    1. Generate N CoT traces for the same question.
    2. Truncate each trace at a fixed ratio (default 50%).
    3. Regenerate the remainder from the prefix multiple times.
    4. Compute prefix consistency = fraction of regenerations that reproduce
       the original answer.
    5. Aggregate answers via weighted majority voting using prefix consistency
       as the per-sample weight.
    """

    def harness_subsystems(self) -> Dict[str, bool]:
        return {
            "instructions": True,
            "tools": False,
            "environment": True,
            "state": True,      # tracks multiple reasoning paths
            "feedback": True,   # prefix consistency is a reliability feedback signal
        }

    def __init__(
        self,
        model,
        task,
        prompt_template_path: str = "prompts/base_cot.txt",
        n_paths: int = 5,
        truncation_ratio: float = 0.5,
        regen_count: int = 3,
        weight_fn: str = "linear",
        temperature: float = 0.7,
        **kwargs
    ):
        super().__init__(name="prefix_consistency", model=model, task=task, **kwargs)
        self.n_paths = n_paths
        self.truncation_ratio = truncation_ratio
        self.regen_count = regen_count
        self.weight_fn = weight_fn
        self.temperature = temperature
        self.prompt_template_path = prompt_template_path
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        if not os.path.exists(self.prompt_template_path):
            return (
                "You are solving a math word problem. Think step by step and explain your reasoning clearly.\n\n"
                "Question: {question}\n"
                "Options: {options}\n\n"
                "At the end of your response, you must state your final answer choice on a single line in exactly this format:\n"
                "Answer: X\n"
                "where X is one of A, B, C, D, or E."
            )
        with open(self.prompt_template_path, "r", encoding="utf-8") as f:
            return f.read()

    def _truncate_text(self, text: str, ratio: float) -> str:
        """Truncate text at the given character ratio."""
        if not text:
            return text
        truncate_at = max(1, int(len(text) * ratio))
        return text[:truncate_at]

    def _compute_weight(self, consistency: float) -> float:
        """Map consistency rate to vote weight."""
        if self.weight_fn == "linear":
            return consistency
        elif self.weight_fn == "quadratic":
            return consistency ** 2
        elif self.weight_fn == "cubic":
            return consistency ** 3
        elif self.weight_fn == "unanimous":
            return 1.0 if consistency >= 1.0 else 0.0
        else:
            return consistency

    def run(self, example: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        question = example.get("question", "")
        options = example.get("options", [])
        options_text = " ".join(options)

        prompt = self.prompt_template.format(question=question, options=options_text)

        n = kwargs.get("n_paths", self.n_paths)
        temp = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", 1024)
        trunc_ratio = kwargs.get("truncation_ratio", self.truncation_ratio)
        regen_count = kwargs.get("regen_count", self.regen_count)

        # Phase 1: Generate initial CoT traces
        outputs = []
        for i in range(n):
            print(f"    [Prefix-Consistency] Generating path {i+1}/{n}...", end=" ")
            batch = self.model.generate(
                prompt,
                temperature=temp,
                max_tokens=max_tokens,
                n=1,
            )
            outputs.extend(batch)
            pred_preview = self.task.extract_answer(batch[0]) if batch else ""
            print(f"→ {pred_preview}")

        # Extract initial predictions
        init_predictions: List[str] = []
        for raw in outputs:
            pred = self.task.extract_answer(raw)
            init_predictions.append(pred)

        # Phase 2: Truncate and regenerate for each trace
        # Structure: regen_results[i] = list of regen outputs for trace i
        regen_results: List[List[str]] = []
        consistencies: List[float] = []

        for i, (raw_output, init_pred) in enumerate(zip(outputs, init_predictions)):
            prefix = self._truncate_text(raw_output, trunc_ratio)
            print(f"    [Prefix-Consistency] Trace {i+1}/{n} trunc@{trunc_ratio} → regen×{regen_count}...", end=" ")

            regen_outputs = []
            match_count = 0
            for j in range(regen_count):
                regen_batch = self.model.generate(
                    prefix,
                    temperature=temp,
                    max_tokens=max_tokens,
                    n=1,
                )
                regen_text = regen_batch[0] if regen_batch else ""
                regen_pred = self.task.extract_answer(regen_text)
                regen_outputs.append(regen_text)

                if regen_pred and regen_pred.upper() == init_pred.upper():
                    match_count += 1

            consistency = match_count / regen_count if regen_count > 0 else 0.0
            consistencies.append(consistency)
            regen_results.append(regen_outputs)
            print(f"consistency={consistency:.2f}")

        # Phase 3: Weighted majority voting by prefix consistency
        weighted_votes: Counter = Counter()
        for pred, consistency in zip(init_predictions, consistencies):
            if pred:
                weight = self._compute_weight(consistency)
                weighted_votes[pred.upper()] += weight

        if not weighted_votes:
            final_prediction = ""
        else:
            final_prediction = weighted_votes.most_common(1)[0][0]

        # Build summary output for logging
        summary_lines = [
            f"=== Prefix-Consistency ({n} paths, trunc={trunc_ratio}, regen={regen_count}) ===",
            f"Weight function: {self.weight_fn}",
            f"Weighted votes: {dict(weighted_votes)}",
            f"Final Answer: {final_prediction}",
            "",
            "--- Initial paths ---",
        ]
        for idx, (pred, cons) in enumerate(zip(init_predictions, consistencies)):
            summary_lines.append(f"Path {idx+1}: pred={pred} consistency={cons:.2f}")

        summary_lines.extend(["", "--- Regeneration examples (Path 1) ---"])
        if regen_results:
            for j, regen in enumerate(regen_results[0][:2]):
                summary_lines.append(f"Regen {j+1}: {regen[:200]}...")

        summary_output = "\n".join(summary_lines)

        return {
            "prediction": final_prediction,
            "output": summary_output,
            "metadata": {
                "prompt": prompt,
                "n_paths": n,
                "truncation_ratio": trunc_ratio,
                "regen_count": regen_count,
                "weight_fn": self.weight_fn,
                "all_outputs": outputs,
                "all_predictions": init_predictions,
                "regen_results": regen_results,
                "consistencies": consistencies,
                "weighted_votes": dict(weighted_votes),
            },
        }
