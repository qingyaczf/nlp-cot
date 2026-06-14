"""
Step-Aware Verifier strategy.
Based on: https://arxiv.org/abs/2310.15123

Generate multiple reasoning paths, then use a step-level verifier
to score each reasoning step. Select the path with the highest
aggregate step score as the final answer.

Harness Engineering integration:
- Instructions: separate verifier prompt template with clear scoring rubric
- State: tracks per-step verification scores for each reasoning path
- Feedback: verifier feedback directly influences final answer selection
- Tools: the verifier acts as a meta-reasoning tool over the generator output
"""
import json
import os
import random
import re
from typing import Dict, Any, List

from .base import BaseStrategy


class StepAwareVerifierStrategy(BaseStrategy):
    """Step-Aware Verifier: score each step of reasoning, pick the best path.

    Supports two verification modes:
      - LLM verifier (default): uses the same model via prompt to score steps.
      - Local verifier: uses a DeBERTa (or other local) model passed as
        ``local_verifier``. When provided, ``_verify_step`` calls
        ``local_verifier.score_step()`` instead of the LLM, saving API costs.
    """

    def harness_subsystems(self) -> Dict[str, bool]:
        return {
            "instructions": True,   # generator + verifier prompts
            "tools": True,          # verifier acts as meta-reasoning tool
            "environment": True,
            "state": True,          # per-step scores tracked
            "feedback": True,       # verifier feedback filters paths
        }

    def __init__(
        self,
        model,
        task,
        prompt_template_path: str = "prompts/few_shot_cot.txt",
        verifier_prompt_path: str = "prompts/step_verifier.txt",
        n_paths: int = 5,
        n_prompts: int = 3,
        generator_temperature: float = 0.7,
        verifier_temperature: float = 0.1,
        local_verifier=None,
        few_shot_data_path: str = "data/AQuA/test.json",
        **kwargs
    ):
        super().__init__(name="step_verifier", model=model, task=task, **kwargs)
        self.n_paths = n_paths
        self.n_prompts = n_prompts
        self.generator_temperature = generator_temperature
        self.verifier_temperature = verifier_temperature
        self.local_verifier = local_verifier
        self.prompt_template_path = prompt_template_path
        self.verifier_prompt_path = verifier_prompt_path
        self.few_shot_data_path = few_shot_data_path
        self.prompt_template = self._load_prompt_template()
        self.verifier_template = self._load_verifier_template()

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

    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy metadata, including verifier type (JSON-safe)."""
        info = super().get_strategy_info()
        info["verifier"] = (
            "local"
            if self.local_verifier is not None
            else "LLM"
        )
        if self.local_verifier is not None:
            info["verifier_model_info"] = self.local_verifier.get_model_info()
        return info

    def _load_verifier_template(self) -> str:
        if not os.path.exists(self.verifier_prompt_path):
            return self._default_verifier_template()
        with open(self.verifier_prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _default_verifier_template(self) -> str:
        return (
            "You are a rigorous math reasoning verifier. Your job is to evaluate whether a "
            "given reasoning step is logically correct and helps solve the problem.\n\n"
            "Question: {question}\n"
            "Options: {options}\n\n"
            "Reasoning so far:\n{previous_steps}\n\n"
            "Step to verify:\n{step}\n\n"
            "Rate this step from 1 to 10, where:\n"
            "- 10: Perfectly correct, logically sound, and directly advances toward the answer\n"
            "- 7-9: Correct with minor issues or slightly inefficient\n"
            "- 4-6: Partially correct but contains errors, gaps, or irrelevant information\n"
            "- 1-3: Fundamentally incorrect, illogical, or misleading\n\n"
            "Respond with ONLY a numeric score on a single line:\n"
            "Score: X"
        )

    # ──────────────────────────────────────────────
    #  Diverse prompt generation (N prompts × M paths)
    # ──────────────────────────────────────────────

    def _load_few_shot_pool(self) -> List[Dict[str, Any]]:
        """Load AQuA data and return a list of candidate few-shot samples."""
        if not hasattr(self, "_pool_cache"):
            pool = []
            path = self.few_shot_data_path
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            pool.append(json.loads(line))
            self._pool_cache = pool
        return self._pool_cache

    def _build_diverse_prompts(self, question: str, options: str, n: int) -> List[str]:
        """
        Build ``n`` different prompts for the same question.

        Each prompt gets a different random set of few-shot examples from
        the AQuA pool, producing naturally diverse generation contexts.
        """
        pool = self._load_few_shot_pool()
        prompts = []

        for i in range(n):
            # Sample 3 random few-shot examples (no overlap within this batch)
            k = min(3, len(pool))
            shots = random.sample(pool, k)

            shot_blocks = []
            for s in shots:
                q = s.get("question", "")
                o = " ".join(s.get("options", []))
                c = s.get("correct", "")
                shot_blocks.append(
                    f"Example {i+1}:\n"
                    f"Question: {q}\n"
                    f"Options: {o}\n"
                    f"Answer: {c}\n"
                )

            few_shot_text = "\n".join(shot_blocks)
            prompts.append(
                f"You are solving math word problems. Here are some examples:\n\n"
                f"{few_shot_text}\n"
                f"Now solve:\n\n"
                f"Question: {question}\n"
                f"Options: {options}\n\n"
                f"Let's think step by step."
            )

        return prompts

    def _split_into_steps(self, text: str) -> List[str]:
        """
        Split reasoning text into individual steps.
        Strategy:
        1. Look for numbered steps (e.g., "Step 1:", "1.", "(1)")
        2. Fall back to paragraph splitting by blank lines
        3. Filter out very short fragments and the final answer line
        """
        # Remove the final answer line to avoid treating it as a step
        text = re.sub(r"(?i)answer\s*[:：]\s*[A-E]", "", text).strip()

        # Try numbered step pattern: lines starting with numbers or "Step N"
        step_pattern = re.compile(
            r"(?:^|\n)\s*(?:Step\s*\d+[:.\)]?\s*|\d+[:.\)]\s+|\(\d+\)\s+)",
            re.IGNORECASE
        )
        parts = step_pattern.split(text)
        if len(parts) > 2:
            steps = [p.strip() for p in parts[1:] if len(p.strip()) > 10]
            if steps:
                return steps

        # Fallback: split by blank lines
        steps = []
        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if len(paragraph) > 15 and not paragraph.lower().startswith("answer"):
                # Further split long paragraphs by newlines if they look like steps
                lines = [l.strip() for l in paragraph.split("\n") if l.strip()]
                if len(lines) > 1:
                    steps.extend(lines)
                else:
                    steps.append(paragraph)
        return steps if steps else [text]

    def _verify_step(
        self,
        question: str,
        options: str,
        previous_steps: str,
        step: str
    ) -> float:
        """
        Score a single reasoning step. Returns a float between 0 and 10.

        If ``self.local_verifier`` is set, delegates to its
        ``score_step()`` method (no API call).  Otherwise falls back to
        the LLM-based verifier prompt.
        """
        # ── Local verifier (DeBERTa, etc.) ──
        if self.local_verifier is not None:
            return self.local_verifier.score_step(
                question=question,
                options=options,
                previous_steps=previous_steps,
                step=step,
            )

        # ── LLM-based verifier (original behaviour) ──
        prompt = self.verifier_template.format(
            question=question,
            options=options,
            previous_steps=previous_steps if previous_steps else "(none)",
            step=step,
        )
        outputs = self.model.generate(
            prompt,
            temperature=self.verifier_temperature,
            max_tokens=64,
            n=1,
        )
        raw = outputs[0] if outputs else ""

        # Extract numeric score
        match = re.search(r"Score\s*[:：]\s*(\d+(?:\.\d+)?)", raw, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return max(0.0, min(10.0, score))
        # Fallback: look for any number
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", raw)
        if match:
            score = float(match.group(1))
            return max(0.0, min(10.0, score))
        return 5.0  # Neutral default

    def run(self, example: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        question = example.get("question", "")
        options = example.get("options", [])
        options_text = " ".join(options)

        # --- Phase 1: Generate diverse reasoning paths ---
        n_prompts = kwargs.get("n_prompts", self.n_prompts)
        n_per_prompt = kwargs.get("n_paths", self.n_paths)
        gen_temp = kwargs.get("generator_temperature", self.generator_temperature)
        max_tokens = kwargs.get("max_tokens", 1024)

        # Build N different prompts (each with different random few-shot examples)
        diverse_prompts = self._build_diverse_prompts(question, options_text, n_prompts)
        total_paths = n_prompts * n_per_prompt

        outputs = []
        prompt_idx = 0
        for pi, prompt in enumerate(diverse_prompts):
            print(f"    [Step-Verifier] Prompt {pi+1}/{n_prompts}, generating {n_per_prompt} paths...")
            for j in range(n_per_prompt):
                batch = self.model.generate(
                    prompt,
                    temperature=gen_temp,
                    max_tokens=max_tokens,
                    n=1,
                )
                outputs.extend(batch)
                pred_preview = self.task.extract_answer(batch[0]) if batch else ""
                print(f"      path {j+1}/{n_per_prompt} → {pred_preview}")

        # --- Phase 2: Verify each path step by step ---
        verifier_mode = "local (DeBERTa)" if self.local_verifier else "LLM"
        print(f"    [Step-Verifier] Verifying {len(outputs)} paths via {verifier_mode}...")
        path_scores: List[float] = []
        path_details: List[List[Dict[str, Any]]] = []
        predictions: List[str] = []

        for raw_output in outputs:
            pred = self.task.extract_answer(raw_output)
            predictions.append(pred)

            steps = self._split_into_steps(raw_output)
            step_records: List[Dict[str, Any]] = []
            previous_steps_text = ""
            total_score = 0.0

            for si, step in enumerate(steps):
                print(f"      Verifying step {si+1}/{len(steps)}...", end=" ")
                score = self._verify_step(
                    question=question,
                    options=options_text,
                    previous_steps=previous_steps_text,
                    step=step,
                )
                print(f"Score={score:.1f}")
                step_records.append({
                    "step": step,
                    "score": score,
                })
                total_score += score
                previous_steps_text += f"- {step}\n"

            avg_score = total_score / len(steps) if steps else 0.0
            path_scores.append(avg_score)
            path_details.append(step_records)

        # --- Phase 3: Weighted voting ---
        # Each path's avg step score = voting weight
        answer_labels = ["A", "B", "C", "D", "E"]
        vote_weights = {a: 0.0 for a in answer_labels}
        vote_paths: Dict[str, List[int]] = {a: [] for a in answer_labels}

        for idx, (pred, score) in enumerate(zip(predictions, path_scores)):
            pred = pred.upper().strip()
            if pred in vote_weights:
                vote_weights[pred] += score
                vote_paths[pred].append(idx + 1)

        # Sort answers by total weight descending
        ranked = sorted(vote_weights.items(), key=lambda x: -x[1])
        final_prediction = ranked[0][0] if ranked[0][1] > 0 else (predictions[0] if predictions else "")
        total_weight = sum(path_scores)

        # Best individual path (for reference)
        best_idx = max(range(len(path_scores)), key=lambda i: path_scores[i]) if path_scores else 0
        best_score = path_scores[best_idx] if path_scores else 0.0
        best_output = outputs[best_idx] if outputs else ""

        # Build summary output
        summary_lines = [
            f"=== Step-Aware Verifier ({total_paths} paths, {n_prompts} prompts) ===",
            f"Weighted Voting Result:",
        ]
        for ans, weight in ranked:
            marker = " <<<<" if ans == final_prediction else ""
            paths_str = f" (paths: {vote_paths[ans]})" if vote_paths[ans] else ""
            pct = f" {weight/total_weight*100:.0f}%" if total_weight > 0 else ""
            summary_lines.append(
                f"  {ans}: weight={weight:.3f}{pct}{paths_str}{marker}"
            )
        summary_lines.extend([
            f"Final Answer: {final_prediction}",
            "",
            f"Best individual path: Path {best_idx + 1} (avg step score: {best_score:.2f})",
            "--- Step Scores of Best Path ---",
        ])
        for i, rec in enumerate(path_details[best_idx]):
            summary_lines.append(f"Step {i+1} (score {rec['score']:.1f}): {rec['step'][:100]}...")
        summary_lines.extend(["", "--- Full Selected Path ---", best_output])
        summary_output = "\n".join(summary_lines)

        return {
            "prediction": final_prediction,
            "output": summary_output,
            "metadata": {
                "n_prompts": n_prompts,
                "n_paths": n_per_prompt,
                "verifier": "local" if self.local_verifier is not None else "LLM",
                "all_outputs": outputs,
                "all_predictions": predictions,
                "path_scores": path_scores,
                "best_path_index": best_idx,
                "best_path_avg_score": best_score,
                "path_step_details": [
                    {
                        "path_index": i,
                        "avg_score": path_scores[i],
                        "steps": path_details[i],
                    }
                    for i in range(len(outputs))
                ],
            },
        }
