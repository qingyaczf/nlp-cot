"""
Few-Shot Chain-of-Thought (CoT) strategy.

Randomly selects 5 examples from the AQuA dataset and uses the LLM to
generate a reasoning chain for each, then includes them as few-shot
demonstrations in the prompt.

Reference: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning
in Large Language Models" (NeurIPS 2022).
"""
import json
import os
import random
from typing import Any, Dict, List, Optional

from .base_cot import BaseCOTStrategy


class FewShotCOTStrategy(BaseCOTStrategy):
    """CoT with few-shot demonstrations sampled from the dataset."""

    def __init__(
        self,
        model,
        task,
        prompt_template_path: str = "prompts/few_shot_cot.txt",
        n_shots: int = 5,
        few_shot_data_path: str = "data/AQuA/test.json",
        few_shot_cache_path: Optional[str] = None,
        generator_temperature: float = 0.5,
        **kwargs,
    ):
        # Use our custom prompt template
        super().__init__(
            model=model,
            task=task,
            prompt_template_path=prompt_template_path,
            **kwargs,
        )
        self.n_shots = n_shots
        self.few_shot_data_path = few_shot_data_path
        self.few_shot_cache_path = few_shot_cache_path
        self.generator_temperature = generator_temperature

        # Few-shot examples are re-sampled randomly on every run() call

    def _load_few_shot_pool(self) -> List[Dict[str, Any]]:
        """Load AQuA data and filter out samples usable as few-shots."""
        if not os.path.exists(self.few_shot_data_path):
            raise FileNotFoundError(
                f"Cannot load few-shot data: {self.few_shot_data_path}"
            )
        pool = []
        with open(self.few_shot_data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pool.append(json.loads(line))
        return pool

    def _build_few_shot_prompt_block(self) -> str:
        """
        Build the ``{few_shot_examples}`` block by:

        1. Selecting ``n_shots`` random samples from the AQuA pool.
        2. Generating a CoT reasoning chain for each via the LLM.
        3. Formatting as ``Question / Options / Reasoning / Answer``.
        """
        pool = self._load_few_shot_pool()

        if len(pool) < self.n_shots:
            raise ValueError(
                f"Few-shot pool has {len(pool)} samples, need {self.n_shots}"
            )

        selected = random.sample(pool, self.n_shots)

        blocks = []
        for i, sample in enumerate(selected, 1):
            question = sample.get("question", "")
            options_text = " ".join(sample.get("options", []))
            correct = sample.get("correct", "")

            # Generate a reasoning chain using the LLM
            chain = self._generate_reasoning_chain(question, options_text)

            blocks.append(
                f"Example {i}:\n"
                f"Question: {question}\n"
                f"Options: {options_text}\n"
                f"Reasoning:\n{chain}\n"
                f"Answer: {correct}\n"
            )

        return "\n".join(blocks)

    def _generate_reasoning_chain(self, question: str, options: str) -> str:
        """Ask the LLM to produce a step-by-step reasoning for a sample."""
        prompt = (
            "Solve this math problem step by step. "
            "End with 'Answer: X' where X is A, B, C, D, or E.\n\n"
            f"Question: {question}\n"
            f"Options: {options}\n\n"
            "Let's think step by step."
        )
        outputs = self.model.generate(
            prompt,
            temperature=self.generator_temperature,
            max_tokens=512,
            n=1,
        )
        return outputs[0].strip() if outputs else "(no reasoning generated)"

    def get_strategy_info(self) -> Dict[str, Any]:
        info = super().get_strategy_info()
        info["n_shots"] = self.n_shots
        info["few_shot_source"] = self.few_shot_data_path
        return info

    def run(self, example: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        question = example.get("question", "")
        options = example.get("options", [])
        options_text = " ".join(options)

        # Randomly sample fresh few-shot examples for each question
        print(f"    [FewShot] Sampling {self.n_shots} random examples...")
        few_shot_block = self._build_few_shot_prompt_block()

        prompt = self.prompt_template.format(
            few_shot_examples=few_shot_block,
            question=question,
            options=options_text,
        )

        outputs = self.model.generate(
            prompt,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
            n=kwargs.get("n", 1),
        )

        raw_output = outputs[0] if outputs else ""
        prediction = self.task.extract_answer(raw_output)

        return {
            "prediction": prediction,
            "output": raw_output,
            "metadata": {
                "prompt": prompt,
                "n_shots": self.n_shots,
                "num_samples": len(outputs),
            },
        }
