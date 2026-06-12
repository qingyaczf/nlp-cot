"""
Dry-run verification script for feat-011 (Prefix Consistency strategy).
Runs the full pipeline with a MockModel to verify correctness without API calls.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from datetime import datetime

from eval.metrics import compute_metrics
from models.base import BaseModel
from strategies import PrefixConsistencyStrategy
from tasks import AQuATask


class MockModel(BaseModel):
    """
    Mock LLM that returns canned responses for prefix-consistency testing.

    Responses are served in a fixed order matching the expected call sequence:
      1-3: initial paths (A, B, A)
      4-5: regen for path1 -> A, A (consistent)
      6-7: regen for path2 -> C, D (inconsistent)
      8-9: regen for path3 -> A, A (consistent)
    """

    def __init__(self):
        super().__init__(model_name="mock-model")
        self.call_count = 0
        self._response_idx = 0
        self._responses = [
            # Initial generation: 3 paths
            "Step 1: analyze the problem carefully.\nStep 2: compute the value.\nAnswer: A",
            "Let me think through this.\nAfter calculation, the result is clear.\nAnswer: B",
            "Reasoning step by step...\nFinal conclusion.\nAnswer: A",
            # Regeneration for path1 (x2): consistent -> A
            "Continuing the reasoning...\nTherefore, the answer is A.\nAnswer: A",
            "Continuing the reasoning...\nTherefore, the answer is A.\nAnswer: A",
            # Regeneration for path2 (x2): inconsistent -> C, D
            "Wait, let me reconsider.\nActually the answer is C.\nAnswer: C",
            "Hmm, maybe not.\nFinal answer: D.\nAnswer: D",
            # Regeneration for path3 (x2): consistent -> A
            "Continuing the reasoning...\nTherefore, the answer is A.\nAnswer: A",
            "Continuing the reasoning...\nTherefore, the answer is A.\nAnswer: A",
        ]

    def generate(self, prompt, temperature=0.7, max_tokens=1024, stop=None, n=1, **kwargs):
        self.call_count += 1
        out = []
        for _ in range(n):
            # Cycle through predefined responses so multiple examples work
            out.append(self._responses[self._response_idx % len(self._responses)])
            self._response_idx += 1
        return out

    def chat(self, messages, temperature=0.7, max_tokens=1024, stop=None, n=1, **kwargs):
        content = messages[0]["content"] if messages else ""
        return self.generate(content, temperature, max_tokens, stop, n, **kwargs)


def verify():
    print("=== feat-011 Verification (Prefix Consistency Dry-Run) ===\n")

    # 1. Load task and data
    task = AQuATask()
    examples = task.load_data(split="test")
    print(f"[1/7] Loaded {len(examples)} examples from AQuA test set")

    # Use only first 3 examples for quick verification
    examples = examples[:3]

    # 2. Verify prompt formatting
    sample_prompt = task.format_prompt(examples[0])
    assert "Question:" in sample_prompt
    assert "Options:" in sample_prompt
    print(f"[2/7] Prompt formatting OK\n  Sample prompt length: {len(sample_prompt)} chars")

    # 3. Verify answer extraction
    test_cases = [
        ("Some reasoning...\nAnswer: C", "C"),
        ("The answer is B.", "B"),
        ("Correct answer: D", "D"),
        ("I think A is right", "A"),
    ]
    for text, expected in test_cases:
        got = task.extract_answer(text)
        assert got == expected, f"Expected {expected}, got {got} for: {text}"
    print(f"[3/7] Answer extraction OK ({len(test_cases)} cases passed)")

    # 4. Run strategy with MockModel (linear weight)
    model = MockModel()
    strategy = PrefixConsistencyStrategy(
        model=model,
        task=task,
        n_paths=3,
        truncation_ratio=0.5,
        regen_count=2,
        weight_fn="linear",
    )
    results = []
    for ex in examples:
        result = strategy.run(ex, temperature=0.7, max_tokens=512)
        results.append(result)
        # Verify metadata
        assert "all_outputs" in result["metadata"]
        assert "all_predictions" in result["metadata"]
        assert "consistencies" in result["metadata"]
        assert len(result["metadata"]["all_outputs"]) == 3
        assert len(result["metadata"]["consistencies"]) == 3
        # path1 (A) and path3 (A) should be consistent; path2 (B) inconsistent
        consistencies = result["metadata"]["consistencies"]
        assert consistencies[0] == 1.0, f"Expected path1 consistency=1.0, got {consistencies[0]}"
        assert consistencies[1] == 0.0, f"Expected path2 consistency=0.0, got {consistencies[1]}"
        assert consistencies[2] == 1.0, f"Expected path3 consistency=1.0, got {consistencies[2]}"
        # Weighted votes: A gets 1.0+1.0=2.0, B gets 0.0 -> final A
        assert result["prediction"] == "A", f"Expected A, got {result['prediction']}"
    print(f"[4/7] Strategy run OK (linear weight, {model.call_count} mock API calls)")
    print(f"  Consistencies verified: [1.0, 0.0, 1.0] -> weighted A=2.0, B=0.0 -> final=A")

    # 5. Verify weight functions
    for wf in ("quadratic", "cubic", "unanimous"):
        model2 = MockModel()
        strat2 = PrefixConsistencyStrategy(
            model=model2,
            task=task,
            n_paths=3,
            truncation_ratio=0.5,
            regen_count=2,
            weight_fn=wf,
        )
        r = strat2.run(examples[0], temperature=0.7, max_tokens=512)
        # With this mock, unanimous also gives A full weight (1.0 vs 0.0)
        assert r["prediction"] == "A", f"Weight fn {wf} failed: expected A, got {r['prediction']}"
    print(f"[5/7] Alternative weight functions OK (quadratic, cubic, unanimous)")

    # 6. Compute metrics
    metrics = compute_metrics(results, examples)
    assert "accuracy" in metrics
    assert "correct" in metrics
    assert "total" in metrics
    print(f"[6/7] Metrics computed OK")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")

    # 7. Save results to experiments/runs/
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "experiments/runs"
    os.makedirs(output_dir, exist_ok=True)
    run_path = os.path.join(output_dir, f"verify_feat011_{run_id}.json")

    run_record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "verification": True,
        "config": {
            "strategy": "prefix_consistency",
            "task": "aqua",
            "model": "mock-model",
            "n_paths": 3,
            "truncation_ratio": 0.5,
            "regen_count": 2,
            "weight_fn": "linear",
            "n_samples": len(examples),
        },
        "metrics": metrics,
        "results": [
            {
                "prediction": r["prediction"],
                "correct": ex.get("correct", ""),
                "output": r["output"],
            }
            for r, ex in zip(results, examples)
        ],
    }
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_record, f, ensure_ascii=False, indent=2)
    print(f"[7/7] Results saved to {run_path}")

    print("\n=== feat-011 Verification PASSED ===")
    print("The Prefix Consistency strategy pipeline is ready for real API experiments.")
    print("\nNext step: run a real experiment with")
    print("  python harness.py --strategy prefix_consistency --dataset aqua")
    return 0


if __name__ == "__main__":
    sys.exit(verify())
