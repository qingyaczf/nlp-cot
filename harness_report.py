"""
Harness Engineering Subsystem Coverage Report.

Generates a coverage matrix showing which of the five Harness
subsystems (Instructions/Tools/Environment/State/Feedback) each
COT strategy utilizes.

Usage:
    python harness_report.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from models.base import BaseModel
from tasks.aqua_task import AQuATask
from strategies import (
    BaseCOTStrategy,
    SelfConsistencyStrategy,
    StepAwareVerifierStrategy,
    RAGCOTStrategy,
    MultiAgentDebateStrategy,
    PrefixConsistencyStrategy,
)
from retrieval import SimpleKeywordRetriever


class DummyModel(BaseModel):
    def __init__(self):
        super().__init__("dummy")

    def generate(self, prompt, **kwargs):
        return ["Answer: A"]

    def chat(self, messages, **kwargs):
        return ["Answer: A"]


def main():
    print("=" * 70)
    print("Harness Engineering Subsystem Coverage Matrix")
    print("=" * 70)
    print()
    print("Five subsystems (walkinglabs model):")
    print("  [I] Instructions  — structured prompt templates")
    print("  [T] Tools         — external tools (retriever, verifier, etc.)")
    print("  [E] Environment   — task environment (dataset, evaluation)")
    print("  [S] State         — runtime state tracking")
    print("  [F] Feedback      — feedback loops (critique, verification)")
    print()

    model = DummyModel()
    task = AQuATask(data_dir="data/AQuA")
    retriever = SimpleKeywordRetriever(knowledge_path="data/knowledge_base.json")

    strategies = [
        ("base_cot", BaseCOTStrategy(model, task)),
        ("self_consistency", SelfConsistencyStrategy(model, task)),
        ("prefix_consistency", PrefixConsistencyStrategy(model, task)),
        ("step_verifier", StepAwareVerifierStrategy(model, task)),
        ("rag_cot", RAGCOTStrategy(model, task, retriever)),
        ("multi_agent_debate", MultiAgentDebateStrategy(model, task)),
    ]

    headers = ["Strategy", "I", "T", "E", "S", "F"]
    rows = []
    for name, strat in strategies:
        sub = strat.harness_subsystems()
        row = {
            "Strategy": name,
            "I": "●" if sub["instructions"] else "○",
            "T": "●" if sub["tools"] else "○",
            "E": "●" if sub["environment"] else "○",
            "S": "●" if sub["state"] else "○",
            "F": "●" if sub["feedback"] else "○",
        }
        rows.append(row)

    # Print table
    col_widths = {h: max(len(h), max(len(r[h]) for r in rows)) + 2 for h in headers}
    header_line = "".join(h.center(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("".join(row[h].center(col_widths[h]) for h in headers))

    print()
    print("=" * 70)
    print("Subsystem Evolution Across Strategies")
    print("=" * 70)
    print()
    print("base_cot            → I + E               (baseline)")
    print("self_consistency    → I + E + S           (add state: multiple paths)")
    print("prefix_consistency  → I + E + S + F       (add feedback: prefix regeneration reliability)")
    print("rag_cot             → I + E + S + T       (add tools: retriever)")
    print("step_verifier       → I + E + S + T + F   (add feedback: verifier)")
    print("multi_agent_debate  → I + E + S + F       (add feedback: inter-agent critique)")
    print()
    print("Key insight: advanced strategies progressively activate more Harness")
    print("subsystems. The full 5-subsystem coverage (step_verifier) represents")
    print("the deepest integration of Harness Engineering design philosophy.")
    print("Prefix consistency achieves Feedback coverage without Tools, making it")
    print("a lightweight yet powerful alternative to step_verifier.")
    print("=" * 70)


if __name__ == "__main__":
    main()
