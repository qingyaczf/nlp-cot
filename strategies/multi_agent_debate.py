"""
Multi-Agent Debate strategy (improved).
Based on: https://arxiv.org/abs/2305.14325

Key improvements over baseline:
- 5 agents with distinct roles + different temperatures for diversity
- Parallel execution via ThreadPoolExecutor (all agents run simultaneously)
- Cross-review phase: each agent critiques all others' answers
- Revision phase: agents revise based on full debate context
- Convergence detection: stop early if all agents agree
- Majority vote aggregation

Harness Engineering integration:
- Instructions: 5 distinct role prompts per agent
- State: per-agent, per-round reasoning + convergence tracking
- Feedback: inter-agent critique loop + convergence detection
- Environment: debate arena where agents interact
"""
import concurrent.futures
import os
from collections import Counter
from typing import Dict, Any, List

from .base import BaseStrategy


class MultiAgentDebateStrategy(BaseStrategy):
    """Multi-Agent Debate: 5 agents, parallel, cross-review, convergence."""

    def harness_subsystems(self) -> Dict[str, bool]:
        return {
            "instructions": True,
            "tools": False,
            "environment": True,
            "state": True,
            "feedback": True,
        }

    def __init__(
        self,
        model,
        task,
        n_agents: int = 5,
        n_rounds: int = 3,
        temperature: float = 0.7,
        **kwargs
    ):
        super().__init__(name="multi_agent_debate", model=model, task=task, **kwargs)
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.temperature = temperature

        # Agent definitions: (name, role_system_prompt, temperature)
        self.agent_configs = self._build_agent_configs()

    def _build_agent_configs(self) -> List[Dict[str, Any]]:
        """Build diverse agent configs with different roles and temperatures."""
        role_pool = [
            ("分析师", "你是逻辑严密的数学分析师。面对选择题，你总是先拆解条件、逐步推导、代入验证。", 0.3),
            ("批判者", "你是挑剔的批判型思考者。你习惯质疑常规思路，寻找陷阱和边界情况，指出他人推理中的漏洞。", 0.8),
            ("直觉者", "你依靠数学直觉和经验快速判断。你擅长模式识别，看到题目就能联想到经典题型和常见解法。", 1.2),
            ("验证者", "你是严谨的验证者。你总是通过代入具体数值、检查边界条件、反证法来验证答案的正确性。", 0.5),
            ("综合者", "你是善于整合信息的综合者。你倾听各方观点，取其精华去其糟粕，权衡各种推理路径后给出最可靠的答案。", 0.7),
        ]
        configs = []
        for i in range(self.n_agents):
            name, role, temp = role_pool[i % len(role_pool)]
            configs.append({
                "id": i,
                "name": name,
                "system_prompt": f"你是{name}。{role}\n最后一行以 'Answer: X' 给出你的最终答案（X为A/B/C/D/E之一）。",
                "temperature": temp,
            })
        return configs

    def _generate(self, agent_cfg: Dict, prompt: str) -> str:
        """Generate a response for one agent."""
        outputs = self.model.generate(
            prompt=prompt,
            temperature=agent_cfg["temperature"],
            max_tokens=1024,
            n=1,
            system_prompt=agent_cfg["system_prompt"],
        )
        return outputs[0] if outputs else ""

    def _parallel_generate(self, prompt: str) -> Dict[int, str]:
        """Run all agents in parallel and return {agent_id: response}."""
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_agents) as ex:
            futures = {
                ex.submit(self._generate, cfg, prompt): cfg["id"]
                for cfg in self.agent_configs
            }
            for f in concurrent.futures.as_completed(futures):
                agent_id = futures[f]
                results[agent_id] = f.result()
        return results

    def _cross_review(
        self,
        question: str,
        options_text: str,
        answers: Dict[int, str],
    ) -> Dict[int, str]:
        """Each agent critiques all other agents' answers."""
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_agents) as ex:
            def review_one(cfg):
                others = {
                    c["id"]: answers.get(c["id"], "")
                    for c in self.agent_configs
                    if c["id"] != cfg["id"]
                }
                if not others:
                    return cfg["id"], ""
                others_text = "\n\n".join(
                    f"[{self.agent_configs[k]['name']}]: {v}"
                    for k, v in others.items()
                )
                prompt = (
                    f"请审阅以下其他智能体对选择题的回答，指出其中的逻辑漏洞、"
                    f"计算错误或值得商榷之处。不要给出你的最终答案。\n\n"
                    f"题目: {question}\n"
                    f"选项: {options_text}\n\n"
                    f"其他人的回答:\n{others_text}"
                )
                return cfg["id"], self._generate(cfg, prompt)

            futures = {ex.submit(review_one, cfg): cfg["id"] for cfg in self.agent_configs}
            for f in concurrent.futures.as_completed(futures):
                aid, critique = f.result()
                results[aid] = critique
        return results

    def _format_debate_context(
        self,
        answers: Dict[int, str],
        critiques: Dict[int, str],
    ) -> str:
        """Format answers + critiques into a context block for the next round."""
        parts = []
        for cfg in self.agent_configs:
            aid = cfg["id"]
            ans = answers.get(aid, "")
            crit = critiques.get(aid, "")
            parts.append(
                f"### {cfg['name']}\n"
                f"回答: {ans}\n"
                f"对其他人的审阅: {crit}\n"
            )
        return "\n".join(parts)

    def _check_convergence(self, old_answers: Dict[int, str], new_answers: Dict[int, str]) -> bool:
        """Check if all agents' extracted answers are identical across rounds."""
        old_set = {self.task.extract_answer(v) for v in old_answers.values()}
        new_set = {self.task.extract_answer(v) for v in new_answers.values()}
        return old_set == new_set and len(new_set) == 1

    def _majority_vote(self, answers: Dict[int, str]) -> str:
        """Extract answers and return majority vote."""
        predictions = [self.task.extract_answer(v) for v in answers.values()]
        predictions = [p for p in predictions if p]
        if not predictions:
            return ""
        vote_counts = Counter(predictions)
        return vote_counts.most_common(1)[0][0]

    def run(self, example: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        question = example.get("question", "")
        options = example.get("options", [])
        options_text = " ".join(options)

        n_rounds = kwargs.get("n_rounds", self.n_rounds)

        # Round 1: initial answers from all agents
        round1_prompt = (
            f"请回答以下选择题。先逐步分析推理，"
            f"最后一行以 'Answer: X' 给出正确选项字母 (A/B/C/D/E)。\n\n"
            f"题目: {question}\n"
            f"选项: {options_text}"
        )
        answers = self._parallel_generate(round1_prompt)
        debate_rounds = [
            {cfg["name"]: {"pred": self.task.extract_answer(answers.get(cfg["id"], ""))}
             for cfg in self.agent_configs}
        ]

        # Rounds 2+: cross-review + revise
        for r in range(1, n_rounds):
            # Cross-review
            critiques = self._cross_review(question, options_text, answers)
            # Build context
            context = self._format_debate_context(answers, critiques)
            # Revise
            revise_prompt = (
                f"基于以上辩论记录，重新审视原题并给出你的最终答案。\n"
                f"如果你认为其他智能体的观点更有道理，请修正你的答案；\n"
                f"如果坚持原答案，请给出更充分的反驳论据。\n"
                f"最后一行以 'Answer: X' 给出正确选项字母。\n\n"
                f"原题: {question}\n"
                f"选项: {options_text}\n\n"
                f"辩论记录:\n{context}"
            )
            new_answers = self._parallel_generate(revise_prompt)

            # Track round
            debate_rounds.append({
                cfg["name"]: {"pred": self.task.extract_answer(new_answers.get(cfg["id"], ""))}
                for cfg in self.agent_configs
            })

            # Convergence check
            if self._check_convergence(answers, new_answers):
                answers = new_answers
                break
            answers = new_answers

        # Final aggregation
        final_prediction = self._majority_vote(answers)
        vote_counts = Counter(
            self.task.extract_answer(v) for v in answers.values()
        )

        # Build summary
        summary_lines = [
            f"=== Multi-Agent Debate ({self.n_agents} agents, {n_rounds} rounds) ===",
            f"Final Answer: {final_prediction}",
            f"Votes: {dict(vote_counts)}",
            "",
        ]
        for ri, rd in enumerate(debate_rounds):
            summary_lines.append(f"--- Round {ri + 1} ---")
            for name, info in rd.items():
                summary_lines.append(f"  {name}: -> {info['pred']}")

        return {
            "prediction": final_prediction,
            "output": "\n".join(summary_lines),
            "metadata": {
                "n_agents": self.n_agents,
                "n_rounds": min(len(debate_rounds), n_rounds),
                "debate_rounds": debate_rounds,
                "vote_counts": dict(vote_counts),
                "final_answers": {
                    cfg["name"]: self.task.extract_answer(answers.get(cfg["id"], ""))
                    for cfg in self.agent_configs
                },
            },
        }
