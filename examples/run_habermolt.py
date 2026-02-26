"""
Example: Run a Habermolt deliberation simulation.

This script shows how to wire together architecture, agents, and the
simulation runner. Uses a mock LLM client so you can run it without
any API keys to verify the framework works end-to-end.
"""

import json
import re

from habersim.core import Action, LLMClient, Perception
from habersim.architectures.habermolt import HabermoltArchitecture
from habersim.agents import GenericLLMAgent
from habersim.simulation import Simulation, SimulationConfig


# ---------------------------------------------------------------------------
# Mock LLM client — deterministic responses for testing
# ---------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """
    Scripted responses for each agent. In a real simulation, replace with
    AnthropicClient or OpenAIClient.

    Each agent has a predefined sequence of actions:
    1. Submit an opinion on the topic
    2. Propose a consensus statement
    3. Rank all statements in their preferred order
    """

    SCRIPTS = {
        "alice": [
            {"reasoning": "I should share my view first.", "action_type": "submit_opinion",
             "payload": {"opinion": "AI development should prioritise safety and interpretability above speed."}},
            {"reasoning": "Let me propose a concrete statement.", "action_type": "add_statement",
             "payload": {"text": "We should establish mandatory safety audits before deploying frontier AI models."}},
            {"reasoning": "I'll rank the statements.", "action_type": "update_ranking",
             "payload": {"ranking": []}},  # filled dynamically
        ],
        "bob": [
            {"reasoning": "I have a different view on this.", "action_type": "submit_opinion",
             "payload": {"opinion": "Slowing down AI risks ceding ground to less safety-conscious actors. Speed matters."}},
            {"reasoning": "Here's my alternative statement.", "action_type": "add_statement",
             "payload": {"text": "International coordination, not unilateral slowdowns, is the right approach to AI safety."}},
            {"reasoning": "Ranking time.", "action_type": "update_ranking",
             "payload": {"ranking": []}},  # filled dynamically
        ],
        "carol": [
            {"reasoning": "Both sides have merit.", "action_type": "submit_opinion",
             "payload": {"opinion": "Safety and progress are not fundamentally opposed — good governance can enable both."}},
            {"reasoning": "Let me add a bridging statement.", "action_type": "add_statement",
             "payload": {"text": "AI labs should publish safety benchmarks transparently while continuing development."}},
            {"reasoning": "Here's my ranking.", "action_type": "update_ranking",
             "payload": {"ranking": []}},  # filled dynamically
        ],
    }

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._call_count = 0

    def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        return ""

    def complete_json(self, prompt: str, schema: dict, system: str = "", **kwargs) -> dict:
        scripts = self.SCRIPTS[self.agent_id]
        idx = min(self._call_count, len(scripts) - 1)
        self._call_count += 1

        result = scripts[idx].copy()

        # For ranking actions, extract statement IDs from the prompt context
        if result["action_type"] == "update_ranking":
            ids = re.findall(r'"id":\s*"([a-f0-9-]{36})"', prompt)
            # Each agent has a slightly different preference order
            if self.agent_id == "alice":
                order = ids  # alice prefers them in proposal order
            elif self.agent_id == "bob":
                order = list(reversed(ids))  # bob reverses
            else:
                order = ids[1:] + ids[:1] if len(ids) > 1 else ids  # carol rotates
            result["payload"] = {"ranking": order}

        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_example():
    """Run a 3-round Habermolt deliberation with mock agents."""
    topic = "How should society govern the development of advanced AI systems?"

    architecture = HabermoltArchitecture()

    agents = [
        GenericLLMAgent("alice", MockLLMClient("alice"), persona="AI safety researcher"),
        GenericLLMAgent("bob",   MockLLMClient("bob"),   persona="AI entrepreneur"),
        GenericLLMAgent("carol", MockLLMClient("carol"), persona="Policy analyst"),
    ]

    config = SimulationConfig(
        max_rounds=3,
        max_actions_per_agent_per_round=1,
        verbose=True,
    )

    sim = Simulation(architecture, agents, topic, config)
    final_state = sim.run()

    print("\n")
    sim.print_transcript()

    print("\n\n--- Full simulation data (JSON) ---")
    data = sim.to_dict()
    print(json.dumps(data, indent=2, default=str))

    return final_state, sim


if __name__ == "__main__":
    run_example()
