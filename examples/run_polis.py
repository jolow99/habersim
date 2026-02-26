"""
Example: Run a Polis-like deliberation simulation.

Demonstrates the Polis architecture where agents vote agree/disagree/pass
on statements and are clustered into opinion groups. Uses a mock LLM client
so you can run it without any API keys.

The mock agents simulate two camps on AI governance:
- Safety-first camp (alice, carol): tend to agree on regulation statements
- Innovation-first camp (bob, dave): tend to disagree on regulation statements
"""

import json

from habersim.core import Action, LLMClient, Perception
from habersim.architectures.polis import PolisArchitecture
from habersim.agents import GenericLLMAgent
from habersim.simulation import Simulation, SimulationConfig


# ---------------------------------------------------------------------------
# Mock LLM client — deterministic responses for Polis
# ---------------------------------------------------------------------------

class MockPolisLLMClient(LLMClient):
    """
    Scripted responses for a Polis deliberation. Each agent follows a sequence:
    1. Submit a seed statement
    2-N. Vote on statements they haven't seen yet

    Agents have predefined stances that create two clear opinion clusters:
    - alice & carol: pro-regulation (agree with safety statements, disagree with speed)
    - bob & dave: pro-innovation (disagree with safety statements, agree with speed)
    """

    # Seed statements each agent will propose
    SEED_STATEMENTS = {
        "alice": "Frontier AI models should require government approval before deployment.",
        "bob": "AI regulation will inevitably slow progress and benefit less regulated competitors.",
        "carol": "Mandatory safety testing by independent auditors should be required for all large models.",
        "dave": "Open-source AI development should not be restricted by regulation.",
    }

    # How each agent votes: maps keyword patterns to votes.
    # If a statement contains a keyword, the agent votes accordingly.
    VOTE_TENDENCIES = {
        "alice": {"approval": "agree", "regulation": "agree", "safety": "agree",
                  "slow": "disagree", "open-source": "pass", "restrict": "agree",
                  "auditor": "agree", "benefit": "disagree"},
        "bob":   {"approval": "disagree", "regulation": "disagree", "safety": "pass",
                  "slow": "agree", "open-source": "agree", "restrict": "disagree",
                  "auditor": "disagree", "benefit": "agree"},
        "carol": {"approval": "agree", "regulation": "agree", "safety": "agree",
                  "slow": "disagree", "open-source": "pass", "restrict": "pass",
                  "auditor": "agree", "benefit": "disagree"},
        "dave":  {"approval": "disagree", "regulation": "disagree", "safety": "pass",
                  "slow": "agree", "open-source": "agree", "restrict": "disagree",
                  "auditor": "disagree", "benefit": "agree"},
    }

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._submitted_statement = False

    def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        return ""

    def complete_json(self, prompt: str, schema: dict, system: str = "", **kwargs) -> dict:
        # First action: submit a seed statement (if we haven't yet)
        if not self._submitted_statement:
            self._submitted_statement = True
            return {
                "reasoning": "I want to add my perspective to the discussion.",
                "action_type": "submit_statement",
                "payload": {"text": self.SEED_STATEMENTS[self.agent_id]},
            }

        # Subsequent actions: vote on the first unvoted statement
        # Parse the available actions to find the vote action
        if '"vote"' in prompt and '"statement_id"' in prompt:
            # Extract statement_id from the prompt
            import re
            sid_match = re.search(r'"const":\s*"([a-f0-9-]{36})"', prompt)
            if sid_match:
                sid = sid_match.group(1)
                # Find the statement text to determine our vote
                # Look for the statement text near the vote action
                text_match = re.search(r'Vote on statement:\s*"([^"]+)"', prompt)
                statement_text = text_match.group(1).lower() if text_match else ""

                vote = self._decide_vote(statement_text)
                return {
                    "reasoning": f"Based on my stance, I {vote} this statement.",
                    "action_type": "vote",
                    "payload": {"statement_id": sid, "vote": vote},
                }

        # Fallback: submit another statement (shouldn't normally happen)
        return {
            "reasoning": "Nothing to vote on, submitting a thought.",
            "action_type": "submit_statement",
            "payload": {"text": "We need more diverse perspectives in this discussion."},
        }

    def _decide_vote(self, statement_text: str) -> str:
        """Determine vote based on keyword matching against agent tendencies."""
        tendencies = self.VOTE_TENDENCIES[self.agent_id]
        for keyword, vote in tendencies.items():
            if keyword in statement_text:
                return vote
        return "pass"  # default if no keyword matches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_example():
    """Run a Polis deliberation with 4 mock agents on AI governance."""
    topic = "How should society govern the development of advanced AI systems?"

    # Use 2 clusters — we expect to see a safety-first vs innovation-first split
    architecture = PolisArchitecture(n_clusters=2)

    agents = [
        GenericLLMAgent("alice", MockPolisLLMClient("alice"), persona="AI safety researcher"),
        GenericLLMAgent("bob",   MockPolisLLMClient("bob"),   persona="AI startup founder"),
        GenericLLMAgent("carol", MockPolisLLMClient("carol"), persona="Policy analyst"),
        GenericLLMAgent("dave",  MockPolisLLMClient("dave"),  persona="Open-source advocate"),
    ]

    config = SimulationConfig(
        max_rounds=5,  # round 1: submit statements, rounds 2-5: vote
        max_actions_per_agent_per_round=1,
        verbose=True,
    )

    sim = Simulation(architecture, agents, topic, config)
    final_state = sim.run()

    print("\n")
    sim.print_transcript()

    # Print opinion groups
    print("\n\n--- Opinion Groups ---")
    if final_state.groups:
        for group in final_state.groups:
            print(f"\nGroup {group.group_id}: {group.agent_ids}")
            for sid, vote in group.representative_votes.items():
                stmt = final_state.statement_by_id(sid)
                from habersim.architectures.polis import VOTE_NAMES
                vote_name = VOTE_NAMES[vote]
                text = stmt.text if stmt else sid
                print(f"  {vote_name:>8}: {text}")

    print("\n\n--- Full simulation data (JSON) ---")
    data = sim.to_dict()
    print(json.dumps(data, indent=2, default=str))

    return final_state, sim


if __name__ == "__main__":
    run_example()
