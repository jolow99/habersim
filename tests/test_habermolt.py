"""Tests for the Habermolt architecture."""

import pytest
from habersim.core import Action, AgentID, LLMClient, Perception
from habersim.core.base import Agent
from habersim.architectures.habermolt import (
    HabermoltArchitecture,
    HabermoltState,
    schulze_ranking,
)
from habersim.simulation import Simulation, SimulationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockAgent(Agent):
    """Agent with a scripted sequence of actions."""

    def __init__(self, agent_id: str, actions: list[dict]):
        super().__init__(agent_id)
        self._actions = actions
        self._idx = 0

    def act(self, perception: Perception) -> Action:
        if self._idx >= len(self._actions):
            # Repeat last action
            data = self._actions[-1]
        else:
            data = self._actions[self._idx]
        self._idx += 1

        payload = dict(data.get("payload", {}))

        # Dynamically fill ranking with actual statement IDs
        if data["action_type"] == "update_ranking" and not payload.get("ranking"):
            sids = [s["id"] for s in perception.context.get("statements", [])]
            order_fn = data.get("order_fn")
            payload["ranking"] = order_fn(sids) if order_fn else sids

        return Action(
            agent_id=self.agent_id,
            action_type=data["action_type"],
            payload=payload,
        )


# ---------------------------------------------------------------------------
# Schulze method tests
# ---------------------------------------------------------------------------

class TestSchulzeRanking:
    def test_empty(self):
        assert schulze_ranking([], {}) == []

    def test_single_statement(self):
        result = schulze_ranking(["A"], {"v1": ["A"]})
        assert result == ["A"]

    def test_unanimous(self):
        """All voters agree: A > B > C."""
        rankings = {
            "v1": ["A", "B", "C"],
            "v2": ["A", "B", "C"],
            "v3": ["A", "B", "C"],
        }
        result = schulze_ranking(["A", "B", "C"], rankings)
        assert result == ["A", "B", "C"]

    def test_condorcet_winner(self):
        """A beats both B and C pairwise."""
        rankings = {
            "v1": ["A", "B", "C"],
            "v2": ["A", "C", "B"],
            "v3": ["B", "A", "C"],
        }
        result = schulze_ranking(["A", "B", "C"], rankings)
        assert result[0] == "A"  # A is the Condorcet winner

    def test_known_schulze_example(self):
        """Classic Schulze example with a cycle: A>B>C>A among different voter groups."""
        # 3 voters: A>B>C, 2 voters: B>C>A, 2 voters: C>A>B
        rankings = {}
        for i in range(3):
            rankings[f"g1_{i}"] = ["A", "B", "C"]
        for i in range(2):
            rankings[f"g2_{i}"] = ["B", "C", "A"]
        for i in range(2):
            rankings[f"g3_{i}"] = ["C", "A", "B"]

        result = schulze_ranking(["A", "B", "C"], rankings)
        # A beats B 5-2, B beats C 5-2, A beats C 3-4 (C beats A)
        # But Schulze should still rank A first due to strongest paths
        assert result[0] == "A"

    def test_two_candidates_majority(self):
        rankings = {
            "v1": ["X", "Y"],
            "v2": ["X", "Y"],
            "v3": ["Y", "X"],
        }
        result = schulze_ranking(["X", "Y"], rankings)
        assert result == ["X", "Y"]


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestHabermoltArchitecture:
    def setup_method(self):
        self.arch = HabermoltArchitecture()

    def test_initial_state(self):
        state = self.arch.initial_state("topic", ["a", "b"])
        assert state.topic == "topic"
        assert state.participants == ["a", "b"]
        assert state.statements == []
        assert state.rankings == {}

    def test_submit_opinion(self):
        state = self.arch.initial_state("topic", ["alice"])
        action = Action(agent_id="alice", action_type="submit_opinion",
                       payload={"opinion": "my opinion"})
        state = self.arch.update(state, action)
        assert state.opinions["alice"] == "my opinion"
        assert len(state.contributions) == 1

    def test_add_statement(self):
        state = self.arch.initial_state("topic", ["alice", "bob"])
        action = Action(agent_id="alice", action_type="add_statement",
                       payload={"text": "Statement 1"})
        state = self.arch.update(state, action)
        assert len(state.statements) == 1
        assert state.statements[0].text == "Statement 1"
        assert state.statements[0].author == "alice"

    def test_add_statement_empty_text_raises(self):
        state = self.arch.initial_state("topic", ["alice"])
        action = Action(agent_id="alice", action_type="add_statement",
                       payload={"text": ""})
        with pytest.raises(ValueError, match="empty"):
            self.arch.update(state, action)

    def test_predicted_ranking_on_new_statement(self):
        state = self.arch.initial_state("topic", ["alice", "bob"])

        # Alice adds a statement
        action = Action(agent_id="alice", action_type="add_statement",
                       payload={"text": "S1"})
        state = self.arch.update(state, action)
        sid1 = state.statements[0].id

        # Both should have predicted rankings containing the new statement
        assert sid1 in state.predicted_rankings.get("alice", [])
        assert sid1 in state.predicted_rankings.get("bob", [])

    def test_update_ranking_clears_predicted(self):
        state = self.arch.initial_state("topic", ["alice"])
        action = Action(agent_id="alice", action_type="add_statement",
                       payload={"text": "S1"})
        state = self.arch.update(state, action)
        sid = state.statements[0].id

        # Now alice explicitly ranks
        action = Action(agent_id="alice", action_type="update_ranking",
                       payload={"ranking": [sid]})
        state = self.arch.update(state, action)

        assert state.rankings["alice"] == [sid]
        assert "alice" not in state.predicted_rankings

    def test_validate_ranking_rejects_unknown_ids(self):
        state = self.arch.initial_state("topic", ["alice"])
        action = Action(agent_id="alice", action_type="update_ranking",
                       payload={"ranking": ["nonexistent-id"]})
        with pytest.raises(ValueError, match="unknown"):
            self.arch.update(state, action)

    def test_unknown_action_type_raises(self):
        state = self.arch.initial_state("topic", ["alice"])
        action = Action(agent_id="alice", action_type="dance",
                       payload={})
        with pytest.raises(ValueError, match="Unknown action"):
            self.arch.update(state, action)

    def test_perceive_shows_submit_opinion_when_none(self):
        state = self.arch.initial_state("topic", ["alice"])
        perception = self.arch.perceive("alice", state)
        action_names = [a.name for a in perception.available_actions]
        assert "submit_opinion" in action_names

    def test_perceive_hides_submit_opinion_after_submitted(self):
        state = self.arch.initial_state("topic", ["alice"])
        action = Action(agent_id="alice", action_type="submit_opinion",
                       payload={"opinion": "my take"})
        state = self.arch.update(state, action)
        perception = self.arch.perceive("alice", state)
        action_names = [a.name for a in perception.available_actions]
        assert "submit_opinion" not in action_names

    def test_is_terminal_always_false(self):
        state = self.arch.initial_state("topic", ["alice"])
        assert self.arch.is_terminal(state) is False

    def test_aggregate_returns_none_when_no_rankings(self):
        state = self.arch.initial_state("topic", ["alice"])
        assert self.arch.aggregate(state) is None


# ---------------------------------------------------------------------------
# Full simulation test
# ---------------------------------------------------------------------------

class TestHabermoltSimulation:
    def test_full_simulation_with_mock_agents(self):
        arch = HabermoltArchitecture()

        agents = [
            MockAgent("alice", [
                {"action_type": "submit_opinion", "payload": {"opinion": "Safety first"}},
                {"action_type": "add_statement", "payload": {"text": "Mandatory safety audits"}},
                {"action_type": "update_ranking", "payload": {}, "order_fn": lambda ids: ids},
            ]),
            MockAgent("bob", [
                {"action_type": "submit_opinion", "payload": {"opinion": "Speed matters"}},
                {"action_type": "add_statement", "payload": {"text": "International coordination"}},
                {"action_type": "update_ranking", "payload": {}, "order_fn": lambda ids: list(reversed(ids))},
            ]),
        ]

        config = SimulationConfig(max_rounds=3, verbose=False)
        sim = Simulation(arch, agents, "AI governance", config)
        state = sim.run()

        # Both agents submitted opinions
        assert "alice" in state.opinions
        assert "bob" in state.opinions

        # Two statements were added
        assert len(state.statements) == 2

        # Both agents ranked
        assert "alice" in state.rankings
        assert "bob" in state.rankings

        # Collective output exists
        assert state.collective_output is not None
        assert "winner_id" in state.collective_output

        # Event log is populated
        assert len(state.contributions) > 0
