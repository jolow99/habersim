"""Tests for the Polis architecture."""

import pytest
from habersim.core import Action, Perception
from habersim.core.base import Agent
from habersim.architectures.polis import (
    AGREE,
    DISAGREE,
    PASS,
    OpinionGroup,
    PolisArchitecture,
    PolisState,
    PolisStatement,
    cluster_agents,
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
            data = self._actions[-1]
        else:
            data = self._actions[self._idx]
        self._idx += 1

        payload = dict(data.get("payload", {}))

        # Dynamically fill statement_id for votes if needed
        if data["action_type"] == "vote" and "statement_id" not in payload:
            unvoted = perception.context.get("unvoted_statements", [])
            if unvoted:
                payload["statement_id"] = unvoted[0]["id"]

        return Action(
            agent_id=self.agent_id,
            action_type=data["action_type"],
            payload=payload,
        )


def _make_state_with_statements(n_statements: int = 3, participants: list[str] | None = None) -> PolisState:
    """Helper to create a PolisState with pre-populated statements."""
    participants = participants or ["alice", "bob", "carol"]
    state = PolisState(topic="test", participants=participants)
    for i in range(n_statements):
        state.statements.append(PolisStatement(
            id=f"s{i}",
            text=f"Statement {i}",
            author=participants[i % len(participants)],
        ))
    return state


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------

class TestPolisState:
    def test_vote_vector_empty(self):
        state = _make_state_with_statements(3)
        vec = state.vote_vector("alice")
        assert vec == [0, 0, 0]

    def test_vote_vector_with_votes(self):
        state = _make_state_with_statements(3)
        state.votes["alice"] = {"s0": AGREE, "s1": DISAGREE, "s2": PASS}
        vec = state.vote_vector("alice")
        assert vec == [1, -1, 0]

    def test_vote_vector_partial(self):
        state = _make_state_with_statements(3)
        state.votes["alice"] = {"s0": AGREE}
        vec = state.vote_vector("alice")
        assert vec == [1, 0, 0]  # unvoted = 0

    def test_unvoted_statements(self):
        state = _make_state_with_statements(3)
        state.votes["alice"] = {"s0": AGREE}
        unvoted = state.unvoted_statements("alice")
        assert len(unvoted) == 2
        assert unvoted[0].id == "s1"

    def test_statement_by_id(self):
        state = _make_state_with_statements(3)
        s = state.statement_by_id("s1")
        assert s is not None
        assert s.text == "Statement 1"

    def test_statement_by_id_not_found(self):
        state = _make_state_with_statements(1)
        assert state.statement_by_id("nonexistent") is None


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestPolisArchitecture:
    def setup_method(self):
        self.arch = PolisArchitecture(n_clusters=2)

    def test_initial_state(self):
        state = self.arch.initial_state("topic", ["a", "b"])
        assert state.topic == "topic"
        assert state.participants == ["a", "b"]
        assert state.statements == []
        assert state.votes == {}

    def test_submit_statement(self):
        state = self.arch.initial_state("topic", ["alice"])
        action = Action(agent_id="alice", action_type="submit_statement",
                       payload={"text": "Test statement"})
        state = self.arch.update(state, action)
        assert len(state.statements) == 1
        assert state.statements[0].text == "Test statement"
        assert state.statements[0].author == "alice"

    def test_submit_empty_statement_raises(self):
        state = self.arch.initial_state("topic", ["alice"])
        action = Action(agent_id="alice", action_type="submit_statement",
                       payload={"text": ""})
        with pytest.raises(ValueError, match="empty"):
            self.arch.update(state, action)

    def test_vote_agree(self):
        state = self.arch.initial_state("topic", ["alice"])
        # Add a statement first
        state = self.arch.update(state, Action(
            agent_id="alice", action_type="submit_statement",
            payload={"text": "S1"},
        ))
        sid = state.statements[0].id

        # Vote on it
        state = self.arch.update(state, Action(
            agent_id="alice", action_type="vote",
            payload={"statement_id": sid, "vote": "agree"},
        ))
        assert state.votes["alice"][sid] == AGREE

    def test_vote_disagree(self):
        state = self.arch.initial_state("topic", ["alice"])
        state = self.arch.update(state, Action(
            agent_id="alice", action_type="submit_statement",
            payload={"text": "S1"},
        ))
        sid = state.statements[0].id
        state = self.arch.update(state, Action(
            agent_id="alice", action_type="vote",
            payload={"statement_id": sid, "vote": "disagree"},
        ))
        assert state.votes["alice"][sid] == DISAGREE

    def test_vote_pass(self):
        state = self.arch.initial_state("topic", ["alice"])
        state = self.arch.update(state, Action(
            agent_id="alice", action_type="submit_statement",
            payload={"text": "S1"},
        ))
        sid = state.statements[0].id
        state = self.arch.update(state, Action(
            agent_id="alice", action_type="vote",
            payload={"statement_id": sid, "vote": "pass"},
        ))
        assert state.votes["alice"][sid] == PASS

    def test_vote_invalid_raises(self):
        state = self.arch.initial_state("topic", ["alice"])
        state = self.arch.update(state, Action(
            agent_id="alice", action_type="submit_statement",
            payload={"text": "S1"},
        ))
        sid = state.statements[0].id
        with pytest.raises(ValueError, match="Invalid vote"):
            self.arch.update(state, Action(
                agent_id="alice", action_type="vote",
                payload={"statement_id": sid, "vote": "maybe"},
            ))

    def test_vote_unknown_statement_raises(self):
        state = self.arch.initial_state("topic", ["alice"])
        with pytest.raises(ValueError, match="Unknown statement"):
            self.arch.update(state, Action(
                agent_id="alice", action_type="vote",
                payload={"statement_id": "nonexistent", "vote": "agree"},
            ))

    def test_unknown_action_raises(self):
        state = self.arch.initial_state("topic", ["alice"])
        with pytest.raises(ValueError, match="Unknown action"):
            self.arch.update(state, Action(
                agent_id="alice", action_type="dance", payload={},
            ))

    def test_is_terminal_always_false(self):
        state = self.arch.initial_state("topic", ["alice"])
        assert self.arch.is_terminal(state) is False

    def test_perceive_shows_unvoted(self):
        state = self.arch.initial_state("topic", ["alice"])
        state = self.arch.update(state, Action(
            agent_id="alice", action_type="submit_statement",
            payload={"text": "S1"},
        ))
        p = self.arch.perceive("alice", state)
        assert len(p.context["unvoted_statements"]) == 1
        # vote action should be available
        action_names = [a.name for a in p.available_actions]
        assert "vote" in action_names

    def test_perceive_hides_voted(self):
        state = self.arch.initial_state("topic", ["alice"])
        state = self.arch.update(state, Action(
            agent_id="alice", action_type="submit_statement",
            payload={"text": "S1"},
        ))
        sid = state.statements[0].id
        state = self.arch.update(state, Action(
            agent_id="alice", action_type="vote",
            payload={"statement_id": sid, "vote": "agree"},
        ))
        p = self.arch.perceive("alice", state)
        assert len(p.context["unvoted_statements"]) == 0


# ---------------------------------------------------------------------------
# Clustering tests
# ---------------------------------------------------------------------------

class TestClustering:
    def test_empty_state(self):
        state = PolisState(topic="test", participants=[])
        groups = cluster_agents(state)
        assert groups == []

    def test_no_statements(self):
        state = PolisState(topic="test", participants=["a", "b"])
        groups = cluster_agents(state)
        assert groups == []

    def test_two_opposing_clusters(self):
        """Two agents with opposite votes should end up in different clusters."""
        state = _make_state_with_statements(3, ["alice", "bob"])
        state.votes["alice"] = {"s0": AGREE, "s1": AGREE, "s2": AGREE}
        state.votes["bob"] = {"s0": DISAGREE, "s1": DISAGREE, "s2": DISAGREE}

        groups = cluster_agents(state, n_clusters=2)
        assert len(groups) == 2

        # Each group should have exactly one agent
        all_members = []
        for g in groups:
            all_members.extend(g.agent_ids)
        assert sorted(all_members) == ["alice", "bob"]

    def test_similar_agents_same_cluster(self):
        """Agents with identical votes should be in the same cluster."""
        state = _make_state_with_statements(3, ["alice", "bob", "carol"])
        state.votes["alice"] = {"s0": AGREE, "s1": AGREE, "s2": DISAGREE}
        state.votes["bob"] = {"s0": AGREE, "s1": AGREE, "s2": DISAGREE}
        state.votes["carol"] = {"s0": DISAGREE, "s1": DISAGREE, "s2": AGREE}

        groups = cluster_agents(state, n_clusters=2)
        assert len(groups) == 2

        # alice and bob should be together
        for g in groups:
            if "alice" in g.agent_ids:
                assert "bob" in g.agent_ids
                assert "carol" not in g.agent_ids

    def test_cluster_output_shape(self):
        """Verify the structure of OpinionGroup objects."""
        state = _make_state_with_statements(2, ["a", "b"])
        state.votes["a"] = {"s0": AGREE, "s1": DISAGREE}
        state.votes["b"] = {"s0": DISAGREE, "s1": AGREE}

        groups = cluster_agents(state, n_clusters=2)
        for g in groups:
            assert isinstance(g, OpinionGroup)
            assert isinstance(g.group_id, int)
            assert isinstance(g.agent_ids, list)
            assert isinstance(g.representative_votes, dict)
            # Representative votes should cover all statements
            assert set(g.representative_votes.keys()) == {"s0", "s1"}

    def test_single_agent_single_cluster(self):
        state = _make_state_with_statements(2, ["alice"])
        state.votes["alice"] = {"s0": AGREE, "s1": DISAGREE}
        groups = cluster_agents(state, n_clusters=2)
        # Can't have more clusters than agents
        assert len(groups) == 1
        assert groups[0].agent_ids == ["alice"]


# ---------------------------------------------------------------------------
# Full simulation test
# ---------------------------------------------------------------------------

class TestPolisSimulation:
    def test_full_simulation_with_mock_agents(self):
        arch = PolisArchitecture(n_clusters=2)

        agents = [
            MockAgent("alice", [
                {"action_type": "submit_statement", "payload": {"text": "We need regulation"}},
                {"action_type": "vote", "payload": {"vote": "agree"}},
                {"action_type": "vote", "payload": {"vote": "agree"}},
            ]),
            MockAgent("bob", [
                {"action_type": "submit_statement", "payload": {"text": "Innovation first"}},
                {"action_type": "vote", "payload": {"vote": "disagree"}},
                {"action_type": "vote", "payload": {"vote": "disagree"}},
            ]),
        ]

        config = SimulationConfig(max_rounds=3, verbose=False)
        sim = Simulation(arch, agents, "AI governance", config)
        state = sim.run()

        # Both submitted statements
        assert len(state.statements) == 2

        # Votes were recorded
        assert len(state.votes) > 0

        # Collective output exists
        assert state.collective_output is not None
        assert "n_groups" in state.collective_output

        # Contributions logged
        assert len(state.contributions) > 0
