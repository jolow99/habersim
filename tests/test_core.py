"""Tests for core types and base state methods."""

import pytest
from habersim.core.types import Contribution, DeliberationState, ActionSpec, Action, Perception


class TestContribution:
    def test_creation_with_defaults(self):
        c = Contribution()
        assert c.id  # UUID generated
        assert c.agent_id == ""
        assert c.action_type == ""
        assert c.payload == {}
        assert c.predicted is False

    def test_creation_with_values(self):
        c = Contribution(
            agent_id="alice",
            action_type="vote",
            payload={"choice": "yes"},
            predicted=True,
        )
        assert c.agent_id == "alice"
        assert c.action_type == "vote"
        assert c.payload == {"choice": "yes"}
        assert c.predicted is True

    def test_unique_ids(self):
        c1 = Contribution()
        c2 = Contribution()
        assert c1.id != c2.id


class TestDeliberationState:
    def test_creation_with_defaults(self):
        state = DeliberationState()
        assert state.topic == ""
        assert state.participants == []
        assert state.contributions == []
        assert state.collective_output is None
        assert state.round == 0

    def test_log_appends_contribution(self):
        state = DeliberationState(topic="test")
        c = Contribution(agent_id="alice", action_type="speak")
        state.log(c)
        assert len(state.contributions) == 1
        assert state.contributions[0] is c

    def test_log_updates_timestamp(self):
        state = DeliberationState(topic="test")
        old_ts = state.updated_at
        c = Contribution(agent_id="alice", action_type="speak")
        state.log(c)
        assert state.updated_at >= old_ts

    def test_contributions_by_agent(self):
        state = DeliberationState()
        state.log(Contribution(agent_id="alice", action_type="a"))
        state.log(Contribution(agent_id="bob", action_type="b"))
        state.log(Contribution(agent_id="alice", action_type="c"))

        alice_contribs = state.contributions_by("alice")
        assert len(alice_contribs) == 2
        assert all(c.agent_id == "alice" for c in alice_contribs)

    def test_contributions_of_type(self):
        state = DeliberationState()
        state.log(Contribution(agent_id="alice", action_type="vote"))
        state.log(Contribution(agent_id="bob", action_type="speak"))
        state.log(Contribution(agent_id="carol", action_type="vote"))

        votes = state.contributions_of_type("vote")
        assert len(votes) == 2
        assert all(c.action_type == "vote" for c in votes)

    def test_empty_contributions_by(self):
        state = DeliberationState()
        assert state.contributions_by("nonexistent") == []

    def test_multiple_logs(self):
        state = DeliberationState()
        for i in range(10):
            state.log(Contribution(agent_id=f"agent_{i}", action_type="act"))
        assert len(state.contributions) == 10


class TestActionSpec:
    def test_creation(self):
        spec = ActionSpec(
            name="vote",
            description="Cast a vote",
            parameters={"choice": {"type": "string"}},
            required=["choice"],
        )
        assert spec.name == "vote"
        assert spec.required == ["choice"]


class TestAction:
    def test_creation(self):
        action = Action(agent_id="alice", action_type="vote", payload={"choice": "yes"})
        assert action.agent_id == "alice"
        assert action.reasoning == ""


class TestPerception:
    def test_creation(self):
        p = Perception(
            agent_id="alice",
            topic="test topic",
            context={"key": "value"},
            available_actions=[],
        )
        assert p.agent_id == "alice"
        assert p.topic == "test topic"
        assert p.instruction == ""
