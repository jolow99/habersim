"""
Core types for the habersim deliberation research framework.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeVar
from uuid import uuid4


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

AgentID = str
StatementID = str
ContributionID = str
DeliberationID = str


# ---------------------------------------------------------------------------
# Contributions — the universal event log
#
# Every action any agent takes in any architecture is recorded as a
# Contribution. This gives a architecture-agnostic substrate for analysis,
# replay, and cross-architecture comparison.
# ---------------------------------------------------------------------------

@dataclass
class Contribution:
    """A single logged action in a deliberation, forming the universal event log."""

    id: ContributionID = field(default_factory=lambda: str(uuid4()))
    agent_id: AgentID = ""
    action_type: str = ""          # e.g. "submit_opinion", "add_statement", "update_ranking"
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    predicted: bool = False        # was this contribution system-predicted vs agent-authored?


# ---------------------------------------------------------------------------
# DeliberationState — shared base, subclassed per architecture
# ---------------------------------------------------------------------------

@dataclass
class DeliberationState:
    """
    Architecture-agnostic base state. Every architecture subclasses this
    and adds its own fields. The base fields are enough for the simulation
    runner and cross-architecture analysis tools to operate without knowing
    which architecture is in use.
    """

    id: DeliberationID = field(default_factory=lambda: str(uuid4()))
    topic: str = ""
    participants: list[AgentID] = field(default_factory=list)
    contributions: list[Contribution] = field(default_factory=list)  # full event log
    collective_output: Any = None   # architecture-defined; winning statement, clusters, etc.
    round: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_terminal: bool = False

    def log(self, contribution: Contribution) -> None:
        """Append a contribution to the event log and update timestamp."""
        self.contributions.append(contribution)
        self.updated_at = datetime.utcnow()

    def contributions_by(self, agent_id: AgentID) -> list[Contribution]:
        """Return all contributions by a specific agent."""
        return [c for c in self.contributions if c.agent_id == agent_id]

    def contributions_of_type(self, action_type: str) -> list[Contribution]:
        """Return all contributions of a specific action type."""
        return [c for c in self.contributions if c.action_type == action_type]


# ---------------------------------------------------------------------------
# Perception — what an agent sees at a given moment
# ---------------------------------------------------------------------------

@dataclass
class Perception:
    """
    What the architecture surfaces to an agent before it acts.
    The ``context`` dict is architecture-specific — the architecture decides
    what to include. ``available_actions`` tells the agent what it can do.
    """

    agent_id: AgentID
    topic: str
    context: dict[str, Any]              # architecture-specific view of state
    available_actions: list[ActionSpec]  # what this agent can do right now
    instruction: str = ""                # optional natural language prompt hint


@dataclass
class ActionSpec:
    """
    Describes a single action an agent can take.
    Modelled loosely after tool-use schemas so LLM agents can parse them easily.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON-schema-like
    required: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Action — what an agent returns
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """An action chosen by an agent in response to a perception."""

    agent_id: AgentID
    action_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""   # optional chain-of-thought, useful for research logging


# ---------------------------------------------------------------------------
# LLM abstraction
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """
    Minimal interface for an LLM. Any provider can implement this.
    Deliberately thin — just text in, text out.
    """

    @abstractmethod
    def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        """Return a completion given a prompt."""
        ...

    @abstractmethod
    def complete_json(self, prompt: str, schema: dict, system: str = "", **kwargs) -> dict:
        """Return a structured JSON completion conforming to schema."""
        ...


# ---------------------------------------------------------------------------
# StateT — generic type variable for architecture-specific state subclasses
# ---------------------------------------------------------------------------

StateT = TypeVar("StateT", bound=DeliberationState)
