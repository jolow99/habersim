"""
Abstract base classes for Architecture and Agent.

These are the two primary extension points for researchers.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic

from habersim.core.types import (
    Action,
    AgentID,
    DeliberationState,
    LLMClient,
    Perception,
    StateT,
)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class Architecture(ABC, Generic[StateT]):
    """
    Defines a deliberation architecture. Researchers subclass this to
    implement new deliberation formats (e.g. Polis, Delphi, direct dialogue).

    The architecture owns:
    - what agents perceive
    - what actions are available to them
    - how actions update state
    - how state is aggregated into a collective output
    - when deliberation terminates
    """

    @abstractmethod
    def initial_state(self, topic: str, participants: list[AgentID]) -> StateT:
        """Create the initial deliberation state for a given topic and participant list."""
        ...

    @abstractmethod
    def perceive(self, agent_id: AgentID, state: StateT) -> Perception:
        """Construct what a specific agent sees at this moment in the deliberation."""
        ...

    @abstractmethod
    def update(self, state: StateT, action: Action) -> StateT:
        """
        Apply an agent's action to the state. Should:
        1. Validate the action is legal
        2. Mutate / return new state
        3. Log the action as a Contribution
        4. Recompute collective_output if needed
        """
        ...

    @abstractmethod
    def aggregate(self, state: StateT) -> object:
        """
        Compute the current collective output from state.
        Called after every update. Result stored in state.collective_output.
        """
        ...

    @abstractmethod
    def is_terminal(self, state: StateT) -> bool:
        """Return True if the deliberation should end."""
        ...

    def name(self) -> str:
        """Return the name of this architecture."""
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent(ABC):
    """
    An agent that can participate in a deliberation. Researchers subclass
    this to implement different agent behaviours — LLM-backed, rule-based,
    human-proxy, etc.

    Agents are architecture-agnostic: they only see a Perception and must
    return an Action.
    """

    def __init__(self, agent_id: AgentID, persona: str = ""):
        self.agent_id = agent_id
        self.persona = persona

    @abstractmethod
    def act(self, perception: Perception) -> Action:
        """Given a perception of the current deliberation state, return an action."""
        ...

    def name(self) -> str:
        """Return the agent's identifier."""
        return self.agent_id


# ---------------------------------------------------------------------------
# LLMAgent — base class for LLM-backed agents
# ---------------------------------------------------------------------------

class LLMAgent(Agent):
    """
    Base class for agents backed by an LLM.
    Handles prompt construction boilerplate; subclasses define the
    specific prompting strategy.
    """

    def __init__(self, agent_id: AgentID, llm: LLMClient, persona: str = ""):
        super().__init__(agent_id, persona)
        self.llm = llm

    def system_prompt(self) -> str:
        """Build the system prompt for this agent."""
        base = (
            "You are an AI agent participating in a structured deliberation. "
            "You will be shown the current state of a discussion and asked to take an action. "
            "Always reason carefully before acting."
        )
        if self.persona:
            base += f"\n\nYour persona: {self.persona}"
        return base

    @abstractmethod
    def act(self, perception: Perception) -> Action:
        ...
