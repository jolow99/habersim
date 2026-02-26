"""
habersim — a Python research simulation framework for multi-agent deliberation.

Provides abstractions for defining deliberation architectures, running simulations
with LLM-backed agents, and analyzing collective decision-making processes.
"""

__version__ = "0.1.0"

from habersim.core.types import (
    Action,
    ActionSpec,
    AgentID,
    Contribution,
    ContributionID,
    DeliberationID,
    DeliberationState,
    LLMClient,
    Perception,
    StateT,
    StatementID,
)
from habersim.core.base import Agent, Architecture, LLMAgent

__all__ = [
    "Action",
    "ActionSpec",
    "AgentID",
    "Agent",
    "Architecture",
    "Contribution",
    "ContributionID",
    "DeliberationID",
    "DeliberationState",
    "LLMAgent",
    "LLMClient",
    "Perception",
    "StateT",
    "StatementID",
]
