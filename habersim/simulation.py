"""
Simulation runner.

Orchestrates agents and architecture, handles turn scheduling,
and logs the full trajectory for research analysis.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Generic

from habersim.core import (
    Action,
    Agent,
    AgentID,
    Architecture,
    DeliberationState,
    StateT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Turn scheduling
# ---------------------------------------------------------------------------

class TurnScheduler:
    """
    Determines which agents act and in what order each round.
    Subclass to implement custom scheduling (random, priority-based, etc.)
    """

    def next_agents(self, state: DeliberationState, agents: list[Agent]) -> list[Agent]:
        """Return the agents who should act in the next round."""
        return agents  # default: all agents every round


class RandomTurnScheduler(TurnScheduler):
    """Each round, shuffle agent order."""

    def next_agents(self, state: DeliberationState, agents: list[Agent]) -> list[Agent]:
        """Return agents in a random order."""
        import random
        shuffled = list(agents)
        random.shuffle(shuffled)
        return shuffled


# ---------------------------------------------------------------------------
# Simulation event log (separate from deliberation contributions)
# ---------------------------------------------------------------------------

@dataclass
class SimulationEvent:
    """A simulation-level event (round start, agent error, terminal condition, etc.)"""

    event_type: str
    data: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""

    max_rounds: int = 10
    max_actions_per_agent_per_round: int = 1
    stop_on_terminal: bool = True
    verbose: bool = True


class Simulation(Generic[StateT]):
    """
    Runs a deliberation simulation: given an architecture, a set of agents,
    and a topic, drives the deliberation loop and records everything.

    Usage::

        sim = Simulation(architecture, agents, topic, config)
        final_state = sim.run()
        sim.print_transcript()
    """

    def __init__(
        self,
        architecture: Architecture[StateT],
        agents: list[Agent],
        topic: str,
        config: SimulationConfig | None = None,
        scheduler: TurnScheduler | None = None,
        on_action: Callable[[Action, StateT], None] | None = None,
    ):
        self.architecture = architecture
        self.agents = agents
        self.topic = topic
        self.config = config or SimulationConfig()
        self.scheduler = scheduler or TurnScheduler()
        self.on_action = on_action  # optional callback after each action

        self._agent_map: dict[AgentID, Agent] = {a.agent_id: a for a in agents}
        self.state: StateT | None = None
        self.events: list[SimulationEvent] = []

    def run(self) -> StateT:
        """Execute the simulation and return the final state."""
        participant_ids = [a.agent_id for a in self.agents]
        self.state = self.architecture.initial_state(self.topic, participant_ids)

        self._log_event("simulation_start", {
            "topic": self.topic,
            "architecture": self.architecture.name(),
            "participants": participant_ids,
            "config": self.config.__dict__,
        })

        for round_num in range(1, self.config.max_rounds + 1):
            self.state.round = round_num
            self._log_event("round_start", {"round": round_num})

            if self.config.verbose:
                print(f"\n{'='*60}")
                print(f"  Round {round_num}")
                print(f"{'='*60}")

            acting_agents = self.scheduler.next_agents(self.state, self.agents)

            for agent in acting_agents:
                for _ in range(self.config.max_actions_per_agent_per_round):
                    try:
                        perception = self.architecture.perceive(agent.agent_id, self.state)

                        if not perception.available_actions:
                            break  # nothing to do

                        action = agent.act(perception)

                        if self.config.verbose:
                            self._print_action(agent.agent_id, action)

                        self.state = self.architecture.update(self.state, action)

                        self._log_event("action", {
                            "round": round_num,
                            "agent_id": agent.agent_id,
                            "action_type": action.action_type,
                            "payload": action.payload,
                            "reasoning": action.reasoning,
                        })

                        if self.on_action:
                            self.on_action(action, self.state)

                    except Exception as e:
                        logger.warning(f"Agent {agent.agent_id} error: {e}")
                        self._log_event("agent_error", {
                            "agent_id": agent.agent_id,
                            "error": str(e),
                        })

            if self.config.stop_on_terminal and self.architecture.is_terminal(self.state):
                self._log_event("terminal", {"round": round_num})
                if self.config.verbose:
                    print(f"\n[Terminal condition reached at round {round_num}]")
                break

        self._log_event("simulation_end", {
            "rounds_completed": self.state.round,
            "collective_output": self.state.collective_output,
        })

        if self.config.verbose:
            self.print_result()

        return self.state

    def print_result(self) -> None:
        """Print the collective output."""
        print(f"\n{'='*60}")
        print("  COLLECTIVE OUTPUT")
        print(f"{'='*60}")
        print(json.dumps(self.state.collective_output, indent=2, default=str))

    def print_transcript(self) -> None:
        """Print a human-readable transcript of the full deliberation."""
        print(f"\n{'='*60}")
        print(f"  TRANSCRIPT: {self.topic}")
        print(f"{'='*60}")
        for c in self.state.contributions:
            predicted_tag = " [predicted]" if c.predicted else ""
            print(f"\n[{c.timestamp.strftime('%H:%M:%S')}] {c.agent_id} → {c.action_type}{predicted_tag}")
            for k, v in c.payload.items():
                print(f"  {k}: {v}")

    def to_dict(self) -> dict:
        """Export full simulation data for research analysis."""
        return {
            "topic": self.topic,
            "architecture": self.architecture.name(),
            "participants": [a.agent_id for a in self.agents],
            "rounds_completed": self.state.round if self.state else 0,
            "contributions": [
                {
                    "id": c.id,
                    "agent_id": c.agent_id,
                    "action_type": c.action_type,
                    "payload": c.payload,
                    "timestamp": c.timestamp.isoformat(),
                    "predicted": c.predicted,
                }
                for c in (self.state.contributions if self.state else [])
            ],
            "collective_output": self.state.collective_output if self.state else None,
            "events": [
                {"type": e.event_type, "data": e.data, "timestamp": e.timestamp.isoformat()}
                for e in self.events
            ],
        }

    def _log_event(self, event_type: str, data: dict) -> None:
        self.events.append(SimulationEvent(event_type=event_type, data=data))

    def _print_action(self, agent_id: AgentID, action: Action) -> None:
        print(f"\n  [{agent_id}] {action.action_type}")
        if action.reasoning:
            print(f"    reasoning: {action.reasoning[:120]}...")
        for k, v in action.payload.items():
            val_str = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
            print(f"    {k}: {val_str}")
