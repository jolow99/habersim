"""
Habermolt architecture.

Agents submit opinions, propose consensus statements, and rank them.
The winning statement is determined by the Schulze method.
Deliberation is async — agents can act at any time.
When a new statement is added, rankings are predicted for all agents
who haven't explicitly ranked it yet.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from habersim.core import (
    Action,
    ActionSpec,
    AgentID,
    Architecture,
    Contribution,
    DeliberationState,
    Perception,
    StatementID,
)


# ---------------------------------------------------------------------------
# Habermolt-specific state
# ---------------------------------------------------------------------------

@dataclass
class ConsensusStatement:
    """A consensus statement proposed by an agent for others to rank."""

    id: StatementID
    text: str
    author: AgentID
    predicted_for: list[AgentID] = field(default_factory=list)  # agents with predicted rankings


@dataclass
class HabermoltState(DeliberationState):
    """State for the Habermolt architecture, extending the base deliberation state."""

    opinions: dict[AgentID, str] = field(default_factory=dict)
    statements: list[ConsensusStatement] = field(default_factory=list)
    # rankings[agent_id] = ordered list of statement IDs, most preferred first
    rankings: dict[AgentID, list[StatementID]] = field(default_factory=dict)
    # predicted_rankings[agent_id] = system-predicted ranking (before agent confirms)
    predicted_rankings: dict[AgentID, list[StatementID]] = field(default_factory=dict)

    def statement_by_id(self, sid: StatementID) -> ConsensusStatement | None:
        """Look up a statement by its ID."""
        return next((s for s in self.statements if s.id == sid), None)

    def all_statement_ids(self) -> list[StatementID]:
        """Return all statement IDs in order of creation."""
        return [s.id for s in self.statements]


# ---------------------------------------------------------------------------
# Schulze method implementation
# ---------------------------------------------------------------------------

def schulze_ranking(
    statements: list[StatementID],
    rankings: dict[AgentID, list[StatementID]],
) -> list[StatementID]:
    """
    Compute the Schulze (beatpath) ranking over statements given agent rankings.
    Returns statements ordered from most to least preferred collectively.
    """
    n = len(statements)
    if n == 0:
        return []

    idx = {s: i for i, s in enumerate(statements)}

    # Build pairwise preference matrix d[i][j] = number of voters who prefer i over j
    d = [[0] * n for _ in range(n)]
    for ranking in rankings.values():
        ranked = [s for s in ranking if s in idx]  # filter unknown statements
        for pos_i, s_i in enumerate(ranked):
            for s_j in ranked[pos_i + 1:]:
                d[idx[s_i]][idx[s_j]] += 1  # s_i preferred over s_j

    # Floyd-Warshall to find strongest paths p[i][j]
    p = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                p[i][j] = d[i][j] if d[i][j] > d[j][i] else 0

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i != j and i != k and j != k:
                    p[i][j] = max(p[i][j], min(p[i][k], p[k][j]))

    # Rank by number of statements each beats
    wins = [sum(1 for j in range(n) if i != j and p[i][j] > p[j][i]) for i in range(n)]
    return [s for s, _ in sorted(zip(statements, wins), key=lambda x: -x[1])]


# ---------------------------------------------------------------------------
# Habermolt Architecture
# ---------------------------------------------------------------------------

class HabermoltArchitecture(Architecture[HabermoltState]):
    """
    The core Habermolt deliberation format:

    - Agents write an initial opinion
    - Any agent can add a consensus statement
    - Agents rank statements; Schulze method determines collective preference
    - New statements trigger predicted rankings for all agents
    - Agents can correct predicted rankings at any time
    """

    # Actions available in this architecture
    ACTION_SUBMIT_OPINION = "submit_opinion"
    ACTION_ADD_STATEMENT = "add_statement"
    ACTION_UPDATE_RANKING = "update_ranking"
    ACTION_CONFIRM_PREDICTED_RANKING = "confirm_predicted_ranking"

    def initial_state(self, topic: str, participants: list[AgentID]) -> HabermoltState:
        """Create a fresh Habermolt deliberation state."""
        return HabermoltState(topic=topic, participants=list(participants))

    def perceive(self, agent_id: AgentID, state: HabermoltState) -> Perception:
        """Build the perception for a specific agent in the current state."""
        has_opinion = agent_id in state.opinions
        statements = state.all_statement_ids()
        current_ranking = state.rankings.get(agent_id, [])
        predicted_ranking = state.predicted_rankings.get(agent_id, [])

        # What can this agent do right now?
        available_actions: list[ActionSpec] = []

        if not has_opinion:
            available_actions.append(ActionSpec(
                name=self.ACTION_SUBMIT_OPINION,
                description="Write your initial opinion on the topic.",
                parameters={"opinion": {"type": "string"}},
                required=["opinion"],
            ))

        available_actions.append(ActionSpec(
            name=self.ACTION_ADD_STATEMENT,
            description="Propose a new consensus statement for others to rank.",
            parameters={"text": {"type": "string"}},
            required=["text"],
        ))

        if statements:
            available_actions.append(ActionSpec(
                name=self.ACTION_UPDATE_RANKING,
                description=(
                    "Submit or update your ranking of consensus statements, "
                    "ordered from most to least preferred. "
                    "Include all statement IDs."
                ),
                parameters={
                    "ranking": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Statement IDs ordered most→least preferred",
                    }
                },
                required=["ranking"],
            ))

            if predicted_ranking and predicted_ranking != current_ranking:
                available_actions.append(ActionSpec(
                    name=self.ACTION_CONFIRM_PREDICTED_RANKING,
                    description="Accept the system-predicted ranking as your own.",
                    parameters={},
                    required=[],
                ))

        context: dict[str, Any] = {
            "your_opinion": state.opinions.get(agent_id),
            "statements": [
                {"id": s.id, "text": s.text, "author": s.author}
                for s in state.statements
            ],
            "your_current_ranking": current_ranking,
            "predicted_ranking_for_you": predicted_ranking,
            "collective_output": state.collective_output,
            "other_agents": [p for p in state.participants if p != agent_id],
        }

        return Perception(
            agent_id=agent_id,
            topic=state.topic,
            context=context,
            available_actions=available_actions,
        )

    def update(self, state: HabermoltState, action: Action) -> HabermoltState:
        """Apply an agent action to the Habermolt state."""
        from uuid import uuid4

        agent_id = action.agent_id
        payload = action.payload

        if action.action_type == self.ACTION_SUBMIT_OPINION:
            opinion = payload.get("opinion", "")
            state.opinions[agent_id] = opinion
            state.log(Contribution(
                agent_id=agent_id,
                action_type=self.ACTION_SUBMIT_OPINION,
                payload={"opinion": opinion},
            ))

        elif action.action_type == self.ACTION_ADD_STATEMENT:
            text = payload.get("text", "").strip()
            if not text:
                raise ValueError("Statement text cannot be empty.")
            sid = str(uuid4())
            statement = ConsensusStatement(id=sid, text=text, author=agent_id)
            state.statements.append(statement)
            state.log(Contribution(
                agent_id=agent_id,
                action_type=self.ACTION_ADD_STATEMENT,
                payload={"statement_id": sid, "text": text},
            ))
            # Insert new statement at end of every existing agent's ranking
            self._insert_new_statement(state, sid)

        elif action.action_type == self.ACTION_UPDATE_RANKING:
            ranking = payload.get("ranking", [])
            self._validate_ranking(state, ranking)
            state.rankings[agent_id] = ranking
            # Clear predicted ranking once agent explicitly sets their own
            state.predicted_rankings.pop(agent_id, None)
            state.log(Contribution(
                agent_id=agent_id,
                action_type=self.ACTION_UPDATE_RANKING,
                payload={"ranking": ranking},
                predicted=False,
            ))

        elif action.action_type == self.ACTION_CONFIRM_PREDICTED_RANKING:
            predicted = state.predicted_rankings.get(agent_id)
            if predicted:
                state.rankings[agent_id] = predicted
                state.predicted_rankings.pop(agent_id)
                state.log(Contribution(
                    agent_id=agent_id,
                    action_type=self.ACTION_UPDATE_RANKING,
                    payload={"ranking": predicted},
                    predicted=True,
                ))

        else:
            raise ValueError(f"Unknown action type: {action.action_type!r}")

        # Recompute collective output after every update
        state.collective_output = self.aggregate(state)
        return state

    def aggregate(self, state: HabermoltState) -> dict | None:
        """Run Schulze on all confirmed rankings. Returns ranked statements."""
        sids = state.all_statement_ids()
        if not sids or not state.rankings:
            return None

        ordered = schulze_ranking(sids, state.rankings)
        statements_by_id = {s.id: s.text for s in state.statements}

        return {
            "winner_id": ordered[0] if ordered else None,
            "winner_text": statements_by_id.get(ordered[0]) if ordered else None,
            "full_ranking": [
                {"id": sid, "text": statements_by_id[sid]}
                for sid in ordered
            ],
            "participants_voted": len(state.rankings),
            "total_participants": len(state.participants),
        }

    def is_terminal(self, state: HabermoltState) -> bool:
        """Habermolt is async/continuous — never terminates automatically."""
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert_new_statement(self, state: HabermoltState, new_sid: StatementID) -> None:
        """
        When a new statement is added, insert it at the end of every agent's
        existing ranking (as a placeholder). Mark as predicted so agents know
        to review.
        """
        for agent_id in state.participants:
            if agent_id in state.rankings:
                current = state.rankings[agent_id]
                if new_sid not in current:
                    predicted = current + [new_sid]
                    state.predicted_rankings[agent_id] = predicted
            else:
                existing_predicted = state.predicted_rankings.get(agent_id, [])
                if new_sid not in existing_predicted:
                    state.predicted_rankings[agent_id] = existing_predicted + [new_sid]

    def _validate_ranking(self, state: HabermoltState, ranking: list[StatementID]) -> None:
        """Validate that a ranking only contains known statement IDs."""
        known = set(state.all_statement_ids())
        submitted = set(ranking)
        unknown = submitted - known
        if unknown:
            raise ValueError(f"Ranking contains unknown statement IDs: {unknown}")
