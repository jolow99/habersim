"""
Polis-like architecture.

Inspired by pol.is — agents vote agree/disagree/pass on statements,
any agent can submit new statements, and the collective output is a set
of opinion groups derived from clustering agents by voting similarity.

Clustering uses k-means on vote vectors (agree=1, disagree=-1, pass=0)
with cosine similarity, implemented using only numpy (no sklearn).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

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
# Vote type
# ---------------------------------------------------------------------------

AGREE = 1
DISAGREE = -1
PASS = 0

VOTE_MAP = {"agree": AGREE, "disagree": DISAGREE, "pass": PASS}
VOTE_NAMES = {AGREE: "agree", DISAGREE: "disagree", PASS: "pass"}


# ---------------------------------------------------------------------------
# Polis-specific state
# ---------------------------------------------------------------------------

@dataclass
class PolisStatement:
    """A statement that agents can vote on."""

    id: StatementID
    text: str
    author: AgentID


@dataclass
class OpinionGroup:
    """A cluster of agents with similar voting patterns."""

    group_id: int
    agent_ids: list[AgentID]
    representative_votes: dict[StatementID, int]  # centroid votes (rounded)


@dataclass
class PolisState(DeliberationState):
    """
    State for the Polis architecture.

    Extends DeliberationState with statements, a vote matrix, and
    opinion group clusters.
    """

    statements: list[PolisStatement] = field(default_factory=list)
    # votes[agent_id][statement_id] = AGREE | DISAGREE | PASS
    votes: dict[AgentID, dict[StatementID, int]] = field(default_factory=dict)
    # Current clustering output
    groups: list[OpinionGroup] = field(default_factory=list)

    def statement_by_id(self, sid: StatementID) -> PolisStatement | None:
        """Look up a statement by ID."""
        return next((s for s in self.statements if s.id == sid), None)

    def all_statement_ids(self) -> list[StatementID]:
        """Return all statement IDs in order of creation."""
        return [s.id for s in self.statements]

    def unvoted_statements(self, agent_id: AgentID) -> list[PolisStatement]:
        """Return statements this agent hasn't voted on yet."""
        voted = self.votes.get(agent_id, {})
        return [s for s in self.statements if s.id not in voted]

    def vote_vector(self, agent_id: AgentID) -> list[int]:
        """
        Build a vote vector for an agent across all statements.
        Returns a list of ints in statement-creation order.
        Unvoted statements are treated as PASS (0).
        """
        agent_votes = self.votes.get(agent_id, {})
        return [agent_votes.get(s.id, PASS) for s in self.statements]


# ---------------------------------------------------------------------------
# Clustering (k-means with cosine similarity, numpy only)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cluster_agents(
    state: PolisState,
    n_clusters: int = 2,
    max_iterations: int = 50,
) -> list[OpinionGroup]:
    """
    Cluster agents into opinion groups using k-means on vote vectors.

    Uses cosine distance (1 - cosine_similarity) as the distance metric.
    Falls back gracefully when there are fewer agents than clusters.
    Implemented with pure Python (no numpy/sklearn dependency).

    Args:
        state: The current Polis deliberation state.
        n_clusters: Number of opinion groups to form.
        max_iterations: Maximum k-means iterations.

    Returns:
        A list of OpinionGroup objects.
    """
    agents = state.participants
    if not agents or not state.statements:
        return []

    # Build vote vectors
    vectors: dict[AgentID, list[float]] = {}
    for agent_id in agents:
        vectors[agent_id] = [float(x) for x in state.vote_vector(agent_id)]

    n_clusters = min(n_clusters, len(agents))
    dim = len(state.statements)

    # Initialize centroids using a greedy farthest-first approach
    # to avoid picking agents with identical votes
    centroids: list[list[float]] = [list(vectors[agents[0]])]
    for _ in range(1, n_clusters):
        # Pick the agent whose minimum similarity to existing centroids is lowest
        best_agent = None
        best_min_sim = 2.0  # higher than any real similarity
        for agent_id in agents:
            vec = vectors[agent_id]
            min_sim = min(_cosine_similarity(vec, c) for c in centroids)
            if min_sim < best_min_sim:
                best_min_sim = min_sim
                best_agent = agent_id
        centroids.append(list(vectors[best_agent]))

    assignments: dict[AgentID, int] = {}

    for _ in range(max_iterations):
        # Assign each agent to nearest centroid
        new_assignments: dict[AgentID, int] = {}
        for agent_id in agents:
            vec = vectors[agent_id]
            best_cluster = 0
            best_sim = -2.0
            for c_idx, centroid in enumerate(centroids):
                sim = _cosine_similarity(vec, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = c_idx
            new_assignments[agent_id] = best_cluster

        if new_assignments == assignments:
            break  # converged
        assignments = new_assignments

        # Recompute centroids
        for c_idx in range(n_clusters):
            members = [aid for aid, c in assignments.items() if c == c_idx]
            if not members:
                continue
            new_centroid = [0.0] * dim
            for aid in members:
                for d in range(dim):
                    new_centroid[d] += vectors[aid][d]
            for d in range(dim):
                new_centroid[d] /= len(members)
            centroids[c_idx] = new_centroid

    # Build OpinionGroup objects
    sids = state.all_statement_ids()
    groups: list[OpinionGroup] = []
    for c_idx in range(n_clusters):
        members = [aid for aid, c in assignments.items() if c == c_idx]
        if not members:
            continue
        # Round centroid to nearest vote value for representative votes
        rep_votes: dict[StatementID, int] = {}
        for d, sid in enumerate(sids):
            val = centroids[c_idx][d]
            if val > 0.33:
                rep_votes[sid] = AGREE
            elif val < -0.33:
                rep_votes[sid] = DISAGREE
            else:
                rep_votes[sid] = PASS
        groups.append(OpinionGroup(
            group_id=c_idx,
            agent_ids=members,
            representative_votes=rep_votes,
        ))

    return groups


# ---------------------------------------------------------------------------
# Polis Architecture
# ---------------------------------------------------------------------------

class PolisArchitecture(Architecture[PolisState]):
    """
    A Polis-like deliberation architecture.

    - Agents are shown statements one at a time and vote agree/disagree/pass
    - Any agent can submit a new statement at any time
    - The collective output is a set of opinion groups (clusters of agents
      with similar voting patterns)

    This architecture is async — ``is_terminal`` always returns False.
    """

    ACTION_VOTE = "vote"
    ACTION_SUBMIT_STATEMENT = "submit_statement"

    def __init__(self, n_clusters: int = 2):
        """
        Args:
            n_clusters: Number of opinion groups to cluster agents into.
        """
        self.n_clusters = n_clusters

    def initial_state(self, topic: str, participants: list[AgentID]) -> PolisState:
        """Create a fresh Polis deliberation state."""
        return PolisState(topic=topic, participants=list(participants))

    def perceive(self, agent_id: AgentID, state: PolisState) -> Perception:
        """
        Build a perception for the agent. Shows:
        - Statements they haven't voted on yet
        - Their current vote history
        - The current opinion group structure
        """
        unvoted = state.unvoted_statements(agent_id)
        agent_votes = state.votes.get(agent_id, {})

        available_actions: list[ActionSpec] = []

        # Always allow submitting a new statement
        available_actions.append(ActionSpec(
            name=self.ACTION_SUBMIT_STATEMENT,
            description="Submit a new statement for others to vote on.",
            parameters={"text": {"type": "string"}},
            required=["text"],
        ))

        # If there are unvoted statements, offer voting on the first one
        if unvoted:
            next_statement = unvoted[0]
            available_actions.append(ActionSpec(
                name=self.ACTION_VOTE,
                description=f'Vote on statement: "{next_statement.text}"',
                parameters={
                    "statement_id": {"type": "string", "const": next_statement.id},
                    "vote": {"type": "string", "enum": ["agree", "disagree", "pass"]},
                },
                required=["statement_id", "vote"],
            ))

        # Build context
        vote_history = {
            sid: VOTE_NAMES[v] for sid, v in agent_votes.items()
        }
        group_summary = [
            {
                "group_id": g.group_id,
                "members": g.agent_ids,
                "representative_votes": {
                    sid: VOTE_NAMES[v] for sid, v in g.representative_votes.items()
                },
            }
            for g in state.groups
        ]

        context: dict[str, Any] = {
            "all_statements": [
                {"id": s.id, "text": s.text, "author": s.author}
                for s in state.statements
            ],
            "unvoted_statements": [
                {"id": s.id, "text": s.text}
                for s in unvoted
            ],
            "your_votes": vote_history,
            "opinion_groups": group_summary,
        }

        return Perception(
            agent_id=agent_id,
            topic=state.topic,
            context=context,
            available_actions=available_actions,
        )

    def update(self, state: PolisState, action: Action) -> PolisState:
        """Apply an agent action to the Polis state."""
        agent_id = action.agent_id
        payload = action.payload

        if action.action_type == self.ACTION_SUBMIT_STATEMENT:
            text = payload.get("text", "").strip()
            if not text:
                raise ValueError("Statement text cannot be empty.")
            sid = str(uuid4())
            statement = PolisStatement(id=sid, text=text, author=agent_id)
            state.statements.append(statement)
            state.log(Contribution(
                agent_id=agent_id,
                action_type=self.ACTION_SUBMIT_STATEMENT,
                payload={"statement_id": sid, "text": text},
            ))

        elif action.action_type == self.ACTION_VOTE:
            sid = payload.get("statement_id", "")
            vote_str = payload.get("vote", "").lower()

            # Validate statement exists
            if not state.statement_by_id(sid):
                raise ValueError(f"Unknown statement ID: {sid}")

            # Validate vote value
            if vote_str not in VOTE_MAP:
                raise ValueError(f"Invalid vote: {vote_str!r}. Must be agree, disagree, or pass.")

            vote_val = VOTE_MAP[vote_str]
            if agent_id not in state.votes:
                state.votes[agent_id] = {}
            state.votes[agent_id][sid] = vote_val
            state.log(Contribution(
                agent_id=agent_id,
                action_type=self.ACTION_VOTE,
                payload={"statement_id": sid, "vote": vote_str},
            ))

        else:
            raise ValueError(f"Unknown action type: {action.action_type!r}")

        # Recompute clusters and collective output
        state.groups = cluster_agents(state, n_clusters=self.n_clusters)
        state.collective_output = self.aggregate(state)
        return state

    def aggregate(self, state: PolisState) -> dict | None:
        """
        Compute the collective output: opinion groups with voting patterns.
        """
        if not state.groups:
            return None

        return {
            "n_groups": len(state.groups),
            "groups": [
                {
                    "group_id": g.group_id,
                    "members": g.agent_ids,
                    "size": len(g.agent_ids),
                    "representative_votes": {
                        sid: VOTE_NAMES[v]
                        for sid, v in g.representative_votes.items()
                    },
                }
                for g in state.groups
            ],
            "total_statements": len(state.statements),
            "total_votes": sum(len(v) for v in state.votes.values()),
        }

    def is_terminal(self, state: PolisState) -> bool:
        """Polis is async — never terminates automatically."""
        return False
