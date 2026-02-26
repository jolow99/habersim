# habersim

A Python research simulation framework for multi-agent deliberation — how groups of AI agents reach (or fail to reach) collective decisions on contested topics. Designed for researchers studying AI deliberation, collective intelligence, social choice theory, and democratic decision-making. Think of it like [Concordia](https://github.com/google-deepmind/concordia) but focused specifically on deliberation and collective preference rather than narrative simulation.

habersim is the simulation research layer built alongside the [Habermolt](https://habermolt.com) platform, where humans send AI agents to deliberate on their behalf.

## Core abstractions

A simulation consists of three parts:

- **Architecture** — defines what agents can do, what they see, and how their inputs are aggregated into a collective output (e.g. Schulze voting, clustering)
- **Agents** — LLM-backed participants that receive a perception of the current deliberation state and choose an action
- **Simulation runner** — orchestrates the deliberation loop, logs every action, and exports the full trajectory for analysis

### Architecture

The primary extension point. Implement this to define a new deliberation format.

```python
class MyArchitecture(Architecture[MyState]):
    def initial_state(self, topic, participants) -> MyState: ...
    def perceive(self, agent_id, state) -> Perception: ...
    def update(self, state, action) -> MyState: ...
    def aggregate(self, state) -> object: ...
    def is_terminal(self, state) -> bool: ...
```

### DeliberationState

A shared base class for all architecture states. Every state has:

- `topic` — what is being deliberated
- `participants` — list of agent IDs
- `contributions` — a timestamped event log of every action taken (architecture-agnostic)
- `collective_output` — the current result, architecture-defined
- `round`, `created_at`, `updated_at`, `is_terminal`

Architectures subclass this to add their own fields (e.g. rankings, statements, vote matrices).

### Agent

Agents are architecture-agnostic — they only see a `Perception` (the topic, a context dict, and a list of available actions) and return an `Action`.

```python
class MyAgent(Agent):
    def act(self, perception: Perception) -> Action: ...
```

### Contribution

The universal event log entry. Every action in every architecture is recorded as:

```python
@dataclass
class Contribution:
    agent_id: AgentID
    action_type: str
    payload: dict
    timestamp: datetime
    predicted: bool   # system-predicted vs agent-authored
```

This gives a common substrate for cross-architecture analysis and replay.

## Included architectures

### Habermolt

The core Habermolt deliberation format, used in production research:

- Agents write an initial opinion on the topic
- Any agent can propose a consensus statement
- Agents rank statements in order of preference
- The collective preference is computed using the **Schulze method** (a Condorcet-compliant voting system)
- When a new statement is added, the system predicts updated rankings for all agents — agents can confirm or correct these

Fully asynchronous — agents can act in any order at any time.

### Polis

Inspired by [pol.is](https://pol.is), a real-time survey tool for surfacing opinion groups:

- Agents are shown statements one at a time and vote **agree**, **disagree**, or **pass**
- Any agent can submit a new statement at any time
- The collective output is a set of **opinion groups** — clusters of agents with similar voting patterns
- Clustering uses k-means on vote vectors (agree=1, disagree=-1, pass=0) with cosine similarity

Fully asynchronous. No external dependencies beyond the standard library.

## Quickstart

```bash
uv sync
```

Run the examples (no API keys needed — they use mock LLM clients):

```bash
uv run python examples/run_habermolt.py
uv run python examples/run_polis.py
```

Run the tests:

```bash
uv sync --extra dev
uv run pytest
```

### Using with a real LLM

```bash
uv sync --extra anthropic  # or --extra openai, or --extra all
```

```python
from habersim.architectures.habermolt import HabermoltArchitecture
from habersim.agents import GenericLLMAgent, AnthropicClient
from habersim.simulation import Simulation, SimulationConfig

architecture = HabermoltArchitecture()
llm = AnthropicClient(model="claude-sonnet-4-20250514", api_key="...")

agents = [
    GenericLLMAgent("alice", llm, persona="AI safety researcher"),
    GenericLLMAgent("bob",   llm, persona="AI entrepreneur"),
    GenericLLMAgent("carol", llm, persona="Policy analyst"),
]

sim = Simulation(
    architecture=architecture,
    agents=agents,
    topic="How should society govern the development of advanced AI systems?",
    config=SimulationConfig(max_rounds=5),
)

final_state = sim.run()
sim.print_transcript()

# Export full trajectory for analysis
data = sim.to_dict()
```

See `examples/run_habermolt.py` and `examples/run_polis.py` for complete runnable examples.

## Implementing your own architecture

The point of the framework is to make it easy to drop in a new deliberation format and run it against the same agent population. Here's a minimal example:

```python
from dataclasses import dataclass, field
from habersim.core import (
    Architecture, DeliberationState, Perception,
    Action, ActionSpec, AgentID, Contribution,
)

@dataclass
class MyState(DeliberationState):
    votes: dict = field(default_factory=dict)

class SimpleVoteArchitecture(Architecture[MyState]):

    def initial_state(self, topic, participants):
        return MyState(topic=topic, participants=list(participants))

    def perceive(self, agent_id, state):
        return Perception(
            agent_id=agent_id,
            topic=state.topic,
            context={"current_votes": state.votes},
            available_actions=[
                ActionSpec(
                    name="vote",
                    description="Cast your vote: yes or no.",
                    parameters={"choice": {"type": "string", "enum": ["yes", "no"]}},
                    required=["choice"],
                )
            ],
        )

    def update(self, state, action):
        state.votes[action.agent_id] = action.payload["choice"]
        state.log(Contribution(
            agent_id=action.agent_id, action_type="vote", payload=action.payload,
        ))
        state.collective_output = self.aggregate(state)
        return state

    def aggregate(self, state):
        yes = sum(1 for v in state.votes.values() if v == "yes")
        return {"yes": yes, "no": len(state.votes) - yes}

    def is_terminal(self, state):
        return len(state.votes) == len(state.participants)
```

The same `Simulation` runner, agents, and logging infrastructure work out of the box.

## Comparison: habersim vs Concordia

| | **habersim** | **Concordia** |
|---|---|---|
| **Scope** | Deliberation and collective decision-making | General generative agent simulation |
| **Domain** | Social choice, democratic processes, collective intelligence | Narrative, physical, and social simulation |
| **Aggregation** | Schulze voting, clustering, extensible | No built-in aggregation |
| **Memory model** | Event log (Contributions) — architecture-agnostic | Per-agent associative memory |
| **Agent design** | Architecture-agnostic (perceive → act) | Environment-coupled components |
| **Output** | Collective decisions, opinion groups | Emergent narratives |
| **Extension point** | `Architecture` class | `Component` / `Game Master` classes |

## Project structure

```
habersim/
├── habersim/
│   ├── core/
│   │   ├── types.py          # Contribution, DeliberationState, Perception, ActionSpec, Action, LLMClient
│   │   └── base.py           # Architecture, Agent, LLMAgent (abstract base classes)
│   ├── architectures/
│   │   ├── habermolt.py      # HabermoltArchitecture, Schulze implementation
│   │   └── polis.py          # PolisArchitecture, k-means clustering
│   ├── agents.py             # GenericLLMAgent, AnthropicClient, OpenAIClient
│   └── simulation.py         # Simulation runner, SimulationConfig, TurnScheduler
├── examples/
│   ├── run_habermolt.py      # Habermolt example with mock LLM
│   └── run_polis.py          # Polis example with mock LLM
├── tests/
│   ├── test_core.py
│   ├── test_habermolt.py
│   └── test_polis.py
├── pyproject.toml
└── README.md
```

## Research applications

The framework is designed to support questions like:

- Do different LLMs reach different collective decisions on the same topic?
- Does the deliberation architecture affect the quality or legitimacy of outcomes?
- How stable are LLM agent preferences? Do rankings shift as new statements are introduced?
- Are predicted rankings a good proxy for actual agent preferences?
- Does agent persona affect consensus formation?
- Do opinion clusters in Polis-like settings reflect meaningful ideological divisions?

Every simulation exports a complete `to_dict()` trajectory — all contributions, all events, all intermediate collective outputs — suitable for downstream analysis.

## Roadmap
TBD
# habersim
