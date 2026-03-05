# Adaptive Belief Ecology System (ABES)

<p align="center">
  <img src="docs/assets/hero-banner.svg" alt="ABES: Adaptive Belief Ecology System" width="100%" />
</p>

<p align="center">
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg" alt="License: AGPL v3" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+" /></a>
  <a href="tests/"><img src="https://img.shields.io/badge/tests-821%20passing-brightgreen.svg" alt="Tests" /></a>
  <a href="docs/EVALUATIONS.md"><img src="https://img.shields.io/badge/cognitive%20eval-825%2F1000-blue.svg" alt="Cognitive Eval" /></a>
</p>

ABES is a living memory ecology where beliefs reinforce, contradict, mutate, and decay. It runs as a headless engine for autonomous AI agents.

## Quick Navigation

- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [The Demo and CLI Reference](#the-demo-and-cli-reference)
- [Testing and Verification](#testing-and-verification)
- [Belief Model](#belief-model)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Quick Start

### Local setup

```bash
git clone https://github.com/Aftermath-Technologies-Ltd/adaptive-belief-ecology-system.git
cd adaptive-belief-ecology-system

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Start the engine

```bash
# Terminal 1
PYTHONPATH=$PWD uvicorn backend.api.app:app --host 0.0.0.0 --port 8000

# Terminal 2
curl -X POST http://localhost:8000/beliefs \
  -H "Content-Type: application/json" \
  -d '{"content": "System target is alpha-node-4", "confidence": 0.9, "source": "agent"}'

curl http://localhost:8000/beliefs | python3 -m json.tool
```

### Installation and Docker

Requirements: Python 3.10+, Node.js 18+ (visual debugger), Ollama (optional).

```bash
# Optional visual debugger
cd frontend
npm install
npm run dev
```

```bash
# Docker options
docker compose up
docker compose up --profile ui
docker compose up --profile llm --profile ui
```

For persistence across restarts:

```bash
STORAGE_BACKEND=sqlite docker compose up
```

---

## Key Features

| Feature | Source | Tests |
|---------|--------|-------|
| Belief data model (confidence, salience, tension, evidence, graph links) | [backend/core/models/belief.py](backend/core/models/belief.py) | [tests/core/test_belief_ecology_extensions.py](tests/core/test_belief_ecology_extensions.py) |
| 15-phase scheduler | [backend/agents/scheduler.py](backend/agents/scheduler.py) | [tests/agents/test_scheduler.py](tests/agents/test_scheduler.py) |
| Perception, reinforcement, decay | [backend/agents/](backend/agents/) | [tests/agents/](tests/agents/) |
| Contradiction auditing and tension scoring | [backend/agents/contradiction_auditor.py](backend/agents/contradiction_auditor.py) | [tests/agents/test_tension_formula.py](tests/agents/test_tension_formula.py) |
| Consolidation of near-duplicates | [backend/agents/consolidation.py](backend/agents/consolidation.py) | [tests/agents/test_consolidation.py](tests/agents/test_consolidation.py) |
| Belief stack selection | [backend/core/bel/stack.py](backend/core/bel/stack.py) | [tests/core/test_belief_stack.py](tests/core/test_belief_stack.py) |
| Semantic contradiction detection | [backend/core/bel/semantic_contradiction.py](backend/core/bel/semantic_contradiction.py) | [tests/core/test_semantic_contradiction.py](tests/core/test_semantic_contradiction.py) |
| FastAPI REST and WebSocket API | [backend/api/app.py](backend/api/app.py) | [tests/api/test_routes.py](tests/api/test_routes.py) |
| Ingestion pipeline service | [backend/chat/service.py](backend/chat/service.py) | [tests/chat/test_response_validator.py](tests/chat/test_response_validator.py) |
| 1000-prompt evaluation suite | [tests/cognitive/eval/](tests/cognitive/eval/) | [tests/cognitive/test_prompt_bank.py](tests/cognitive/test_prompt_bank.py) |

---

## Architecture

<p align="center">
  <img src="docs/assets/architecture.svg" alt="ABES System Architecture" width="100%" />
</p>

Core flow:
1. Agents submit payloads through REST or WebSocket endpoints.
2. The scheduler runs 15 phases over belief state.
3. Beliefs are updated in memory or SQLite storage.
4. The relevance stack is selected for response generation.
5. Safety and response validation run before output.

Ingestion pipeline stages:
- Perception
- Creation
- Reinforcement
- Decay
- Contradiction audit
- Mutation
- Relevance ranking
- LLM generation
- Response validation

API groups:
- `/auth`
- `/chat` (ingestion endpoints)
- `/beliefs`
- `/bel`
- `/agents`
- `/clusters`
- `/snapshots`

---

## The Demo and CLI Reference

### Demo

`abes demo` runs a 12-turn scripted ingestion sequence that triggers belief creation, reinforcement, contradiction, recall, and update behavior.

Script file: [examples/demo_conversation.json](examples/demo_conversation.json)

### CLI commands

All commands are available after `pip install -e ".[dev]"`:

| Command | Description |
|---------|-------------|
| `abes demo` | Run scripted ingestion demo |
| `abes chat` | Launch backend with visual debugger |
| `abes seed` | Load seed beliefs from JSON |
| `abes inspect` | Show current ecology state |
| `abes verify-quick` | Run cognitive smoke test |
| `abes verify-determinism` | Compare repeated runs for reproducibility |

Examples:

```bash
abes demo --headless
abes demo --headless --with-decay --decay-hours 12
abes inspect --json-out | jq .
abes verify-quick --prompts 200
abes verify-determinism --runs 5
```

---

## Testing and Verification

### Unit and integration tests

```bash
PYTHONPATH=$PWD pytest tests/ -q
```

Current status: **821 passed, 0 failed**.

### Verification experiments

```bash
PYTHONPATH=$PWD python experiments/run_all.py
```

Artifacts are in [results/](results/), including determinism, offline operation, conflict resolution, drift comparison, and decay sweep outputs.

### 1000-prompt evaluation summary

Detailed breakdowns are in [docs/EVALUATIONS.md](docs/EVALUATIONS.md).

| Metric | Result |
|--------|--------|
| Overall score | **825/1000 (82.5%)** |
| Episodic memory | **96.8%** |
| Working memory | **94.4%** |
| Semantic memory | **92.8%** |

Moral reasoning shortfalls stem from LLM refusals, not ecology mechanics.

---

## Belief Model

### Lifecycle

<p align="center">
  <img src="docs/assets/belief-lifecycle.svg" alt="Belief Lifecycle State Machine" width="100%" />
</p>

The lifecycle diagram shows how beliefs transition through active, decaying, dormant, mutated, and deprecated states.

Core fields:
- `id`, `content`, `confidence`, `tension`, `salience`
- `status`: `active`, `decaying`, `dormant`, `mutated`, `deprecated`
- `half_life_days`, `evidence_for`, `evidence_against`, `evidence_balance`
- `links`, `parent_id`, `user_id`, `session_id`, `origin`

Formulas used in ranking and state updates:
- Salience decay: `s(t) = s0 * 0.5^(elapsed_hours / (half_life_days * 24))`
- Confidence update: `posterior = 0.7 * evidence_weight + 0.3 * prior_confidence`
- Stack ranking: weighted score over confidence, relevance, salience, recency, and tension

Contradiction benchmark details are in [backend/core/bel/semantic_contradiction.py](backend/core/bel/semantic_contradiction.py) and [data/contradiction_corpus.json](data/contradiction_corpus.json).

Contradiction benchmark summary: the semantic detector outperforms the legacy detector on quantifiers (81.8% vs 54.5%) and numeric or unit cases (83.3% vs 66.7%), but is weaker on modality (45.5%) and temporal (36.4%) categories.

---

## Limitations

- Modality and temporal contradiction detection are weaker than quantifier and numeric rules.
- Moral reasoning remains the weakest evaluation area due to model behavior.
- In-memory storage is default; use SQLite for persistence.
- Scale behavior above roughly 10,000 beliefs is not yet fully characterized.
- Multi-agent concurrency has not been load-tested.

---

## Roadmap

### Completed

- CLI entrypoint with demo, inspect, seed, and verification commands
- Docker Compose profiles for backend, UI, and local LLM
- 1000-prompt evaluation suite
- Belief ecology extensions: salience, evidence ledger, graph edges, dormancy
- Consolidation agent and belief stack selection
- NLI fallback for contradiction detection
- Response validation and hybrid LLM routing
- 200-prompt stress test and 50-prompt cognitive battery

### Next Up

- Belief explorer UI for relationship graphs
- Document ingestion service for bulk context streams
- Benchmarks against production memory systems
- Better moral reasoning results through model and prompt strategy
- Load testing at larger belief counts
- Multi-agent concurrent ingestion testing

---

## License

GNU Affero General Public License v3.0 (AGPL-3.0)

Copyright (C) 2026 Bradley R. Kinnard. All Rights Reserved.

Attribution requirements are documented in [NOTICE](NOTICE).
Citation metadata is in [CITATION.cff](CITATION.cff).
