# purpose-driven-agent

[![PyPI version](https://img.shields.io/pypi/v/purpose-driven-agent.svg)](https://pypi.org/project/purpose-driven-agent/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/ASISaga/purpose-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ASISaga/purpose-agent/actions/workflows/ci.yml)

**The fundamental building block of the Agent Operating System (AOS).**

`purpose-driven-agent` provides `PurposeDrivenAgent` — an abstract base class
for **perpetual, purpose-driven AI agents** that run indefinitely, maintain
rich state across every interaction, and work toward a long-term assigned
purpose rather than short-term tasks.

---

## Table of Contents

1. [What is a Purpose-Driven Agent?](#what-is-a-purpose-driven-agent)
2. [AOS Container Hierarchy](#aos-container-hierarchy)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Architecture Overview](#architecture-overview)
6. [Inheritance Hierarchy](#inheritance-hierarchy)
7. [Usage Examples](#usage-examples)
   - [GenericPurposeDrivenAgent](#genericpurposedrivenagent)
   - [Custom Subclass](#custom-subclass)
   - [Event Handling](#event-handling)
   - [Goal Tracking](#goal-tracking)
   - [Purpose Alignment](#purpose-alignment)
   - [ML Pipeline (IMLService)](#ml-pipeline-imlservice)
8. [FAS Hosting](#fas-hosting)
9. [Routing Tag Enforcement](#routing-tag-enforcement)
10. [RoutingMixin](#routingmixin)
11. [LoRA Adapter Pattern](#lora-adapter-pattern)
12. [MCP Context Preservation](#mcp-context-preservation)
13. [Perpetual Operation Pattern](#perpetual-operation-pattern)
14. [Configuration](#configuration)
15. [Testing](#testing)
16. [API Reference](#api-reference)
17. [Contributing](#contributing)
18. [Related Packages](#related-packages)
19. [License](#license)

---

## What is a Purpose-Driven Agent?

Traditional AI agents are **task-based**: they start, execute a task, and
terminate.  A `PurposeDrivenAgent` is different in every fundamental way:

| Dimension | Task-Based Agent | Purpose-Driven Agent |
|---|---|---|
| Lifetime | One task → terminates | Registers once → runs indefinitely |
| State | Ephemeral (per task) | Persistent (across all events) |
| Goal | Complete the task | Work toward a long-term purpose |
| Sleep | N/A | Sleeps when idle; awakens on events |
| Adapter | N/A | Mapped to a LoRA adapter for domain expertise |
| Context | Rebuilt each time | Preserved by `ContextMCPServer` (MCP) |

This is the core architectural difference that makes AOS an **operating system
for AI agents**, not just an orchestration framework.

---

## AOS Container Hierarchy

`purpose-driven-agent` is **Layer 2** in the AOS container stack.  Each layer
inherits from the one above it — no layer re-implements what a parent already
provides.

```
infrastructure             (Layer 1) — Python 3.12, MAF 1.3.0, FAS hosting adapter
  └── purpose-driven-agent (Layer 2) — THIS REPO — PurposeDrivenAgent + aos_mcp_servers
        └── leadership-agent    (Layer 3)
              └── business-agent     (Layer 4)
                    └── founder-agent      (Layer 5, FAS target)
```

- **Layer 1** (`infrastructure`) installs Python 3.12, the Microsoft Agent
  Framework (MAF), and `agent-framework-foundry-hosting` which runs the FAS
  HTTP server.
- **Layer 2** (this repo) adds `PurposeDrivenAgent`, `aos_mcp_servers`, and
  the FAS hosting adapter (`hosting.py`).  Any descendant image is immediately
  hostable by Azure AI Foundry Agent Service.
- **Layers 3–5** add domain logic without touching hosting plumbing.

---

## Installation

```bash
# Core installation
pip install purpose-driven-agent

# Development
pip install "purpose-driven-agent[dev]"
```

**Requirements:** Python 3.12 or higher.

### Core Dependencies

| Package | Purpose |
|---|---|
| `agent-framework` | Core Microsoft Agent Framework for agent logic |
| `agent-framework-foundry` | Adapter converting agents into Foundry Agent Service-compatible hosted services |
| `azure-ai-agents` | Azure AI Agents Service client — engine hosting agents in the cloud |
| `azure-identity` | Secure, keyless authentication via Entra Agent ID |
| `pydantic` | Configuration and data validation |

> **Note:** `purpose-driven-agent` is a **code-only library** — it is not
> deployed to Azure as its own service.  It is consumed by agent repos
> (e.g. `leadership-agent`, `founder-agent`) that *are* deployed.

---

## Quick Start

```python
import asyncio
from purpose_driven_agent import GenericPurposeDrivenAgent

async def main():
    agent = GenericPurposeDrivenAgent(
        agent_id="assistant",
        purpose="Assist users with information retrieval and task coordination",
        adapter_name="general",
    )

    await agent.initialize()
    await agent.start()

    # Process an event
    result = await agent.handle_event({
        "type": "user_query",
        "data": {"query": "What is AOS?"},
    })
    print(result["status"])   # "success"

    # Check status
    status = await agent.get_purpose_status()
    print(status["total_events_processed"])  # 1

    await agent.stop()

asyncio.run(main())
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PurposeDrivenAgent                            │
│                     (abstract base class)                            │
│                                                                      │
│  ┌────────────────────────┐   ┌──────────────────────────────────┐   │
│  │    Purpose Context     │   │         LoRA Adapter             │   │
│  │  ──────────────────    │   │  ──────────────────────────      │   │
│  │  purpose               │   │  adapter_name → domain model     │   │
│  │  purpose_scope         │   │  "general", "leadership",        │   │
│  │  purpose_metrics       │   │  "marketing", "finance", ...     │   │
│  └────────────────────────┘   └──────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                   ContextMCPServer (MCP)                     │    │
│  │  Persistent context · Event history · Semantic memory       │    │
│  │  One dedicated instance per agent                            │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌────────────────────────┐   ┌──────────────────────────────────┐   │
│  │    Perpetual Loop      │   │          IMLService              │   │
│  │  asyncio.create_task   │   │  trigger_lora_training()         │   │
│  │  Sleep / Awaken        │   │  run_pipeline()                  │   │
│  │  Event subscriptions   │   │  infer()                         │   │
│  └────────────────────────┘   └──────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

Three foundational components work together:

1. **LoRA Adapters** — provide domain-specific vocabulary, knowledge, and
   agent persona via the `adapter_name` parameter.
2. **Core Purpose** — stored in the primary LLM context to guide all decisions.
3. **ContextMCPServer** — persists state across every event and system restart.

---

## Inheritance Hierarchy

```
agent_framework.Agent          ← Microsoft Agent Framework (optional)
        │
        ▼
PurposeDrivenAgent             ← abstract base (this package)
        │
        ├── GenericPurposeDrivenAgent   ← concrete, general-purpose
        │
        ├── LeadershipAgent            ← pip install leadership-agent (Layer 3)
        │       ├── BusinessAgent      ← pip install business-agent   (Layer 4)
        │       │     └── FounderAgent ← pip install founder-agent    (Layer 5, FAS target)
        │       ├── CFOAgent
        │       ├── CTOAgent
        │       ├── CSOAgent
        │       └── CMOAgent
        │
        └── <YourCustomAgent>          ← extend for your domain
```

`PurposeDrivenAgent` inherits from `agent_framework.Agent` (Microsoft Agent
Framework) when the package is installed.  A transparent stub is used when the
package is absent, so the library works without it.

---

## Usage Examples

### GenericPurposeDrivenAgent

The simplest concrete implementation:

```python
from purpose_driven_agent import GenericPurposeDrivenAgent

agent = GenericPurposeDrivenAgent(
    agent_id="cfo-assistant",
    purpose="Support the CFO with financial analysis and reporting",
    purpose_scope="Financial data, budgeting, forecasting",
    adapter_name="finance",
)

await agent.initialize()
await agent.start()
```

### Custom Subclass

Extend `PurposeDrivenAgent` for domain-specific agents:

```python
from typing import List
from purpose_driven_agent import PurposeDrivenAgent

class LegalAgent(PurposeDrivenAgent):
    """Specialist legal advisory agent."""

    def get_agent_type(self) -> List[str]:
        available = self.get_available_personas()
        return ["legal"] if "legal" in available else ["legal"]

    async def review_contract(self, contract: dict) -> dict:
        alignment = await self.evaluate_purpose_alignment(
            {"type": "contract_review"}
        )
        # Domain-specific logic here …
        return {
            "status": "reviewed",
            "risk_score": 0.2,
            "alignment": alignment["alignment_score"],
        }

agent = LegalAgent(
    agent_id="legal-001",
    purpose="Provide legal analysis and contract review",
    adapter_name="legal",
)
await agent.initialize()
result = await agent.review_contract({"title": "NDA", "pages": 12})
```

### Event Handling

```python
# Subscribe before starting
async def on_budget_request(data: dict) -> dict:
    approved = data.get("amount", 0) < 50_000
    return {"approved": approved, "reason": "Within policy limits"}

await agent.subscribe_to_event("budget_request", on_budget_request)

# Trigger from anywhere
result = await agent.handle_event({
    "type": "budget_request",
    "data": {"amount": 25_000, "department": "Engineering"},
})
print(result["handler_results"][0]["approved"])  # True
```

### Goal Tracking

```python
# Add a goal
goal_id = await agent.add_goal(
    "Complete Q2 financial close",
    success_criteria=["All transactions reconciled", "Report filed"],
    deadline="2025-06-30T23:59:59",
)

# Update progress
await agent.update_goal_progress(goal_id, 0.5, notes="Reconciliation 50% done")
await agent.update_goal_progress(goal_id, 1.0, notes="All done")

# Check metrics
status = await agent.get_purpose_status()
print(status["metrics"]["goals_achieved"])  # 1
```

### Purpose Alignment

```python
# Evaluate any action before executing it
alignment = await agent.evaluate_purpose_alignment({
    "type": "approve_vendor",
    "vendor": "Acme Corp",
    "value": 120_000,
})

if alignment["aligned"]:
    print(f"Aligned (score={alignment['alignment_score']:.2f})")
    # proceed

# Make a purpose-driven decision
decision = await agent.make_purpose_driven_decision({
    "options": [
        {"type": "cut_costs", "description": "Reduce vendor spend by 15%"},
        {"type": "invest",    "description": "Invest in new tooling"},
        {"type": "hold",      "description": "Maintain current budget"},
    ]
})
print(decision["selected_option"])
```

### ML Pipeline (IMLService)

Plug in your own ML backend:

```python
from purpose_driven_agent.ml_interface import IMLService
from typing import Any, Dict, List

class MyAzureMLService(IMLService):
    async def trigger_lora_training(
        self, training_params: Dict[str, Any], adapters: List[Dict[str, Any]]
    ) -> str:
        # Call Azure ML here
        return "run-001"

    async def run_pipeline(
        self, subscription_id: str, resource_group: str, workspace_name: str
    ) -> str:
        return "pipeline-001"

    async def infer(self, agent_id: str, prompt: str) -> Dict[str, Any]:
        return {"text": f"Response for {agent_id}"}

agent = GenericPurposeDrivenAgent(
    agent_id="trained-agent",
    purpose="Fine-tuned domain reasoning",
    adapter_name="finance",
    ml_service=MyAzureMLService(),
)
await agent.act("aml_infer", {"prompt": "Summarise Q2 expenses"})
```

---

## FAS Hosting

Any `PurposeDrivenAgent` subclass can be hosted by Azure AI Foundry Agent
Service (FAS) without additional plumbing.  Run the container with:

```bash
python -m purpose_driven_agent
```

This invokes `purpose_driven_agent/hosting.py → run_server()`, which:

1. Seeds the `__init_subclass__` registry by importing all packages in `/app`.
2. Discovers the concrete agent class using a three-strategy cascade:
   - **Entry point** — reads `agent_framework.hosted_agents:default` from
     `importlib.metadata` (respects `AGENT_ENTRY_POINT` env var).
   - **Registry** — calls `PurposeDrivenAgent.get_hosted_agent()` for the
     most-derived registered subclass.
   - **Fallback** — uses `PurposeDrivenAgent` itself (test-only).
3. Instantiates the class and hands it to `AgentServer.serve()`.

### Entry-point declaration

This repo declares `PurposeDrivenAgent` as the default hosted agent in
`pyproject.toml`:

```toml
[project.entry-points."agent_framework.hosted_agents"]
default = "purpose_driven_agent.agent:PurposeDrivenAgent"
```

Leaf agent repos (e.g. `founder-agent`) override this with their own
concrete class, requiring zero changes to the hosting plumbing.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `AGENT_ENTRY_POINT` | `default` | Entry point name to look up |
| `AGENT_SERVICE_PORT` | `8000` | HTTP port the FAS server listens on |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `PYTHONPATH` | `/app` | Module search path (set in Dockerfile) |

---

## Routing Tag Enforcement

The FAS workflow reads routing tags from every agent response to decide which
agent to invoke next.  `PurposeDrivenAgent` enforces tag presence and validity
in code — unconditionally, without relying on LLM compliance.

### Tag protocol

| Tag | Emitter | Meaning |
|---|---|---|
| `[ROUTE:CFO]` | Orchestrator agents | Route to CFO specialist |
| `[ROUTE:CMO]` | Orchestrator agents | Route to CMO specialist |
| `[COMPLETE]` | Orchestrator agents | End deliberation |
| `[HANDBACK]` | Specialist agents | Hand control back to orchestrator |

### Enforcement method

```python
# On every LLM response before returning it to the workflow:
response = agent.enforce_routing_tag(llm_output)
```

The algorithm:
1. Scan the last 120 characters of the response for any known tag.
2. Tag present and allowed → return unchanged.
3. Tag present but disallowed → replace with the agent's default tag.
4. No tag → append the agent's default tag on a new line.

Override `get_routing_tags()` and `get_default_routing_tag()` in subclasses
to restrict the allowed set.

---

## RoutingMixin

Concrete subclasses declare their role with `RoutingMixin` — no need to
implement `get_routing_tags` and `get_default_routing_tag` manually:

```python
from purpose_driven_agent.routing_mixin import RoutingMixin

class FounderAgent(RoutingMixin, BusinessAgent):
    ROUTING_ROLE = "orchestrator"
    # allowed tags: [ROUTE:CFO], [ROUTE:CMO], [COMPLETE]
    # default tag:  [COMPLETE]

class CFOAgent(RoutingMixin, BusinessAgent):
    ROUTING_ROLE = "specialist"
    # allowed tags: [HANDBACK]
    # default tag:  [HANDBACK]
```

`RoutingMixin` must appear **before** `PurposeDrivenAgent` in the MRO so that
its `get_routing_tags` and `get_default_routing_tag` override the base
`NotImplementedError` raises.

---

The `adapter_name` parameter maps the agent's purpose to a LoRA adapter:

```python
GenericPurposeDrivenAgent(adapter_name="finance")   # → finance domain
GenericPurposeDrivenAgent(adapter_name="legal")     # → legal domain
GenericPurposeDrivenAgent(adapter_name="hr")        # → HR domain
```

In the full AOS runtime, the **LoRAx** server superimposes selected adapters
concurrently on a shared base model — enabling memory-efficient multi-agent
deployments with minimal per-adapter overhead.

---

## MCP Context Preservation

Every agent has a dedicated `ContextMCPServer` instance automatically created
during `initialize()`.  It stores the agent's full state persistently:

```python
# Automatic (set by the agent on every event)
# purpose, purpose_scope, active_goals,
# completed_goals, purpose_metrics, wake_count, last_active …

# Manual — store anything
await agent.mcp_context_server.set_context("q2_budget", 500_000)
budget = await agent.mcp_context_server.get_context("q2_budget")

# Semantic memory
await agent.mcp_context_server.add_memory({
    "type": "decision", "summary": "Approved vendor contract"
})

# Event history
await agent.mcp_context_server.add_event({"type": "report_filed", "quarter": "Q2"})
```

---

## Perpetual Operation Pattern

```
initialize()  →  loads saved state from MCP
    │
start()       →  spawns _perpetual_loop() background task
    │
    ▼
[Event arrives] → _awaken() → handle_event() → _sleep()
       │
       ├── evaluate_purpose_alignment()
       ├── dispatch to subscribed handlers
       └── _save_context_to_mcp()
    │
stop()        →  saves state, exits loop
```

---

## Configuration

Pass a `config` dict to customise behaviour:

```python
agent = GenericPurposeDrivenAgent(
    agent_id="configured-agent",
    purpose="...",
    config={
        "context_server": {
            "max_history_size": 5000,   # event history limit (default 1000)
            "max_memory_size": 2000,    # memory item limit (default 500)
        }
    },
)
```

---

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=purpose_driven_agent --cov-report=term-missing

# Single test file
pytest tests/test_purpose_driven_agent.py -v
```

Tests use `pytest-asyncio` with `asyncio_mode = "auto"` (configured in
`pyproject.toml`).  No `@pytest.mark.asyncio` decorator is required on
individual test functions.

---

## API Reference

Full API documentation: [`docs/api-reference.md`](docs/api-reference.md)

Key classes:

| Class | Module | Description |
|---|---|---|
| `PurposeDrivenAgent` | `purpose_driven_agent` | Abstract base class |
| `GenericPurposeDrivenAgent` | `purpose_driven_agent` | Concrete general-purpose implementation |
| `A2AAgentTool` | `purpose_driven_agent` | Agent-to-Agent tool representation |
| `ContextMCPServer` | `purpose_driven_agent` | Lightweight MCP context server |
| `IMLService` | `purpose_driven_agent` | Abstract ML service interface |
| `NoOpMLService` | `purpose_driven_agent` | No-op placeholder implementation |
| `RoutingMixin` | `purpose_driven_agent.routing_mixin` | Orchestrator/specialist role declaration |
| `RoutingClassifier` | `aos_mcp_servers.routing` | Stateless routing tag detector |

---

## Contributing

See [`docs/contributing.md`](docs/contributing.md) for setup, testing,
linting, and pull-request guidelines.

Quick start:

```bash
git clone https://github.com/ASISaga/purpose-agent.git
cd purpose-agent
pip install -e ".[dev]"
pytest tests/ -v
pylint src/purpose_driven_agent
```

---

## Related Packages

| Package | Description |
|---|---|
| [`leadership-agent`](https://github.com/ASISaga/leadership-agent) | LeadershipAgent: Layer 3 — decision-making, multi-agent orchestration |
| [`business-agent`](https://github.com/ASISaga/business-agent) | BusinessAgent: Layer 4 — business strategy and operations |
| [`founder-agent`](https://github.com/ASISaga/founder-agent) | FounderAgent: Layer 5 — FAS-hosted orchestrator |
| [`AgentOperatingSystem`](https://github.com/ASISaga/AgentOperatingSystem) | Full AOS runtime with Azure, LoRAx, and orchestration |

---

## License

[Apache License 2.0](LICENSE) — © 2024 ASISaga
