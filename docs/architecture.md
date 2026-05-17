# Architecture — purpose-driven-agent

## Overview

`purpose-driven-agent` provides the fundamental building block of the **Agent
Operating System (AOS)**: a **Purpose-Driven, Perpetual Agent**.

Unlike traditional task-based agents that start, execute, and terminate,
`PurposeDrivenAgent` runs indefinitely, maintaining rich state across every
interaction.

---

## AOS Container Hierarchy

The AOS agent stack is built as a sequence of Docker image layers.  Each
layer extends the one above it — no layer re-implements what a parent already
provides.

```
infrastructure             (Layer 1) — Python 3.12, MAF 1.3.0, FAS hosting adapter
  └── purpose-driven-agent (Layer 2) — THIS REPO — PurposeDrivenAgent + aos_mcp_servers
        └── leadership-agent    (Layer 3) — multi-agent orchestration
              └── business-agent     (Layer 4) — business strategy
                    └── founder-agent      (Layer 5) — FAS-hosted orchestrator
```

**Layer 2 contract:**

| Layer provides | Layer requires from parent |
|---|---|
| `PurposeDrivenAgent` ABC + `GenericPurposeDrivenAgent` | Python 3.12 runtime |
| `aos_mcp_servers` routing and transport stubs | MAF 1.3.0 installed |
| FAS hosting adapter (`hosting.py`) | `agent-framework-foundry-hosting` |
| `ENV PYTHONPATH=/app` | `WORKDIR /app` |

The compiled `.pyc` artifacts from all layers are assembled in the **FAS
stage** (`FROM base AS fas`): `.py` files are stripped and only `.pyc`
files survive, loaded by `SourcelessFileLoader` at runtime.

---

## Core Design Principles

| Principle | Description |
|---|---|
| **Perpetual** | Agent registers once; runs indefinitely |
| **Event-driven** | Awakens on events; sleeps when idle |
| **Purpose-driven** | Works toward a long-term, assigned purpose |
| **Stateful** | Full state preserved via `ContextMCPServer` (MCP) |
| **Adapter-mapped** | Purpose linked to a LoRA adapter for domain expertise |
| **Autonomous** | Evaluates and aligns decisions with its purpose |

---

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PurposeDrivenAgent                            │
│                     (abstract base class)                            │
│                                                                      │
│  ┌──────────────────────┐   ┌──────────────────────────────────────┐ │
│  │   Purpose Context    │   │          LoRA Adapter                │ │
│  │  ─────────────────   │   │  ──────────────────────────          │ │
│  │  purpose (string)    │   │  adapter_name → domain model         │ │
│  │  purpose_scope       │   │  e.g. "general", "leadership",       │ │
│  │  purpose_metrics     │   │       "marketing", "finance"         │ │
│  └──────────────────────┘   └──────────────────────────────────────┘ │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                    ContextMCPServer (MCP)                        │ │
│  │  ─────────────────────────────────────────────────────────────  │ │
│  │  Persistent key/value context store                              │ │
│  │  Event history tracking                                          │ │
│  │  Semantic memory management                                      │ │
│  │  One dedicated instance per agent                                │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌──────────────────────┐   ┌──────────────────────────────────────┐ │
│  │   Perpetual Loop     │   │          IMLService                  │ │
│  │  ─────────────────   │   │  ──────────────────────────          │ │
│  │  asyncio.create_task │   │  trigger_lora_training()             │ │
│  │  Sleep / Awaken      │   │  run_pipeline()                      │ │
│  │  Event subscriptions │   │  infer()                             │ │
│  └──────────────────────┘   └──────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Inheritance Hierarchy

```
agent_framework.Agent   (Microsoft Agent Framework — optional)
        │
        ▼
PurposeDrivenAgent      ← abstract base class (this package, Layer 2)
        │
        ├── GenericPurposeDrivenAgent   ← concrete, general-purpose
        │
        ├── LeadershipAgent             ← leadership-agent (Layer 3)
        │       └── BusinessAgent       ← business-agent (Layer 4)
        │               └── FounderAgent  ← founder-agent (Layer 5, FAS target)
        │
        └── <YourCustomAgent>           ← extend for your domain
```

---

## LoRA Adapter Pattern

The `adapter_name` parameter maps the agent's purpose to a LoRA (Low-Rank
Adaptation) adapter that provides:

- **Domain vocabulary** — terminology natural to the role.
- **Domain knowledge** — facts, heuristics, and reasoning patterns.
- **Agent persona** — communication style, decision tendencies.

In the full AOS runtime, the AgentOperatingSystem uses **LoRAx** to
superimpose multiple adapters concurrently on a shared base model, enabling
memory-efficient multi-agent deployments.

```python
agent = GenericPurposeDrivenAgent(
    agent_id="cfo",
    purpose="Manage financial health and long-term fiscal strategy",
    adapter_name="finance",   # → LoRA adapter: finance domain knowledge + CFO persona
)
```

---

## MCP (Model Context Protocol) Integration

Every `PurposeDrivenAgent` holds a dedicated `ContextMCPServer` instance that
preserves full state across events and restarts.

### Context stored automatically

| Key | Description |
|---|---|
| `purpose` | Agent's purpose string |
| `purpose_scope` | Scope/boundaries |
| `active_goals` | Currently tracked goals |
| `completed_goals` | Completed goal history |
| `purpose_metrics` | Counters for aligned actions, decisions, etc. |
| `wake_count` | Number of awaken cycles |
| `total_events_processed` | Cumulative event count |
| `last_active` | ISO timestamp of last activity |

### Custom context

```python
await agent.mcp_context_server.set_context("custom_key", {"data": 42})
value = await agent.mcp_context_server.get_context("custom_key")
```

---

## Perpetual Operation Pattern

```
  ┌──────────────────────────────────────┐
  │           initialize()               │  → sets up MCP, loads saved state
  └────────────────────┬─────────────────┘
                       │
  ┌────────────────────▼─────────────────┐
  │              start()                 │  → spawns _perpetual_loop() task
  └────────────────────┬─────────────────┘
                       │
  ┌────────────────────▼─────────────────┐
  │          _perpetual_loop()           │  → runs forever until stop()
  │                                      │
  │  ┌────────────────────────────────┐  │
  │  │  Event received → _awaken()    │  │
  │  │  handle_event(event)           │  │
  │  │  evaluate_purpose_alignment()  │  │
  │  │  dispatch to handlers          │  │
  │  │  _save_context_to_mcp()        │  │
  │  │  _sleep()                      │  │
  │  └────────────────────────────────┘  │
  └────────────────────┬─────────────────┘
                       │ (stop() called)
  ┌────────────────────▼─────────────────┐
  │               stop()                 │  → saves state, sets is_running=False
  └──────────────────────────────────────┘
```

---

## Extending PurposeDrivenAgent

```python
from typing import List
from purpose_driven_agent import PurposeDrivenAgent

class FinanceAgent(PurposeDrivenAgent):
    """Domain-specific agent for financial operations."""

    def get_agent_type(self) -> List[str]:
        available = self.get_available_personas()
        return ["finance"] if "finance" in available else ["finance"]

    async def analyse_budget(self, budget: dict) -> dict:
        alignment = await self.evaluate_purpose_alignment(
            {"type": "budget_analysis"}
        )
        # ... domain logic ...
        return {"status": "analysed", "alignment": alignment}
```

---

## Standalone vs. Full AOS Runtime

| Feature | Standalone (`purpose-driven-agent`) | Full AOS |
|---|---|---|
| ContextMCPServer | In-process dict store | Azure Table / Blob Storage |
| ML Service | `NoOpMLService` (plug in your own) | Azure ML + LoRAx |
| Event bus | In-process subscriptions | Azure Service Bus |
| Multi-agent | Not included | Full orchestration engine |
| Personas | Configurable list | AOS persona registry |

The standalone package is designed for development, testing, and environments
where Azure services are not available.  Swap each component by providing an
implementation of the relevant interface.

---

## FAS Hosting Adapter

Layer 2 includes `purpose_driven_agent/hosting.py` — the bridge between the
`agent-framework-foundry-hosting` server (installed in Layer 1) and
`PurposeDrivenAgent`.

### Entry point

```
python -m purpose_driven_agent
   └── purpose_driven_agent/__main__.py
         └── hosting.run_server()
               ├── _ensure_imports()      — seeds __init_subclass__ registry
               ├── _discover_agent_class() — three-strategy discovery
               └── AgentServer.serve()    — starts HTTP listener
```

### Agent class discovery

| Strategy | Mechanism | When used |
|---|---|---|
| 1 | `importlib.metadata` entry point `agent_framework.hosted_agents:default` | Package installed |
| 2 | `PurposeDrivenAgent.get_hosted_agent()` — most-derived registered subclass | Running from `PYTHONPATH` |
| 3 | `PurposeDrivenAgent` itself | Test / fallback only |

### `__init_subclass__` registry

Every subclass is automatically registered at import time:

```python
_AGENT_REGISTRY: dict[str, type[PurposeDrivenAgent]] = {}

class PurposeDrivenAgent:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _AGENT_REGISTRY[cls.__qualname__] = cls
```

`get_hosted_agent()` returns the entry with the longest MRO (most-derived
class) — meaning `FounderAgent` wins over `LeadershipAgent` in a Layer 5
image.

---

## Routing Tag Protocol

The FAS workflow reads routing tags from agent responses to decide the next
routing step.  Tags are enforced in Python code — unconditionally — by
`PurposeDrivenAgent.enforce_routing_tag()`.

### Tags

| Tag | Emitted by | Meaning |
|---|---|---|
| `[ROUTE:CFO]` | Orchestrator agents | Route to CFO specialist |
| `[ROUTE:CMO]` | Orchestrator agents | Route to CMO specialist |
| `[COMPLETE]` | Orchestrator agents | End deliberation |
| `[HANDBACK]` | Specialist agents | Hand control back to orchestrator |

### Enforcement flow

```
LLM produces response text
        │
enforce_routing_tag(response_text)
        │
        ├── Scan last 120 chars for any known tag
        │       │
        │       ├── tag found AND in allowed set  →  return unchanged
        │       ├── tag found BUT not in allowed  →  replace with default tag
        │       └── no tag found                 →  append default tag
        │
        └── return enforced response
```

### `RoutingMixin` — declarative role declaration

```python
from purpose_driven_agent.routing_mixin import RoutingMixin

class FounderAgent(RoutingMixin, BusinessAgent):
    ROUTING_ROLE = "orchestrator"   # allowed: [ROUTE:CFO], [ROUTE:CMO], [COMPLETE]

class CFOAgent(RoutingMixin, BusinessAgent):
    ROUTING_ROLE = "specialist"     # allowed: [HANDBACK]
```

`RoutingMixin` implements `get_routing_tags()` and `get_default_routing_tag()`
based on `ROUTING_ROLE`, satisfying the abstract contract on `PurposeDrivenAgent`.
