# purpose-driven-agent Repository Specification

**Version**: 1.0.0  
**Status**: Active  
**Last Updated**: 2026-03-07

## Overview

`purpose-driven-agent` is a **code-only Python library** that provides `PurposeDrivenAgent` — the abstract base class and fundamental building block of the Agent Operating System (AOS). It enables perpetual, purpose-driven AI agents that run indefinitely, maintain rich state across every interaction, and work toward a long-term assigned purpose.

## Scope

- Repository role in the AOS ecosystem
- Technology stack and coding patterns
- Testing and validation workflows
- Key design principles for agents and contributors

## Repository Role

| Concern | Owner |
|---------|-------|
| Abstract agent base class (`PurposeDrivenAgent`) | **purpose-driven-agent** |
| Concrete general-purpose implementation (`GenericPurposeDrivenAgent`) | **purpose-driven-agent** |
| MCP context preservation (`ContextMCPServer`) | **purpose-driven-agent** |
| ML/LoRA adapter interface (`IMLService`) | **purpose-driven-agent** |
| Agent-to-Agent tool representation (`A2AAgentTool`) | **purpose-driven-agent** |
| Domain-specific agents (leadership, CEO, CFO, etc.) | Downstream packages |
| AOS runtime, orchestration, messaging, storage | AOS ecosystem |

`purpose-driven-agent` is **library-only** — it is not deployed as its own service. It is consumed by agent repos (e.g. `leadership-agent`, `ceo-agent`) that are deployed.

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Runtime | Python 3.10+ |
| Core dependency | `agent-framework` — Microsoft Agent Framework |
| Agent hosting adapter | `azure-ai-agentservice-agentframework` |
| Agent hosting engine | `azure-ai-agentservice-core` |
| Authentication | `azure-identity` — keyless Entra Agent ID |
| MCP tool routing | `aos-mcp-servers` |
| Data validation | `pydantic>=2.12` |
| Tests | `pytest` + `pytest-asyncio` |
| Linter | `pylint` |
| Type checking | `mypy` |
| Formatter | `black` + `isort` |

## Directory Structure

```
purpose-driven-agent/
├── src/
│   └── purpose_driven_agent/
│       ├── __init__.py          # Public API exports
│       ├── agent.py             # PurposeDrivenAgent, GenericPurposeDrivenAgent, A2AAgentTool
│       ├── context_server.py    # ContextMCPServer — MCP-based state persistence
│       └── ml_interface.py      # IMLService, NoOpMLService — LoRA adapter interface
├── tests/
│   ├── conftest.py              # Shared pytest fixtures
│   └── test_purpose_driven_agent.py  # pytest unit tests
├── docs/
│   ├── api-reference.md
│   └── contributing.md
├── examples/                    # Usage examples
└── pyproject.toml               # Build config, dependencies, pytest settings
```

## Core Patterns

### GenericPurposeDrivenAgent (simplest usage)

```python
from purpose_driven_agent import GenericPurposeDrivenAgent

agent = GenericPurposeDrivenAgent(
    agent_id="assistant",
    purpose="Assist users with information retrieval and task coordination",
    adapter_name="general",
)

await agent.initialize()
await agent.start()

result = await agent.handle_event({"type": "user_query", "data": {"query": "What is AOS?"}})
print(result["status"])  # "success"

await agent.stop()
```

### Custom Subclass Pattern

```python
from typing import List
from purpose_driven_agent import PurposeDrivenAgent

class LegalAgent(PurposeDrivenAgent):
    def get_agent_type(self) -> List[str]:
        available = self.get_available_personas()
        return ["legal"] if "legal" in available else ["legal"]
```

### Perpetual Operation

All agents are **perpetual and purpose-driven** — they run indefinitely toward their assigned purpose:

```
initialize()  →  loads saved state from MCP
start()       →  spawns _perpetual_loop() background task
[Event] → _awaken() → handle_event() → _sleep()
stop()        →  saves state, exits loop
```

### LoRA Adapter Mapping

```python
GenericPurposeDrivenAgent(adapter_name="finance")    # → finance domain
GenericPurposeDrivenAgent(adapter_name="legal")      # → legal domain
GenericPurposeDrivenAgent(adapter_name="general")    # → general purpose
```

## Testing Workflow

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=purpose_driven_agent --cov-report=term-missing

# Lint
pylint src/purpose_driven_agent

# Type check
mypy src/purpose_driven_agent

# Specific test
pytest tests/test_purpose_driven_agent.py -v -k "test_initialize"
```

**CI**: GitHub Actions runs `pytest` across Python 3.10, 3.11, and 3.12 on every push/PR to `main`.

→ **CI workflow**: `.github/workflows/ci.yml`

## Related Repositories

| Repository | Role |
|-----------|------|
| [leadership-agent](https://github.com/ASISaga/leadership-agent) | LeadershipAgent: decision-making, multi-agent orchestration |
| [ceo-agent](https://github.com/ASISaga/ceo-agent) | CEOAgent: executive + leadership dual-purpose |
| [cfo-agent](https://github.com/ASISaga/cfo-agent) | CFOAgent: finance + leadership dual-purpose |
| [cto-agent](https://github.com/ASISaga/cto-agent) | CTOAgent: technology + leadership dual-purpose |
| [cso-agent](https://github.com/ASISaga/cso-agent) | CSOAgent: security + leadership dual-purpose |
| [cmo-agent](https://github.com/ASISaga/cmo-agent) | CMOAgent: marketing + leadership dual-purpose |
| [AgentOperatingSystem](https://github.com/ASISaga/AgentOperatingSystem) | Full AOS runtime with Azure, LoRAx, and orchestration |
| [aos-mcp-servers](https://github.com/ASISaga/aos-mcp-servers) | MCP tool routing and transport |

## Key Design Principles

1. **Perpetual** — Agents run indefinitely; there is no finite task completion
2. **Purpose-driven** — Every decision is evaluated against a long-term purpose
3. **MCP-preserved** — `ContextMCPServer` persists all state across restarts
4. **LoRA-mapped** — Domain expertise via adapter_name → LoRA adapter
5. **Library-only** — No deployment scaffolding; consumed by agent repositories

## References

→ **Agent framework**: `.github/specs/agent-intelligence-framework.md`  
→ **Conventional tools**: `.github/docs/conventional-tools.md`  
→ **Python coding standards**: `.github/instructions/python.instructions.md`  
→ **purpose-driven-agent patterns**: `.github/instructions/purpose-driven-agent.instructions.md`
