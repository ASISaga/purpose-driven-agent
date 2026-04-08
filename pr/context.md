The **subconscious.asisaga.com** MCP server is the context management backend for all AOS agents.
It is a multi-agent **conversation persistence** service hosted as an Azure Functions app, exposing a
[Model Context Protocol](https://modelcontextprotocol.io) streamable-HTTP endpoint at `/mcp`.

### 1. The Architecture

The "Context Pipeline" has three layers:

1. **MCP server** (`subconscious.asisaga.com/mcp`) — persists orchestration records and conversation
   history in Azure Table Storage.
2. **`SubconsciousContextProvider`** — calls `get_conversation` to retrieve prior conversation history
   and engineers it into a `CONVERSATION HISTORY` instruction block injected into the agent's LLM
   context.  Also exposes `persist_message` and `persist_conversation_turn` to write new messages
   back to the server after the agent processes an event.
3. **`PurposeDrivenAgent`** — receives the injected context via `handle_event` and caches it in its
   `ContextMCPServer` for cross-restart access.

### 2. MCP Tools (server-side)

| Tool | Parameters | Description |
|---|---|---|
| `create_orchestration` | `orchestration_id`, `purpose`, `agents?` | Register a new orchestration |
| `persist_message` | `orchestration_id`, `agent_id`, `role`, `content`, `metadata?` | Append one message |
| `persist_conversation_turn` | `orchestration_id`, `messages` | Persist multiple messages at once |
| `get_conversation` | `orchestration_id`, `limit=200` | Retrieve full conversation history |
| `list_orchestrations` | `status?` | List all orchestrations |
| `complete_orchestration` | `orchestration_id`, `summary?` | Mark orchestration as completed |

MCP Resource: `orchestration://{orchestration_id}` — full metadata + history.

### 3. Implementation

```python
from purpose_driven_agent import GenericPurposeDrivenAgent
from purpose_driven_agent.context_provider import create_subconscious_provider

# One-call factory wires up the live server via agent_framework.MCPStreamableHTTPTool
provider = create_subconscious_provider(orchestration_id="orch-cmo-2026-q2")

agent = GenericPurposeDrivenAgent(
    agent_id="cmo",
    purpose="Lead marketing strategy and brand growth",
    adapter_name="marketing",
)
await agent.initialize()
agent.set_context_provider(provider)

# handle_event fetches conversation history and injects it as context
result = await agent.handle_event({"type": "strategy_review"})
# result["injected_context"] == "CONVERSATION HISTORY:\n..."

# Persist the agent's response back to the server
await provider.persist_message(
    agent_id="cmo",
    role="assistant",
    content=result.get("response", ""),
)
```

### 4. Key Advantages

* **Statelessness:** The agent fetches conversation history on every invocation; no stale in-process
  state accumulates between restarts.
* **Single source of truth:** All conversation data lives in Azure Table Storage and is served live
  from the MCP server.
* **Isolation:** Each orchestration has its own `orchestration_id`, so a single server serves all
  15+ repos in the ASI Saga ecosystem.
* **Token efficiency:** The provider uses the `limit` parameter to window conversation history,
  keeping the LLM context window lean.
