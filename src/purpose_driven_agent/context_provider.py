"""
context_provider — Bridge for injecting remote MCP context into agent reasoning.

This module implements the Context Pipeline described in ``pr/context.md``.
It bridges remote MCP server data (such as JSONL "subconscious" data from
``subconscious.asisaga.com``) into the :class:`~purpose_driven_agent.PurposeDrivenAgent`
reasoning loop.

Architecture
------------
The pipeline has three layers:

1. **MCP server** — hosts the raw data (JSONL, structured context, etc.).
2. **ContextProvider** — calls an MCP tool to fetch and engineer the data
   into a structured :class:`Context` object.
3. **PurposeDrivenAgent** — injects the ``Context.instructions`` into its
   reasoning loop and persists them via :class:`~purpose_driven_agent.ContextMCPServer`.

Key advantages
--------------
- **Statelessness**: the agent fetches context on every invocation; no
  stale state accumulates between runs.
- **Single source of truth**: data is served live from the MCP server, so
  any update to the underlying JSONL is immediately picked up.
- **Isolation**: each agent passes its own identity to the MCP server, which
  returns only the context relevant to that agent.
- **Token efficiency**: the provider can window or filter messages before
  returning them, keeping the context window lean.

Example::

    from aos_mcp_servers.routing import MCPStreamableHTTPTool, MCPToolDefinition
    from purpose_driven_agent.context_provider import SubconsciousContextProvider
    from purpose_driven_agent import GenericPurposeDrivenAgent

    mcp_server = MCPStreamableHTTPTool(
        url="https://subconscious.asisaga.com/mcp",
        tools=[MCPToolDefinition(name="read_subconscious_jsonl")],
    )
    provider = SubconsciousContextProvider(
        mcp_server=mcp_server,
        agent_name="CMO",
        repo="agent-cmo-repo",
    )

    agent = GenericPurposeDrivenAgent(
        agent_id="cmo",
        purpose="Lead marketing strategy and brand growth",
        adapter_name="marketing",
    )
    await agent.initialize()
    agent.set_context_provider(provider)

    result = await agent.handle_event({"type": "strategy_review"})
    # result["injected_context"] contains the subconscious instructions
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Context:
    """Structured context object injected into the agent's LLM reasoning loop.

    Returned by :meth:`ContextProvider.get_context` and stored in the agent's
    :class:`~purpose_driven_agent.ContextMCPServer` before each reasoning cycle.

    Attributes:
        instructions: System-level instruction block engineered from remote
            data and injected as high-priority context into the LLM window.
        messages: Conversation messages, possibly windowed or filtered by the
            provider for token efficiency.  Defaults to an empty list.
    """

    instructions: str
    messages: List[Dict[str, Any]] = field(default_factory=list)


class ContextProvider(ABC):
    """Abstract base class for agent context providers.

    A ContextProvider bridges an external data source (MCP server, database,
    JSONL repository) into the :class:`~purpose_driven_agent.PurposeDrivenAgent`
    reasoning context.

    Implement :meth:`get_context` to fetch and engineer data into a
    :class:`Context` object.  The agent calls this method before each
    reasoning cycle and stores the resulting instructions in its MCP context
    server.

    Example::

        class MyContextProvider(ContextProvider):
            async def get_context(
                self,
                messages: List[Dict[str, Any]],
                **kwargs: Any,
            ) -> Context:
                data = await fetch_my_data()
                return Context(
                    instructions=f"DOMAIN CONTEXT:\\n{data}",
                    messages=messages,
                )
    """

    @abstractmethod
    async def get_context(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Context:
        """Retrieve and engineer context for injection into the LLM reasoning loop.

        Args:
            messages: Current conversation or event messages.  The provider
                may window or filter these for token efficiency before
                returning them in the :class:`Context`.
            **kwargs: Additional keyword arguments forwarded from the caller.

        Returns:
            :class:`Context` containing engineered instructions and messages.
        """


class SubconsciousContextProvider(ContextProvider):
    """ContextProvider that reads JSONL "subconscious" data from an MCP server.

    Implements the Context Pipeline described in ``pr/context.md``:

    1. Calls ``read_subconscious_jsonl`` (or a configured tool name) on the
       registered MCP server, passing the agent's identity.
    2. Normalises the raw output to a string (handles both dict and str results).
    3. Engineers it into a ``PRIMARY SUBCONSCIOUS CONTEXT`` instruction block.
    4. Returns a :class:`Context` with the instruction block and the
       passed-through messages.

    The MCP server (e.g. ``subconscious.asisaga.com``) uses ``agent_name`` and
    ``repo`` arguments to isolate each agent's context, enabling a single
    server to serve the 15+ repos in the ASI Saga ecosystem.

    Example::

        from aos_mcp_servers.routing import MCPStreamableHTTPTool, MCPToolDefinition
        from purpose_driven_agent.context_provider import SubconsciousContextProvider

        mcp_server = MCPStreamableHTTPTool(
            url="https://subconscious.asisaga.com/mcp",
            tools=[MCPToolDefinition(name="read_subconscious_jsonl")],
        )
        provider = SubconsciousContextProvider(
            mcp_server=mcp_server,
            agent_name="CMO",
            repo="agent-cmo-repo",
        )
        context = await provider.get_context(messages=[])
        # context.instructions == "PRIMARY SUBCONSCIOUS CONTEXT:\\n..."
    """

    def __init__(
        self,
        mcp_server: Any,
        agent_name: str,
        repo: str,
        tool_name: str = "read_subconscious_jsonl",
    ) -> None:
        """Initialise the SubconsciousContextProvider.

        Args:
            mcp_server: MCP server instance that exposes the subconscious
                tool.  Must implement ``async call_tool(tool_name, params)``
                (satisfies :class:`~purpose_driven_agent.MCPServerProtocol`).
            agent_name: Identity of the agent whose subconscious data to
                retrieve (e.g. ``"CMO"``).  Passed to the MCP tool as
                ``agent_name``.
            repo: Repository name containing the agent's JSONL data
                (e.g. ``"agent-cmo-repo"``).  Passed to the MCP tool as
                ``repo``.
            tool_name: Name of the MCP tool to invoke.  Defaults to
                ``"read_subconscious_jsonl"``.
        """
        self.mcp_server = mcp_server
        self.agent_name = agent_name
        self.repo = repo
        self.tool_name = tool_name
        self.logger = logging.getLogger(
            f"purpose_driven_agent.SubconsciousContextProvider.{agent_name}"
        )

    async def get_context(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Context:
        """Fetch subconscious JSONL data and engineer it into a Context object.

        Calls :attr:`tool_name` on :attr:`mcp_server` with this agent's
        identity parameters.  The raw output is normalised to a string and
        formatted as a high-priority ``PRIMARY SUBCONSCIOUS CONTEXT`` block.

        If the MCP call fails, an empty instruction string is returned so
        the agent can continue operating in a degraded mode rather than
        raising an exception.

        Args:
            messages: Conversation messages passed through to the returned
                :class:`Context` unchanged.
            **kwargs: Not used; present for interface compatibility.

        Returns:
            :class:`Context` with the engineered subconscious instruction
            block and the passed-through messages.
        """
        try:
            raw_output = await self.mcp_server.call_tool(
                self.tool_name,
                {"agent_name": self.agent_name, "repo": self.repo},
            )
            # Normalise to string — MCP tools may return dict, list, or str
            if isinstance(raw_output, (dict, list)):
                raw_content = json.dumps(raw_output)
            else:
                raw_content = str(raw_output)

            engineered_context = f"PRIMARY SUBCONSCIOUS CONTEXT:\n{raw_content}"
            self.logger.debug(
                "SubconsciousContextProvider fetched context for agent '%s' from repo '%s'",
                self.agent_name,
                self.repo,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.error(
                "Failed to fetch subconscious context for '%s' from '%s': %s",
                self.agent_name,
                self.repo,
                exc,
            )
            engineered_context = ""

        return Context(instructions=engineered_context, messages=messages)


# ---------------------------------------------------------------------------
# Live server factory
# ---------------------------------------------------------------------------

#: Base URL of the live ASI Saga subconscious MCP server.
SUBCONSCIOUS_MCP_URL: str = "https://subconscious.asisaga.com/mcp"


def create_subconscious_provider(
    agent_name: str,
    repo: str,
    tool_name: str = "read_subconscious_jsonl",
    mcp_url: str = SUBCONSCIOUS_MCP_URL,
) -> SubconsciousContextProvider:
    """Create a :class:`SubconsciousContextProvider` wired to the live
    ``subconscious.asisaga.com`` MCP server.

    Uses ``agent_framework.MCPStreamableHTTPTool`` (the real Microsoft Agent
    Framework HTTP transport) wrapped in an
    :class:`~aos_mcp_servers.routing.AgentFrameworkMCPServerAdapter` that
    adapts its ``**kwargs`` calling convention to the
    :class:`~purpose_driven_agent.MCPServerProtocol` interface expected by
    :class:`SubconsciousContextProvider`.

    Example::

        from purpose_driven_agent import GenericPurposeDrivenAgent
        from purpose_driven_agent.context_provider import create_subconscious_provider

        agent = GenericPurposeDrivenAgent(
            agent_id="cmo",
            purpose="Lead marketing strategy and brand growth",
            adapter_name="marketing",
        )
        await agent.initialize()
        agent.set_context_provider(
            create_subconscious_provider(agent_name="CMO", repo="agent-cmo-repo")
        )
        result = await agent.handle_event({"type": "strategy_review"})
        # result["injected_context"] == "PRIMARY SUBCONSCIOUS CONTEXT:\\n..."

    Args:
        agent_name: Identity of the agent whose subconscious data to fetch
            (e.g. ``"CMO"``).  Forwarded to the MCP tool as ``agent_name``.
        repo: Repository name containing the agent's JSONL subconscious data
            (e.g. ``"agent-cmo-repo"``).  Forwarded to the MCP tool as
            ``repo``.
        tool_name: Name of the MCP tool on the server to invoke.  Defaults to
            ``"read_subconscious_jsonl"``.
        mcp_url: Base URL of the MCP server.  Defaults to
            :data:`SUBCONSCIOUS_MCP_URL` (``https://subconscious.asisaga.com/mcp``).

    Returns:
        A :class:`SubconsciousContextProvider` that connects to the live
        ``subconscious.asisaga.com`` server on first use.

    Raises:
        ImportError: If ``agent_framework`` is not installed.  Install it with
            ``pip install agent-framework`` or ``pip install purpose-driven-agent[azure]``.
    """
    try:
        from agent_framework import MCPStreamableHTTPTool  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "agent-framework is required for create_subconscious_provider(). "
            "Install it with: pip install agent-framework"
        ) from exc

    from aos_mcp_servers.routing import AgentFrameworkMCPServerAdapter

    real_tool = MCPStreamableHTTPTool(
        name="subconscious",
        url=mcp_url,
    )
    adapter: Optional[AgentFrameworkMCPServerAdapter] = AgentFrameworkMCPServerAdapter(real_tool)

    logger = logging.getLogger("purpose_driven_agent.create_subconscious_provider")
    logger.info(
        "Created SubconsciousContextProvider for agent '%s' repo '%s' → %s",
        agent_name,
        repo,
        mcp_url,
    )

    return SubconsciousContextProvider(
        mcp_server=adapter,
        agent_name=agent_name,
        repo=repo,
        tool_name=tool_name,
    )
