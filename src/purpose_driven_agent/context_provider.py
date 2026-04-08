"""
context_provider — Bridge for injecting remote MCP context into agent reasoning.

This module implements the Context Pipeline described in ``pr/context.md``.
It integrates with the ``subconscious.asisaga.com`` MCP server — a multi-agent
**conversation persistence** service — to inject prior conversation history
into the :class:`~purpose_driven_agent.PurposeDrivenAgent` reasoning loop,
and to persist new messages produced by the agent.

Architecture
------------
The pipeline has three layers:

1. **MCP server** (``subconscious.asisaga.com/mcp``) — hosts orchestration
   records and conversation history in Azure Table Storage.
2. **SubconsciousContextProvider** — calls ``get_conversation`` to retrieve
   history for a given orchestration and engineers it into a structured
   :class:`Context` object.  Also exposes ``persist_message`` and
   ``persist_conversation_turn`` to write new messages back to the server.
3. **PurposeDrivenAgent** — injects the ``Context.instructions`` into its
   reasoning loop and caches them via :class:`~purpose_driven_agent.ContextMCPServer`.

Key advantages
--------------
- **Statelessness**: the agent fetches conversation history on every
  invocation; no stale in-process state accumulates between runs.
- **Single source of truth**: all conversation data lives in Azure Table
  Storage and is served live from the MCP server.
- **Isolation**: each orchestration has its own ID, so a single server can
  serve all 15+ repos in the ASI Saga ecosystem.
- **Token efficiency**: the provider windows the history via the ``limit``
  parameter before returning it, keeping the context window lean.

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
        create_subconscious_provider(orchestration_id="orch-cmo-2026-q2")
    )

    result = await agent.handle_event({"type": "strategy_review"})
    # result["injected_context"] contains the prior conversation as context
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
    conversation history) into the :class:`~purpose_driven_agent.PurposeDrivenAgent`
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
    """ContextProvider backed by the ``subconscious.asisaga.com`` MCP server.

    Implements the Context Pipeline described in ``pr/context.md``:

    **Reading (context injection)**

    1. Calls ``get_conversation`` (or a configured tool name) on the registered
       MCP server, passing the orchestration ID and an optional message limit.
    2. Normalises the raw output to a string (handles both ``dict`` and ``str``
       results).
    3. Engineers it into a ``CONVERSATION HISTORY`` instruction block.
    4. Returns a :class:`Context` with the instruction block and the
       passed-through messages.

    **Writing (message persistence)**

    Exposes :meth:`persist_message` and :meth:`persist_conversation_turn` so
    that the orchestrating agent can write new messages back to the server
    after processing an event.

    The MCP server uses the ``orchestration_id`` to isolate each agent's
    conversation, enabling a single server to serve all 15+ repos in the ASI
    Saga ecosystem.

    Example::

        from purpose_driven_agent.context_provider import create_subconscious_provider

        provider = create_subconscious_provider(orchestration_id="orch-cmo-2026-q2")
        context = await provider.get_context(messages=[])
        # context.instructions == "CONVERSATION HISTORY:\\n..."

        # Persist a new message after the agent responds:
        await provider.persist_message(
            agent_id="cmo",
            role="assistant",
            content="Marketing strategy reviewed.",
        )
    """

    def __init__(
        self,
        mcp_server: Any,
        orchestration_id: str,
        tool_name: str = "get_conversation",
        limit: int = 200,
    ) -> None:
        """Initialise the SubconsciousContextProvider.

        Args:
            mcp_server: MCP server instance that exposes the subconscious
                tools.  Must implement ``async call_tool(tool_name, params)``
                (satisfies :class:`~purpose_driven_agent.MCPServerProtocol`).
            orchestration_id: Unique identifier for the orchestration whose
                conversation history to retrieve and persist.  Passed to the
                MCP tools as ``orchestration_id``.
            tool_name: Name of the MCP retrieval tool to invoke.  Defaults to
                ``"get_conversation"``.
            limit: Maximum number of messages to retrieve per call.  Defaults
                to ``200``.
        """
        self.mcp_server = mcp_server
        self.orchestration_id = orchestration_id
        self.tool_name = tool_name
        self.limit = limit
        self.logger = logging.getLogger(
            f"purpose_driven_agent.SubconsciousContextProvider.{orchestration_id}"
        )

    async def get_context(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Context:
        """Fetch conversation history and engineer it into a Context object.

        Calls :attr:`tool_name` on :attr:`mcp_server` with this orchestration's
        ID and the configured :attr:`limit`.  The raw output is normalised to a
        string and formatted as a ``CONVERSATION HISTORY`` block.

        If the MCP call fails, an empty instruction string is returned so
        the agent can continue operating in a degraded mode rather than
        raising an exception.

        Args:
            messages: Conversation messages passed through to the returned
                :class:`Context` unchanged.
            **kwargs: Not used; present for interface compatibility.

        Returns:
            :class:`Context` with the engineered conversation history
            instruction block and the passed-through messages.
        """
        try:
            raw_output = await self.mcp_server.call_tool(
                self.tool_name,
                {"orchestration_id": self.orchestration_id, "limit": self.limit},
            )
            # Normalise to string — MCP tools may return dict, list, or str
            if isinstance(raw_output, (dict, list)):
                raw_content = json.dumps(raw_output)
            else:
                raw_content = str(raw_output)

            engineered_context = f"CONVERSATION HISTORY:\n{raw_content}"
            self.logger.debug(
                "SubconsciousContextProvider fetched context for orchestration '%s'",
                self.orchestration_id,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.error(
                "Failed to fetch subconscious context for orchestration '%s': %s",
                self.orchestration_id,
                exc,
            )
            engineered_context = ""

        return Context(instructions=engineered_context, messages=messages)

    async def persist_message(
        self,
        agent_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Append a single message to this orchestration's conversation.

        Calls the ``persist_message`` tool on :attr:`mcp_server`.  If the
        call fails, the error is logged and ``None`` is returned so that the
        agent can continue without interruption.

        Args:
            agent_id: Identifier of the agent producing the message
                (e.g. ``"cmo"``).
            role: Message role — ``"user"``, ``"assistant"``, ``"system"``,
                or ``"tool"``.
            content: Full text content of the message.
            metadata: Optional structured metadata dict (serialised as JSON
                by the server).

        Returns:
            Confirmation dict from the server (with ``sequence`` and
            ``timestamp`` keys), or ``None`` on failure.
        """
        try:
            return await self.mcp_server.call_tool(
                "persist_message",
                {
                    "orchestration_id": self.orchestration_id,
                    "agent_id": agent_id,
                    "role": role,
                    "content": content,
                    "metadata": metadata,
                },
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.error(
                "Failed to persist message for orchestration '%s': %s",
                self.orchestration_id,
                exc,
            )
            return None

    async def persist_conversation_turn(
        self,
        messages: List[Dict[str, Any]],
    ) -> Any:
        """Persist multiple messages for one orchestration turn in a single call.

        Each element in *messages* must contain ``agent_id``, ``role``, and
        ``content`` keys, with an optional ``metadata`` dict.  Calls the
        ``persist_conversation_turn`` tool on :attr:`mcp_server`.

        Args:
            messages: List of message dicts to persist.

        Returns:
            Summary dict from the server (with ``persisted`` count), or
            ``None`` on failure.
        """
        try:
            return await self.mcp_server.call_tool(
                "persist_conversation_turn",
                {
                    "orchestration_id": self.orchestration_id,
                    "messages": messages,
                },
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.error(
                "Failed to persist conversation turn for orchestration '%s': %s",
                self.orchestration_id,
                exc,
            )
            return None


# ---------------------------------------------------------------------------
# Live server factory
# ---------------------------------------------------------------------------

#: Base URL of the live ASI Saga subconscious MCP server.
SUBCONSCIOUS_MCP_URL: str = "https://subconscious.asisaga.com/mcp"


def create_subconscious_provider(
    orchestration_id: str,
    tool_name: str = "get_conversation",
    limit: int = 200,
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
            create_subconscious_provider(orchestration_id="orch-cmo-2026-q2")
        )
        result = await agent.handle_event({"type": "strategy_review"})
        # result["injected_context"] == "CONVERSATION HISTORY:\\n..."

    Args:
        orchestration_id: Unique identifier for the orchestration whose
            conversation history to retrieve and persist
            (e.g. ``"orch-cmo-2026-q2"``).  Forwarded to the MCP tools
            as ``orchestration_id``.
        tool_name: Name of the MCP retrieval tool to invoke.  Defaults to
            ``"get_conversation"``.
        limit: Maximum number of messages to retrieve per call.  Defaults to
            ``200``.
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
        "Created SubconsciousContextProvider for orchestration '%s' → %s",
        orchestration_id,
        mcp_url,
    )

    return SubconsciousContextProvider(
        mcp_server=adapter,
        orchestration_id=orchestration_id,
        tool_name=tool_name,
        limit=limit,
    )
