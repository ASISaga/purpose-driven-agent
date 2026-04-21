"""
context_provider — Bridge for injecting remote MCP context into agent reasoning.

This module implements the Context Pipeline described in ``pr/context.md``.
It integrates with the ``subconscious.asisaga.com`` MCP server — a multi-agent
**conversation and schema context persistence** service — to inject prior
conversation history and mind-schema documents into the
:class:`~purpose_driven_agent.PurposeDrivenAgent` reasoning loop, and to
persist new messages and schema contexts produced by the agent.

Architecture
------------
The pipeline has three layers:

1. **MCP server** (``subconscious.asisaga.com/mcp``) — hosts orchestration
   records, conversation history, and JSON-LD mind-schema documents
   (Manas, Buddhi, Ahankara, Chitta, entity perspectives) in Azure Table
   Storage.
2. **Context providers** — bridge the MCP server to the agent:

   - :class:`SubconsciousContextProvider` — manages one orchestration:
     retrieves conversation history via ``get_conversation``, creates an
     orchestration via ``create_orchestration``, persists messages via
     ``persist_message`` / ``persist_conversation_turn``, lists all
     orchestrations via ``list_orchestrations``, and closes the orchestration
     via ``complete_orchestration``.
   - :class:`SubconsciousSchemaContextProvider` — manages the JSON-LD mind
     schemas: retrieves a stored schema context document via
     ``get_schema_context``, persists an updated document via
     ``store_schema_context``, lists stored contexts via
     ``list_schema_contexts``, retrieves a schema definition via
     ``get_schema``, lists all schema names via ``list_schemas``, and
     bootstraps schema contexts from the repo's mind-schema files via
     ``initialize_schema_contexts``.

3. **PurposeDrivenAgent** — injects the ``Context.instructions`` into its
   reasoning loop and caches them via :class:`~purpose_driven_agent.ContextMCPServer`.
   Convenience methods cover every subconscious MCP tool so that
   ``Later, logic will be added to invoke these selectively``.

All subconscious MCP tools
--------------------------
Conversation management (via :class:`SubconsciousContextProvider`):

+--------------------------------+-------------------------------------------------------------+
| Tool                           | Method                                                      |
+================================+=============================================================+
| ``create_orchestration``       | :meth:`SubconsciousContextProvider.create_orchestration`    |
+--------------------------------+-------------------------------------------------------------+
| ``get_conversation``           | :meth:`SubconsciousContextProvider.get_context`             |
+--------------------------------+-------------------------------------------------------------+
| ``persist_message``            | :meth:`SubconsciousContextProvider.persist_message`         |
+--------------------------------+-------------------------------------------------------------+
| ``persist_conversation_turn``  | :meth:`SubconsciousContextProvider.persist_conversation_turn`|
+--------------------------------+-------------------------------------------------------------+
| ``list_orchestrations``        | :meth:`SubconsciousContextProvider.list_orchestrations`     |
+--------------------------------+-------------------------------------------------------------+
| ``complete_orchestration``     | :meth:`SubconsciousContextProvider.complete_orchestration`  |
+--------------------------------+-------------------------------------------------------------+

Schema context management (via :class:`SubconsciousSchemaContextProvider`):

+--------------------------------+----------------------------------------------------------------------+
| Tool                           | Method                                                               |
+================================+======================================================================+
| ``get_schema_context``         | :meth:`SubconsciousSchemaContextProvider.get_context` and            |
|                                | :meth:`SubconsciousSchemaContextProvider.get_schema_context`         |
+--------------------------------+----------------------------------------------------------------------+
| ``store_schema_context``       | :meth:`SubconsciousSchemaContextProvider.store_schema_context`       |
+--------------------------------+----------------------------------------------------------------------+
| ``list_schema_contexts``       | :meth:`SubconsciousSchemaContextProvider.list_schema_contexts`       |
+--------------------------------+----------------------------------------------------------------------+
| ``get_schema``                 | :meth:`SubconsciousSchemaContextProvider.get_schema`                 |
+--------------------------------+----------------------------------------------------------------------+
| ``list_schemas``               | :meth:`SubconsciousSchemaContextProvider.list_schemas`               |
+--------------------------------+----------------------------------------------------------------------+
| ``initialize_schema_contexts`` | :meth:`SubconsciousSchemaContextProvider.initialize_schema_contexts` |
+--------------------------------+----------------------------------------------------------------------+

Key advantages
--------------
- **Statelessness**: the agent fetches conversation history and schema
  contexts on every invocation; no stale in-process state accumulates
  between runs.
- **Single source of truth**: all data lives in Azure Table Storage and is
  served live from the MCP server.
- **Isolation**: each orchestration and schema context is keyed by ID, so a
  single server can serve all 15+ repos in the ASI Saga ecosystem.
- **Token efficiency**: the providers window their data before returning it,
  keeping the context window lean.

Example — conversation history::

    from purpose_driven_agent import GenericPurposeDrivenAgent
    from purpose_driven_agent.context_provider import create_subconscious_provider

    agent = GenericPurposeDrivenAgent(
        agent_id="cmo",
        purpose="Lead marketing strategy and brand growth",
        adapter_name="marketing",
    )
    await agent.initialize()
    provider = create_subconscious_provider(orchestration_id="orch-cmo-2026-q2")

    # Register the orchestration on the server before the first event
    await provider.create_orchestration(purpose="Q2 marketing strategy")

    agent.set_context_provider(provider)
    result = await agent.handle_event({"type": "strategy_review"})
    # result["injected_context"] contains the prior conversation as context

    # Mark the orchestration complete when done
    await provider.complete_orchestration(summary="Q2 strategy finalised")

Example — schema context (Manas / agent mind state)::

    from purpose_driven_agent.context_provider import create_subconscious_schema_provider

    schema_provider = create_subconscious_schema_provider(
        schema_name="manas", context_id="cmo"
    )

    # Optionally bootstrap schema contexts from the repo's mind-schema files
    await schema_provider.initialize_schema_contexts()

    agent.set_context_provider(schema_provider)
    result = await agent.handle_event({"type": "strategy_review"})
    # result["injected_context"] contains the agent's Manas document

    # Persist the updated mind state back to the server after processing
    await schema_provider.store_schema_context(updated_manas_document)

    # Discover available schemas and list stored contexts
    schemas = await schema_provider.list_schemas()
    manas_def = await schema_provider.get_schema()
    contexts = await schema_provider.list_schema_contexts()
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

    async def create_orchestration(
        self,
        purpose: str,
        agents: Optional[List[str]] = None,
    ) -> Any:
        """Register this orchestration on the subconscious MCP server.

        Calls ``create_orchestration`` with :attr:`orchestration_id` and the
        given *purpose*.  Should be called once before the first
        :meth:`get_context` or :meth:`persist_message` call.  If the
        orchestration already exists the server is expected to return it
        without error.

        Args:
            purpose: Human-readable purpose for this orchestration
                (e.g. ``"Q2 marketing strategy review"``).
            agents: Optional list of agent IDs that participate in this
                orchestration (e.g. ``["cmo", "cfo"]``).

        Returns:
            Orchestration record dict from the server, or ``None`` on failure.
        """
        try:
            return await self.mcp_server.call_tool(
                "create_orchestration",
                {
                    "orchestration_id": self.orchestration_id,
                    "purpose": purpose,
                    "agents": agents,
                },
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.error(
                "Failed to create orchestration '%s': %s",
                self.orchestration_id,
                exc,
            )
            return None

    async def list_orchestrations(
        self,
        status: Optional[str] = None,
    ) -> Any:
        """List all orchestrations on the subconscious MCP server.

        Calls ``list_orchestrations``.  The result is not filtered by
        :attr:`orchestration_id` — it returns all orchestrations, optionally
        filtered by *status*.

        Args:
            status: Optional status filter (e.g. ``"active"`` or
                ``"completed"``).  When ``None``, all orchestrations are
                returned.

        Returns:
            List of orchestration record dicts from the server, or ``None``
            on failure.
        """
        try:
            return await self.mcp_server.call_tool(
                "list_orchestrations",
                {"status": status},
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.error("Failed to list orchestrations: %s", exc)
            return None

    async def complete_orchestration(
        self,
        summary: Optional[str] = None,
    ) -> Any:
        """Mark this orchestration as completed on the subconscious MCP server.

        Calls ``complete_orchestration`` with :attr:`orchestration_id`.

        Args:
            summary: Optional human-readable summary of the orchestration
                outcome.

        Returns:
            Updated orchestration record from the server, or ``None`` on
            failure.
        """
        try:
            return await self.mcp_server.call_tool(
                "complete_orchestration",
                {
                    "orchestration_id": self.orchestration_id,
                    "summary": summary,
                },
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.error(
                "Failed to complete orchestration '%s': %s",
                self.orchestration_id,
                exc,
            )
            return None


class SubconsciousSchemaContextProvider(ContextProvider):
    """ContextProvider backed by the ``subconscious.asisaga.com`` schema context store.

    Manages **JSON-LD mind-schema documents** (Manas, Buddhi, Ahankara, Chitta,
    and entity perspectives) stored in the subconscious MCP server.

    **Reading (context injection)**

    :meth:`get_context` calls ``get_schema_context`` on the registered MCP
    server, passing ``schema_name`` and ``context_id``.  The raw output is
    normalised to a string and engineered into a ``SCHEMA CONTEXT``
    instruction block injected into the agent's LLM reasoning loop.

    **Writing (schema context persistence)**

    :meth:`store_schema_context` calls ``store_schema_context`` on the MCP
    server to persist an updated JSON-LD mind document back to Azure Table
    Storage.  :meth:`list_schema_contexts` calls ``list_schema_contexts`` to
    enumerate available contexts for this schema.

    Example::

        from purpose_driven_agent.context_provider import (
            create_subconscious_schema_provider,
        )

        provider = create_subconscious_schema_provider(
            schema_name="manas",
            context_id="cmo",
        )

        # Inject Manas document into the agent's reasoning loop:
        context = await provider.get_context(messages=[])
        # context.instructions == "SCHEMA CONTEXT (manas):\\n..."

        # Persist an updated Manas document after the agent processes an event:
        await provider.store_schema_context(updated_manas_document)
    """

    def __init__(
        self,
        mcp_server: Any,
        schema_name: str,
        context_id: str,
    ) -> None:
        """Initialise the SubconsciousSchemaContextProvider.

        Args:
            mcp_server: MCP server instance that exposes the subconscious
                schema context tools.  Must implement
                ``async call_tool(tool_name, params)``
                (satisfies :class:`~purpose_driven_agent.MCPServerProtocol`).
            schema_name: Name of the mind schema to work with.  Must be one
                of ``"manas"``, ``"buddhi"``, ``"ahankara"``, ``"chitta"``,
                ``"action-plan"``, ``"entity-context"``, or
                ``"entity-content"``.
            context_id: Unique identifier for the schema context document to
                retrieve and persist.  Typically the agent's ID
                (e.g. ``"cmo"``).
        """
        self.mcp_server = mcp_server
        self.schema_name = schema_name
        self.context_id = context_id
        self.logger = logging.getLogger(
            f"purpose_driven_agent.SubconsciousSchemaContextProvider"
            f".{schema_name}.{context_id}"
        )

    async def get_context(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Context:
        """Fetch the schema context document and engineer it into a Context object.

        Calls ``get_schema_context`` on :attr:`mcp_server` with
        :attr:`schema_name` and :attr:`context_id`.  The raw output is
        normalised to a string and formatted as a ``SCHEMA CONTEXT`` block.

        If the MCP call fails, an empty instruction string is returned so
        the agent can continue operating in a degraded mode rather than
        raising an exception.

        Args:
            messages: Conversation messages passed through to the returned
                :class:`Context` unchanged.
            **kwargs: Not used; present for interface compatibility.

        Returns:
            :class:`Context` with the engineered schema context instruction
            block and the passed-through messages.
        """
        try:
            raw_output = await self.mcp_server.call_tool(
                "get_schema_context",
                {"schema_name": self.schema_name, "context_id": self.context_id},
            )
            # Normalise to string — MCP tools may return dict, list, or str
            if isinstance(raw_output, (dict, list)):
                raw_content = json.dumps(raw_output)
            else:
                raw_content = str(raw_output)

            engineered_context = f"SCHEMA CONTEXT ({self.schema_name}):\n{raw_content}"
            self.logger.debug(
                "SubconsciousSchemaContextProvider fetched '%s' context for id '%s'",
                self.schema_name,
                self.context_id,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.error(
                "Failed to fetch schema context '%s/%s': %s",
                self.schema_name,
                self.context_id,
                exc,
            )
            engineered_context = ""

        return Context(instructions=engineered_context, messages=messages)

    async def get_schema_context(self) -> Any:
        """Retrieve the raw schema context document from the MCP server.

        Calls ``get_schema_context`` on :attr:`mcp_server` with
        :attr:`schema_name` and :attr:`context_id` and returns the raw
        output without any formatting.

        Returns:
            Raw schema context document returned by the MCP server (typically
            a ``dict`` for a JSON-LD document), or ``None`` on failure.
        """
        try:
            return await self.mcp_server.call_tool(
                "get_schema_context",
                {"schema_name": self.schema_name, "context_id": self.context_id},
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.error(
                "Failed to get schema context '%s/%s': %s",
                self.schema_name,
                self.context_id,
                exc,
            )
            return None

    async def store_schema_context(self, document: Any) -> Any:
        """Persist a JSON-LD schema context document to the MCP server.

        Calls ``store_schema_context`` on :attr:`mcp_server` with
        :attr:`schema_name`, :attr:`context_id`, and the provided *document*.
        If the call fails, the error is logged and ``None`` is returned so
        that the agent can continue without interruption.

        Args:
            document: JSON-LD document conforming to the schema identified
                by :attr:`schema_name`.  Typically a ``dict`` following the
                mind-schema structure (e.g. Manas, Buddhi).

        Returns:
            Confirmation payload from the server, or ``None`` on failure.
        """
        try:
            return await self.mcp_server.call_tool(
                "store_schema_context",
                {
                    "schema_name": self.schema_name,
                    "context_id": self.context_id,
                    "document": document,
                },
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.error(
                "Failed to store schema context '%s/%s': %s",
                self.schema_name,
                self.context_id,
                exc,
            )
            return None

    async def list_schema_contexts(self) -> Any:
        """List stored schema contexts for this schema from the MCP server.

        Calls ``list_schema_contexts`` on :attr:`mcp_server`, filtered by
        :attr:`schema_name`.  If the call fails, the error is logged and
        ``None`` is returned.

        Returns:
            List of available schema context descriptors from the server,
            or ``None`` on failure.
        """
        try:
            return await self.mcp_server.call_tool(
                "list_schema_contexts",
                {"schema_name": self.schema_name},
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.error(
                "Failed to list schema contexts for '%s': %s",
                self.schema_name,
                exc,
            )
            return None

    async def get_schema(self) -> Any:
        """Retrieve the JSON Schema definition for this schema from the MCP server.

        Calls ``get_schema`` on :attr:`mcp_server` with :attr:`schema_name`.
        This returns the *schema definition* (the JSON Schema document that
        describes valid mind documents), not a stored context document.

        Returns:
            Schema definition dict from the server, or ``None`` on failure.
        """
        try:
            return await self.mcp_server.call_tool(
                "get_schema",
                {"schema_name": self.schema_name},
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.error(
                "Failed to get schema definition for '%s': %s",
                self.schema_name,
                exc,
            )
            return None

    async def list_schemas(self) -> Any:
        """List all available mind-schema names from the MCP server.

        Calls ``list_schemas`` on :attr:`mcp_server`.  Returns the names of
        all schema definitions hosted by the server (e.g. ``"manas"``,
        ``"buddhi"``, ``"ahankara"``, ``"chitta"``, ``"action-plan"``,
        ``"entity-context"``, ``"entity-content"``).

        Returns:
            List of schema name strings (or dicts with metadata), or
            ``None`` on failure.
        """
        try:
            return await self.mcp_server.call_tool("list_schemas", {})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.error("Failed to list schemas: %s", exc)
            return None

    async def initialize_schema_contexts(self, force: bool = False) -> Any:
        """Bootstrap schema contexts from the repository's mind-schema files.

        Calls ``initialize_schema_contexts`` on :attr:`mcp_server`.  This
        one-time operation reads the JSON-LD seed documents from the
        ``boardroom/mind/`` directory in the ``subconscious.asisaga.com``
        repository and populates the ``SchemaContexts`` Azure Table with
        initial agent context documents.

        Should be called once before agents start reading from the server.
        It is idempotent by default; set *force* to ``True`` to overwrite
        existing documents.

        Args:
            force: When ``True``, overwrite existing schema context documents
                with the seed data.  Defaults to ``False``.

        Returns:
            Initialisation result dict from the server (e.g. counts of
            created / skipped records), or ``None`` on failure.
        """
        try:
            return await self.mcp_server.call_tool(
                "initialize_schema_contexts",
                {"force": force},
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.error("Failed to initialize schema contexts: %s", exc)
            return None



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


def create_subconscious_schema_provider(
    schema_name: str,
    context_id: str,
    mcp_url: str = SUBCONSCIOUS_MCP_URL,
) -> "SubconsciousSchemaContextProvider":
    """Create a :class:`SubconsciousSchemaContextProvider` wired to the live
    ``subconscious.asisaga.com`` MCP server.

    Uses ``agent_framework.MCPStreamableHTTPTool`` (the real Microsoft Agent
    Framework HTTP transport) wrapped in an
    :class:`~aos_mcp_servers.routing.AgentFrameworkMCPServerAdapter` that
    adapts its ``**kwargs`` calling convention to the
    :class:`~purpose_driven_agent.MCPServerProtocol` interface expected by
    :class:`SubconsciousSchemaContextProvider`.

    The returned provider reads and writes JSON-LD mind-schema documents
    (Manas, Buddhi, Ahankara, Chitta, and entity perspectives) stored in
    Azure Table Storage on the subconscious MCP server.

    Example::

        from purpose_driven_agent import GenericPurposeDrivenAgent
        from purpose_driven_agent.context_provider import (
            create_subconscious_schema_provider,
        )

        agent = GenericPurposeDrivenAgent(
            agent_id="cmo",
            purpose="Lead marketing strategy and brand growth",
            adapter_name="marketing",
        )
        await agent.initialize()
        agent.set_context_provider(
            create_subconscious_schema_provider(schema_name="manas", context_id="cmo")
        )

        # handle_event fetches the Manas document and injects it as context
        result = await agent.handle_event({"type": "strategy_review"})
        # result["injected_context"] == "SCHEMA CONTEXT (manas):\\n..."

        # Persist the updated Manas document after processing:
        schema_provider = agent.context_provider
        await schema_provider.store_schema_context(updated_manas_document)

    Args:
        schema_name: Name of the mind schema to work with.  Must be one of
            ``"manas"``, ``"buddhi"``, ``"ahankara"``, ``"chitta"``,
            ``"action-plan"``, ``"entity-context"``, or ``"entity-content"``.
        context_id: Unique identifier for the schema context document
            (e.g. the agent's ID ``"cmo"``).
        mcp_url: Base URL of the MCP server.  Defaults to
            :data:`SUBCONSCIOUS_MCP_URL` (``https://subconscious.asisaga.com/mcp``).

    Returns:
        A :class:`SubconsciousSchemaContextProvider` that connects to the live
        ``subconscious.asisaga.com`` server on first use.

    Raises:
        ImportError: If ``agent_framework`` is not installed.  Install it with
            ``pip install agent-framework`` or ``pip install purpose-driven-agent[azure]``.
    """
    try:
        from agent_framework import MCPStreamableHTTPTool  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "agent-framework is required for create_subconscious_schema_provider(). "
            "Install it with: pip install agent-framework"
        ) from exc

    from aos_mcp_servers.routing import AgentFrameworkMCPServerAdapter

    real_tool = MCPStreamableHTTPTool(
        name="subconscious",
        url=mcp_url,
    )
    adapter: Optional[AgentFrameworkMCPServerAdapter] = AgentFrameworkMCPServerAdapter(real_tool)

    logger = logging.getLogger("purpose_driven_agent.create_subconscious_schema_provider")
    logger.info(
        "Created SubconsciousSchemaContextProvider for schema '%s' context '%s' → %s",
        schema_name,
        context_id,
        mcp_url,
    )

    return SubconsciousSchemaContextProvider(
        mcp_server=adapter,
        schema_name=schema_name,
        context_id=context_id,
    )
