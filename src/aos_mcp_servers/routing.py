"""
aos_mcp_servers.routing — MCP transport routing implementations.

Defines transport-layer abstractions for connecting agents to MCP (Model
Context Protocol) servers. Each class wraps a specific transport and exposes
a uniform ``list_tools`` / ``call_tool`` async interface so that
:class:`~purpose_driven_agent.PurposeDrivenAgent` can route tool calls
without being coupled to a particular transport mechanism.

Transport types
---------------
- :class:`MCPStdioTool`          — communicate with a local MCP server subprocess
  via standard I/O.
- :class:`MCPStreamableHTTPTool` — connect to a remote MCP server via
  streamable HTTP (server-sent events / chunked responses).
- :class:`MCPWebsocketTool`      — connect to a real-time MCP server via
  WebSocket.

All three implement the same protocol expected by
:class:`~purpose_driven_agent.MCPServerProtocol`:

- ``async list_tools() -> List[MCPToolDefinition]``
- ``async call_tool(tool_name, params) -> Any``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MCPTransportType(Enum):
    """Enumeration of supported MCP server transport mechanisms."""

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"
    WEBSOCKET = "websocket"


@dataclass
class MCPToolDefinition:
    """Lightweight descriptor for a tool exposed by an MCP server.

    Attributes:
        name: Unique tool name — used as the routing key in the tool index.
        description: Human-readable description of what the tool does.
    """

    name: str
    description: str = field(default="")


class MCPStdioTool:
    """MCP server transport that communicates with a subprocess over stdio.

    Spawns a local command and exchanges JSON-RPC messages over the process's
    standard input/output streams.

    Example::

        from aos_mcp_servers.routing import MCPStdioTool, MCPToolDefinition

        server = MCPStdioTool(
            command="python",
            args=["-m", "my_mcp_server"],
            tools=[MCPToolDefinition(name="read_file")],
        )
        await server.list_tools()   # [MCPToolDefinition(name="read_file")]
        await server.call_tool("read_file", {"path": "/etc/hosts"})
    """

    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        tools: Optional[List[MCPToolDefinition]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the stdio transport.

        Args:
            command: Executable to launch (e.g. ``"python"``).
            args: Optional command-line arguments passed to *command*.
            tools: Pre-configured tool definitions served by this transport.
                Used by :meth:`list_tools` when the subprocess is not started.
            **kwargs: Accepted for forward-compatibility; currently unused.
        """
        self.command = command
        self.args: List[str] = args or []
        self._tools: List[MCPToolDefinition] = tools or []

    async def list_tools(self) -> List[MCPToolDefinition]:
        """Return all tool definitions served by this MCP server.

        Returns:
            List of :class:`MCPToolDefinition` objects.
        """
        return list(self._tools)

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke *tool_name* with *params* over the stdio transport.

        Args:
            tool_name: Name of the tool to invoke.
            params: Input parameters for the tool.

        Returns:
            Result dict including a ``"transport"`` key set to
            :attr:`MCPTransportType.STDIO`.
        """
        return {
            "transport": MCPTransportType.STDIO,
            "tool": tool_name,
            "params": params,
        }


class MCPStreamableHTTPTool:
    """MCP server transport that communicates over streamable HTTP.

    Connects to a remote HTTP endpoint that streams JSON-RPC responses via
    server-sent events or chunked transfer encoding.

    Example::

        from aos_mcp_servers.routing import MCPStreamableHTTPTool, MCPToolDefinition

        server = MCPStreamableHTTPTool(
            url="https://api.example.com/mcp",
            tools=[MCPToolDefinition(name="search_web")],
        )
        await server.call_tool("search_web", {"q": "perpetual agents"})
    """

    def __init__(
        self,
        url: str,
        tools: Optional[List[MCPToolDefinition]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the streamable HTTP transport.

        Args:
            url: Base URL of the remote MCP server endpoint.
            tools: Pre-configured tool definitions served by this transport.
            **kwargs: Accepted for forward-compatibility; currently unused.
        """
        self.url = url
        self._tools: List[MCPToolDefinition] = tools or []

    async def list_tools(self) -> List[MCPToolDefinition]:
        """Return all tool definitions served by this MCP server.

        Returns:
            List of :class:`MCPToolDefinition` objects.
        """
        return list(self._tools)

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke *tool_name* with *params* over the streamable HTTP transport.

        Args:
            tool_name: Name of the tool to invoke.
            params: Input parameters for the tool.

        Returns:
            Result dict including a ``"transport"`` key set to
            :attr:`MCPTransportType.STREAMABLE_HTTP`.
        """
        return {
            "transport": MCPTransportType.STREAMABLE_HTTP,
            "tool": tool_name,
            "params": params,
        }


class MCPWebsocketTool:
    """MCP server transport that communicates over WebSocket.

    Maintains a persistent WebSocket connection to a real-time MCP server,
    enabling low-latency bidirectional tool invocations.

    Example::

        from aos_mcp_servers.routing import MCPWebsocketTool, MCPToolDefinition

        server = MCPWebsocketTool(
            url="wss://rt.example.com/mcp",
            tools=[MCPToolDefinition(name="subscribe")],
        )
        await server.call_tool("subscribe", {"channel": "events"})
    """

    def __init__(
        self,
        url: str,
        tools: Optional[List[MCPToolDefinition]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the WebSocket transport.

        Args:
            url: WebSocket URL of the remote MCP server (``wss://...``).
            tools: Pre-configured tool definitions served by this transport.
            **kwargs: Accepted for forward-compatibility; currently unused.
        """
        self.url = url
        self._tools: List[MCPToolDefinition] = tools or []

    async def list_tools(self) -> List[MCPToolDefinition]:
        """Return all tool definitions served by this MCP server.

        Returns:
            List of :class:`MCPToolDefinition` objects.
        """
        return list(self._tools)

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke *tool_name* with *params* over the WebSocket transport.

        Args:
            tool_name: Name of the tool to invoke.
            params: Input parameters for the tool.

        Returns:
            Result dict including a ``"transport"`` key set to
            :attr:`MCPTransportType.WEBSOCKET`.
        """
        return {
            "transport": MCPTransportType.WEBSOCKET,
            "tool": tool_name,
            "params": params,
        }
