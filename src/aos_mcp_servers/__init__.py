"""
aos_mcp_servers — MCP (Model Context Protocol) server transport implementations.

Provides routing abstractions and concrete transport classes for connecting
PurposeDrivenAgent to external MCP servers over stdio, streamable HTTP, and
WebSocket transports.

Public API::

    from aos_mcp_servers.routing import (
        MCPToolDefinition,
        MCPTransportType,
        MCPStdioTool,
        MCPStreamableHTTPTool,
        MCPWebsocketTool,
    )
"""

from aos_mcp_servers.routing import (
    AgentFrameworkMCPServerAdapter,
    MCPStdioTool,
    MCPStreamableHTTPTool,
    MCPToolDefinition,
    MCPTransportType,
    MCPWebsocketTool,
)

__all__ = [
    "MCPToolDefinition",
    "MCPTransportType",
    "MCPStdioTool",
    "MCPStreamableHTTPTool",
    "MCPWebsocketTool",
    "AgentFrameworkMCPServerAdapter",
]
