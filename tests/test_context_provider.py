"""
Tests for context_provider module (Context, ContextProvider, SubconsciousContextProvider)
and ContextProvider integration with PurposeDrivenAgent.

Coverage targets
----------------
- Context dataclass stores instructions and messages.
- ContextProvider is abstract and cannot be instantiated directly.
- SubconsciousContextProvider calls the configured MCP tool and returns Context.
- SubconsciousContextProvider handles MCP call failures gracefully.
- SubconsciousContextProvider normalises dict, list, and str MCP outputs.
- Agent.set_context_provider registers the provider.
- handle_event includes "injected_context" in the result.
- handle_event stores injected_context in the MCP context server.
- handle_event with no provider returns injected_context=None.
- Agent constructor accepts context_provider parameter.
- AgentFrameworkMCPServerAdapter adapts **kwargs tool to params-dict interface.
- create_subconscious_provider factory returns a configured SubconsciousContextProvider.
- SUBCONSCIOUS_MCP_URL constant is exported from the top-level package.
"""

import pytest

from typing import Any

from purpose_driven_agent import (
    SUBCONSCIOUS_MCP_URL,
    Context,
    ContextProvider,
    GenericPurposeDrivenAgent,
    SubconsciousContextProvider,
    create_subconscious_provider,
)
from purpose_driven_agent.context_provider import (
    SUBCONSCIOUS_MCP_URL as _SUBCONSCIOUS_MCP_URL_DIRECT,
    Context as ContextDirect,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubMCPServer:
    """Minimal MCP server stub that records calls and returns a fixed output."""

    def __init__(self, output: object = "subconscious data line 1") -> None:
        self.calls: list = []
        self._output = output

    async def call_tool(self, tool_name: str, params: dict) -> object:
        self.calls.append({"tool": tool_name, "params": params})
        return self._output


class FailingMCPServer:
    """MCP server stub that always raises an exception."""

    async def call_tool(self, tool_name: str, params: dict) -> object:
        raise RuntimeError("MCP server unavailable")


class ConcreteContextProvider(ContextProvider):
    """Minimal concrete ContextProvider for abstract-class tests."""

    def __init__(self, instructions: str = "TEST_INSTRUCTIONS") -> None:
        self._instructions = instructions

    async def get_context(self, messages: list, **kwargs) -> Context:
        return Context(instructions=self._instructions, messages=messages)


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


class TestContext:
    def test_instructions_stored(self) -> None:
        ctx = Context(instructions="Hello world")
        assert ctx.instructions == "Hello world"

    def test_messages_default_empty(self) -> None:
        ctx = Context(instructions="x")
        assert ctx.messages == []

    def test_messages_stored(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        ctx = Context(instructions="x", messages=msgs)
        assert ctx.messages == msgs

    def test_importable_from_top_level(self) -> None:
        from purpose_driven_agent import Context as TopLevelContext

        assert TopLevelContext is ContextDirect


# ---------------------------------------------------------------------------
# ContextProvider (abstract)
# ---------------------------------------------------------------------------


class TestContextProviderAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ContextProvider()  # type: ignore[abstract]

    @pytest.mark.asyncio
    async def test_concrete_subclass_works(self) -> None:
        provider = ConcreteContextProvider("my instructions")
        ctx = await provider.get_context(messages=[])
        assert ctx.instructions == "my instructions"

    @pytest.mark.asyncio
    async def test_get_context_passes_messages_through(self) -> None:
        msgs = [{"type": "ping"}]
        provider = ConcreteContextProvider()
        ctx = await provider.get_context(messages=msgs)
        assert ctx.messages == msgs


# ---------------------------------------------------------------------------
# SubconsciousContextProvider
# ---------------------------------------------------------------------------


class TestSubconsciousContextProvider:
    def test_attributes_stored(self) -> None:
        server = StubMCPServer()
        provider = SubconsciousContextProvider(
            mcp_server=server,
            agent_name="CMO",
            repo="agent-cmo-repo",
        )
        assert provider.agent_name == "CMO"
        assert provider.repo == "agent-cmo-repo"
        assert provider.tool_name == "read_subconscious_jsonl"

    def test_custom_tool_name(self) -> None:
        server = StubMCPServer()
        provider = SubconsciousContextProvider(
            mcp_server=server,
            agent_name="CMO",
            repo="agent-cmo-repo",
            tool_name="custom_tool",
        )
        assert provider.tool_name == "custom_tool"

    @pytest.mark.asyncio
    async def test_get_context_calls_mcp_tool(self) -> None:
        server = StubMCPServer("raw jsonl content")
        provider = SubconsciousContextProvider(
            mcp_server=server,
            agent_name="CMO",
            repo="agent-cmo-repo",
        )
        await provider.get_context(messages=[])
        assert len(server.calls) == 1
        assert server.calls[0]["tool"] == "read_subconscious_jsonl"

    @pytest.mark.asyncio
    async def test_get_context_passes_agent_identity(self) -> None:
        server = StubMCPServer()
        provider = SubconsciousContextProvider(
            mcp_server=server,
            agent_name="CFO",
            repo="agent-cfo-repo",
        )
        await provider.get_context(messages=[])
        params = server.calls[0]["params"]
        assert params["agent_name"] == "CFO"
        assert params["repo"] == "agent-cfo-repo"

    @pytest.mark.asyncio
    async def test_instructions_contain_primary_context_header(self) -> None:
        server = StubMCPServer("my jsonl data")
        provider = SubconsciousContextProvider(
            mcp_server=server, agent_name="CMO", repo="repo"
        )
        ctx = await provider.get_context(messages=[])
        assert ctx.instructions.startswith("PRIMARY SUBCONSCIOUS CONTEXT:\n")

    @pytest.mark.asyncio
    async def test_instructions_contain_raw_output(self) -> None:
        server = StubMCPServer("raw data line")
        provider = SubconsciousContextProvider(
            mcp_server=server, agent_name="CMO", repo="repo"
        )
        ctx = await provider.get_context(messages=[])
        assert "raw data line" in ctx.instructions

    @pytest.mark.asyncio
    async def test_messages_passed_through(self) -> None:
        server = StubMCPServer()
        provider = SubconsciousContextProvider(
            mcp_server=server, agent_name="CMO", repo="repo"
        )
        msgs = [{"type": "strategy_review"}]
        ctx = await provider.get_context(messages=msgs)
        assert ctx.messages == msgs

    @pytest.mark.asyncio
    async def test_dict_output_normalised_to_json_string(self) -> None:
        server = StubMCPServer({"content": "structured data", "score": 0.9})
        provider = SubconsciousContextProvider(
            mcp_server=server, agent_name="CMO", repo="repo"
        )
        ctx = await provider.get_context(messages=[])
        assert "structured data" in ctx.instructions
        assert "0.9" in ctx.instructions

    @pytest.mark.asyncio
    async def test_list_output_normalised_to_json_string(self) -> None:
        server = StubMCPServer(["item1", "item2"])
        provider = SubconsciousContextProvider(
            mcp_server=server, agent_name="CMO", repo="repo"
        )
        ctx = await provider.get_context(messages=[])
        assert "item1" in ctx.instructions
        assert "item2" in ctx.instructions

    @pytest.mark.asyncio
    async def test_failing_mcp_returns_empty_instructions(self) -> None:
        server = FailingMCPServer()
        provider = SubconsciousContextProvider(
            mcp_server=server, agent_name="CMO", repo="repo"
        )
        ctx = await provider.get_context(messages=[])
        assert ctx.instructions == ""

    @pytest.mark.asyncio
    async def test_failing_mcp_still_passes_messages(self) -> None:
        server = FailingMCPServer()
        provider = SubconsciousContextProvider(
            mcp_server=server, agent_name="CMO", repo="repo"
        )
        msgs = [{"type": "fallback"}]
        ctx = await provider.get_context(messages=msgs)
        assert ctx.messages == msgs


# ---------------------------------------------------------------------------
# Agent integration: set_context_provider / handle_event
# ---------------------------------------------------------------------------


class TestAgentContextProviderIntegration:
    @pytest.fixture
    def agent(self) -> GenericPurposeDrivenAgent:
        return GenericPurposeDrivenAgent(
            agent_id="ctx-agent",
            purpose="Test context pipeline",
            adapter_name="test",
        )

    @pytest.fixture
    async def init_agent(self, agent: GenericPurposeDrivenAgent) -> GenericPurposeDrivenAgent:
        await agent.initialize()
        return agent

    def test_context_provider_none_by_default(
        self, agent: GenericPurposeDrivenAgent
    ) -> None:
        assert agent.context_provider is None

    def test_set_context_provider_stores_instance(
        self, agent: GenericPurposeDrivenAgent
    ) -> None:
        provider = ConcreteContextProvider()
        agent.set_context_provider(provider)
        assert agent.context_provider is provider

    def test_constructor_accepts_context_provider(self) -> None:
        provider = ConcreteContextProvider()
        agent = GenericPurposeDrivenAgent(
            agent_id="ctor-agent",
            purpose="Test",
            context_provider=provider,
        )
        assert agent.context_provider is provider

    @pytest.mark.asyncio
    async def test_handle_event_injected_context_none_without_provider(
        self, init_agent: GenericPurposeDrivenAgent
    ) -> None:
        result = await init_agent.handle_event({"type": "test"})
        assert result["injected_context"] is None

    @pytest.mark.asyncio
    async def test_handle_event_injected_context_present_with_provider(
        self, init_agent: GenericPurposeDrivenAgent
    ) -> None:
        provider = ConcreteContextProvider("MY_INSTRUCTIONS")
        init_agent.set_context_provider(provider)
        result = await init_agent.handle_event({"type": "test"})
        assert result["injected_context"] == "MY_INSTRUCTIONS"

    @pytest.mark.asyncio
    async def test_handle_event_stores_context_in_mcp(
        self, init_agent: GenericPurposeDrivenAgent
    ) -> None:
        provider = ConcreteContextProvider("STORED_INSTRUCTIONS")
        init_agent.set_context_provider(provider)
        await init_agent.handle_event({"type": "test"})
        stored = await init_agent.mcp_context_server.get_context("injected_context")
        assert stored == "STORED_INSTRUCTIONS"

    @pytest.mark.asyncio
    async def test_handle_event_empty_instructions_not_stored_in_mcp(
        self, init_agent: GenericPurposeDrivenAgent
    ) -> None:
        provider = ConcreteContextProvider("")
        init_agent.set_context_provider(provider)
        await init_agent.handle_event({"type": "test"})
        stored = await init_agent.mcp_context_server.get_context("injected_context")
        # Empty instructions should not overwrite MCP context
        assert stored is None

    @pytest.mark.asyncio
    async def test_handle_event_with_subconscious_provider(
        self, init_agent: GenericPurposeDrivenAgent
    ) -> None:
        server = StubMCPServer("agent subconscious data")
        provider = SubconsciousContextProvider(
            mcp_server=server,
            agent_name="CMO",
            repo="agent-cmo-repo",
        )
        init_agent.set_context_provider(provider)
        result = await init_agent.handle_event({"type": "strategy_review"})
        assert result["status"] == "success"
        assert "PRIMARY SUBCONSCIOUS CONTEXT:" in result["injected_context"]
        assert "agent subconscious data" in result["injected_context"]

    @pytest.mark.asyncio
    async def test_handle_event_passes_event_as_message_to_provider(
        self, init_agent: GenericPurposeDrivenAgent
    ) -> None:
        received_messages: list = []

        class CapturingProvider(ContextProvider):
            async def get_context(self, messages: list, **kwargs) -> Context:
                received_messages.extend(messages)
                return Context(instructions="captured")

        init_agent.set_context_provider(CapturingProvider())
        event = {"type": "capture_test", "data": {"key": "value"}}
        await init_agent.handle_event(event)
        assert len(received_messages) == 1
        assert received_messages[0] == event

    @pytest.mark.asyncio
    async def test_handle_event_failing_provider_still_succeeds(
        self, init_agent: GenericPurposeDrivenAgent
    ) -> None:
        server = FailingMCPServer()
        provider = SubconsciousContextProvider(
            mcp_server=server, agent_name="CMO", repo="repo"
        )
        init_agent.set_context_provider(provider)
        result = await init_agent.handle_event({"type": "test"})
        assert result["status"] == "success"
        assert result["injected_context"] is None

    @pytest.mark.asyncio
    async def test_replace_context_provider(
        self, init_agent: GenericPurposeDrivenAgent
    ) -> None:
        provider1 = ConcreteContextProvider("FIRST")
        provider2 = ConcreteContextProvider("SECOND")
        init_agent.set_context_provider(provider1)
        init_agent.set_context_provider(provider2)
        result = await init_agent.handle_event({"type": "test"})
        assert result["injected_context"] == "SECOND"


# ---------------------------------------------------------------------------
# AgentFrameworkMCPServerAdapter
# ---------------------------------------------------------------------------


class TestAgentFrameworkMCPServerAdapter:
    """Tests for the adapter that bridges agent_framework tools to MCPServerProtocol."""

    def _make_real_tool_stub(self, return_value: object = "af-output") -> Any:
        """Return an object that mimics agent_framework.MCPTool interface."""

        class FakeAgentFrameworkTool:
            def __init__(self, rv: object) -> None:
                self._rv = rv
                self.connected = False
                self.calls: list = []

            async def connect(self, *, reset: bool = False) -> None:
                self.connected = True

            async def call_tool(self, tool_name: str, **kwargs: Any) -> object:
                self.calls.append({"tool": tool_name, "kwargs": kwargs})
                return self._rv

        return FakeAgentFrameworkTool(return_value)

    @pytest.mark.asyncio
    async def test_adapter_list_tools_returns_empty(self) -> None:
        from aos_mcp_servers.routing import AgentFrameworkMCPServerAdapter

        adapter = AgentFrameworkMCPServerAdapter(self._make_real_tool_stub())
        tools = await adapter.list_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_adapter_calls_connect_on_first_call_tool(self) -> None:
        from aos_mcp_servers.routing import AgentFrameworkMCPServerAdapter

        fake_tool = self._make_real_tool_stub()
        adapter = AgentFrameworkMCPServerAdapter(fake_tool)
        assert not fake_tool.connected
        await adapter.call_tool("my_tool", {})
        assert fake_tool.connected

    @pytest.mark.asyncio
    async def test_adapter_does_not_reconnect_on_second_call(self) -> None:
        from aos_mcp_servers.routing import AgentFrameworkMCPServerAdapter

        class CountingTool:
            connect_count = 0
            async def connect(self, *, reset: bool = False) -> None:
                CountingTool.connect_count += 1
            async def call_tool(self, tool_name: str, **kwargs: Any) -> object:
                return "x"

        tool = CountingTool()
        adapter = AgentFrameworkMCPServerAdapter(tool)
        await adapter.call_tool("t", {})
        await adapter.call_tool("t", {})
        assert tool.connect_count == 1

    @pytest.mark.asyncio
    async def test_adapter_translates_params_dict_to_kwargs(self) -> None:
        from aos_mcp_servers.routing import AgentFrameworkMCPServerAdapter

        fake_tool = self._make_real_tool_stub()
        adapter = AgentFrameworkMCPServerAdapter(fake_tool)
        await adapter.call_tool("read_subconscious_jsonl", {"agent_name": "CMO", "repo": "r"})
        assert fake_tool.calls[0]["kwargs"] == {"agent_name": "CMO", "repo": "r"}

    @pytest.mark.asyncio
    async def test_adapter_returns_tool_output(self) -> None:
        from aos_mcp_servers.routing import AgentFrameworkMCPServerAdapter

        fake_tool = self._make_real_tool_stub("subconscious data")
        adapter = AgentFrameworkMCPServerAdapter(fake_tool)
        result = await adapter.call_tool("read_subconscious_jsonl", {})
        assert result == "subconscious data"


# ---------------------------------------------------------------------------
# create_subconscious_provider factory and SUBCONSCIOUS_MCP_URL
# ---------------------------------------------------------------------------


class TestCreateSubconsciousProvider:
    def test_subconscious_mcp_url_constant(self) -> None:
        assert SUBCONSCIOUS_MCP_URL == "https://subconscious.asisaga.com/mcp"

    def test_subconscious_mcp_url_exported_from_top_level(self) -> None:
        assert SUBCONSCIOUS_MCP_URL is _SUBCONSCIOUS_MCP_URL_DIRECT

    def test_create_returns_subconscious_provider(self) -> None:
        provider = create_subconscious_provider(agent_name="CMO", repo="agent-cmo-repo")
        assert isinstance(provider, SubconsciousContextProvider)

    def test_create_sets_agent_name(self) -> None:
        provider = create_subconscious_provider(agent_name="CFO", repo="agent-cfo-repo")
        assert provider.agent_name == "CFO"

    def test_create_sets_repo(self) -> None:
        provider = create_subconscious_provider(agent_name="CMO", repo="agent-cmo-repo")
        assert provider.repo == "agent-cmo-repo"

    def test_create_default_tool_name(self) -> None:
        provider = create_subconscious_provider(agent_name="CMO", repo="r")
        assert provider.tool_name == "read_subconscious_jsonl"

    def test_create_custom_tool_name(self) -> None:
        provider = create_subconscious_provider(
            agent_name="CMO", repo="r", tool_name="custom_tool"
        )
        assert provider.tool_name == "custom_tool"

    def test_create_custom_mcp_url(self) -> None:
        from aos_mcp_servers.routing import AgentFrameworkMCPServerAdapter

        provider = create_subconscious_provider(
            agent_name="CMO", repo="r", mcp_url="https://staging.asisaga.com/mcp"
        )
        # The adapter wraps a real tool with the custom URL
        adapter = provider.mcp_server
        assert isinstance(adapter, AgentFrameworkMCPServerAdapter)
        assert adapter._tool.url == "https://staging.asisaga.com/mcp"

    def test_create_mcp_server_is_adapter(self) -> None:
        from aos_mcp_servers.routing import AgentFrameworkMCPServerAdapter

        provider = create_subconscious_provider(agent_name="CMO", repo="r")
        assert isinstance(provider.mcp_server, AgentFrameworkMCPServerAdapter)

    def test_create_adapter_wraps_real_tool_with_correct_url(self) -> None:
        from agent_framework import MCPStreamableHTTPTool

        provider = create_subconscious_provider(agent_name="CMO", repo="r")
        adapter = provider.mcp_server
        assert isinstance(adapter._tool, MCPStreamableHTTPTool)
        assert adapter._tool.url == SUBCONSCIOUS_MCP_URL
