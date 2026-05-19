"""Framework-free unit tests for routing and hosting discovery logic."""

from __future__ import annotations

import os
from typing import ClassVar

from aos_mcp_servers.routing import ROUTING_TAGS, RoutingClassifier
from purpose_driven_agent import agent as agent_module
from purpose_driven_agent import hosting
from purpose_driven_agent.agent import PurposeDrivenAgent
from purpose_driven_agent.routing_mixin import RoutingMixin


class DummyOrchestrator(PurposeDrivenAgent):
    """Concrete test double for tag-enforcement unit tests."""

    _allowed: ClassVar[frozenset[str]] = frozenset({"[ROUTE:CFO]", "[COMPLETE]"})

    def get_agent_type(self) -> list[str]:
        return ["dummy"]

    def get_default_routing_tag(self) -> str:
        return "[COMPLETE]"

    def get_routing_tags(self) -> frozenset[str]:
        return self._allowed


class SpecialistRoutingMixin(RoutingMixin):
    ROUTING_ROLE = "specialist"


class TestRoutingClassifier:
    def test_extract_tag_case_insensitive(self) -> None:
        text = "Decision made. [route:cfo]"
        assert RoutingClassifier.extract_tag(text) == "[ROUTE:CFO]"

    def test_extract_tag_only_scans_tail(self) -> None:
        text = "[ROUTE:CMO]" + ("x" * 200)
        assert RoutingClassifier.extract_tag(text) is None

    def test_has_tag_and_route_target(self) -> None:
        assert RoutingClassifier.has_tag("Result [HANDBACK]")
        assert RoutingClassifier.route_target("[ROUTE:CMO]") == "CMO"
        assert RoutingClassifier.route_target("[COMPLETE]") is None

    def test_known_tag_set_contains_expected_values(self) -> None:
        assert ROUTING_TAGS == frozenset({"[ROUTE:CFO]", "[ROUTE:CMO]", "[COMPLETE]", "[HANDBACK]"})


class TestRoutingMixin:
    def test_specialist_defaults(self) -> None:
        mixin = SpecialistRoutingMixin()
        assert mixin.get_routing_tags() == frozenset({"[HANDBACK]"})
        assert mixin.get_default_routing_tag() == "[HANDBACK]"


class TestTagEnforcement:
    def test_missing_tag_appends_default(self) -> None:
        agent = DummyOrchestrator(agent_id="a1", purpose="p")
        output = agent.enforce_routing_tag("No tag here")
        assert output.endswith("[COMPLETE]")

    def test_disallowed_tag_is_replaced(self) -> None:
        agent = DummyOrchestrator(agent_id="a2", purpose="p")
        output = agent.enforce_routing_tag("Route elsewhere [HANDBACK]")
        assert "[HANDBACK]" not in output
        assert "[COMPLETE]" in output

    def test_allowed_tag_is_preserved(self) -> None:
        agent = DummyOrchestrator(agent_id="a3", purpose="p")
        text = "Route to finance [ROUTE:CFO]"
        assert agent.enforce_routing_tag(text) == text


class TestHostingDiscovery:
    def test_discover_agent_class_uses_entry_point(self, monkeypatch) -> None:
        class HostedAgent(PurposeDrivenAgent):
            def get_agent_type(self) -> list[str]:
                return ["hosted"]

            def get_default_routing_tag(self) -> str:
                return "[COMPLETE]"

        class StubEntryPoint:
            name = "default"

            @staticmethod
            def load() -> type:
                return HostedAgent

        monkeypatch.setenv("AGENT_ENTRY_POINT", "default")
        monkeypatch.setattr(hosting.importlib.metadata, "entry_points", lambda group=None: [StubEntryPoint()])

        discovered = hosting._discover_agent_class()
        assert discovered is HostedAgent

    def test_discover_agent_class_falls_back_to_registry(self, monkeypatch) -> None:
        monkeypatch.setenv("AGENT_ENTRY_POINT", "does-not-exist")
        monkeypatch.setattr(hosting.importlib.metadata, "entry_points", lambda group=None: [])

        class RegistryAgent(PurposeDrivenAgent):
            def get_agent_type(self) -> list[str]:
                return ["registry"]

            def get_default_routing_tag(self) -> str:
                return "[COMPLETE]"

        discovered = hosting._discover_agent_class()
        assert discovered is RegistryAgent

    def test_discover_agent_class_returns_base_when_registry_empty(self, monkeypatch) -> None:
        monkeypatch.setenv("AGENT_ENTRY_POINT", "missing")
        monkeypatch.setattr(hosting.importlib.metadata, "entry_points", lambda group=None: [])
        monkeypatch.setattr(agent_module, "_AGENT_REGISTRY", {})

        discovered = hosting._discover_agent_class()
        assert discovered is PurposeDrivenAgent


def teardown_module() -> None:
    os.environ.pop("AGENT_ENTRY_POINT", None)
