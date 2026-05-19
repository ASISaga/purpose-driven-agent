"""
Pytest configuration and shared fixtures for purpose-driven-agent tests.
"""

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from purpose_driven_agent import GenericPurposeDrivenAgent


@pytest.fixture(autouse=True, scope="session")
def _require_agent_framework() -> None:
    """Skip integration suite when agent_framework is unavailable."""
    pytest.importorskip(
        "agent_framework",
        reason="Integration tests require agent_framework.",
    )


@pytest.fixture
def agent_id() -> str:
    """Return a deterministic agent ID for tests."""
    return "test-agent-001"


@pytest.fixture
def basic_agent(agent_id: str) -> GenericPurposeDrivenAgent:
    """Return an uninitialised GenericPurposeDrivenAgent instance."""
    return GenericPurposeDrivenAgent(
        agent_id=agent_id,
        purpose="Test purpose for unit testing",
        purpose_scope="Testing scope",
        adapter_name="test-adapter",
    )


@pytest.fixture
async def initialised_agent(basic_agent: GenericPurposeDrivenAgent) -> GenericPurposeDrivenAgent:
    """Return an initialised GenericPurposeDrivenAgent instance."""
    await basic_agent.initialize()
    return basic_agent
