"""
Tests for MindLoader — the schema-based mind file loader for PurposeDrivenAgent.

These tests are self-contained: they build minimal but schema-valid JSON-LD
fixture files under a pytest ``tmp_path`` temporary directory and exercise
every public method of :class:`~purpose_driven_agent.mind_loader.MindLoader`.

No boardroom-specific or business-infinity dependencies are required.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from purpose_driven_agent.mind_loader import MindLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_ID = "test-agent"


def _write(path: Path, data: Dict[str, Any]) -> Path:
    """Write *data* as JSON to *path*, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_mind_dir(tmp_path: Path, agent_id: str = AGENT_ID) -> Path:
    """Create a minimal but schema-valid mind directory under *tmp_path*.

    Returns the ``mind_dir`` root (the directory that contains the
    ``{agent_id}/`` subdirectory and the ``schemas/`` directory).
    """
    mind_dir = tmp_path / "mind"

    # ── Manas ────────────────────────────────────────────────────────────────
    _write(
        mind_dir / agent_id / "Manas" / f"{agent_id}.jsonld",
        {
            "@context": "https://asisaga.com/ontology/v1/context.jsonld",
            "@id": f"https://asisaga.com/agents/{agent_id}",
            "@type": "Manas",
            "schema_version": "1.0.0",
            "context": {"name": "Test Agent"},
            "context_management": {"access": "read-only", "mutability": "immutable", "manager": "system"},
            "content": {"current_focus": "testing"},
            "content_management": {"access": "read-write", "mutability": "mutable", "manager": "system"},
        },
    )
    _write(
        mind_dir / agent_id / "Manas" / "context" / "company.jsonld",
        {
            "@context": "https://asisaga.com/ontology/v1/context.jsonld",
            "@id": f"https://asisaga.com/agents/{agent_id}/context/company",
            "@type": "EntityContext",
            "name": "Test Company",
            "agent_perspective": agent_id,
            "legend": "Test Legend",
            "domain_knowledge": ["domain A"],
            "skills": ["skill X"],
            "persona": "Analytical thinker",
            "language": "Formal, precise",
        },
    )
    _write(
        mind_dir / agent_id / "Manas" / "content" / "company.jsonld",
        {
            "@context": "https://asisaga.com/ontology/v1/context.jsonld",
            "@id": f"https://asisaga.com/agents/{agent_id}/content/company",
            "@type": "EntityContent",
            "name": "Test Company",
            "agent_perspective": agent_id,
            "legend": "Test Legend",
            "perspective": "From a testing perspective",
            "software_interfaces": [],
            "current_signals": ["signal 1"],
        },
    )

    # ── Buddhi ───────────────────────────────────────────────────────────────
    _write(
        mind_dir / agent_id / "Buddhi" / "buddhi.jsonld",
        {
            "@context": "https://asisaga.com/ontology/v1/context.jsonld",
            "@id": f"https://asisaga.com/agents/{agent_id}/buddhi",
            "@type": "Buddhi",
            "schema_version": "1.0.0",
            "agent_id": agent_id,
            "name": "Test Agent",
            "domain": "Testing",
            "domain_knowledge": ["knowledge A", "knowledge B"],
            "skills": ["skill X", "skill Y"],
            "persona": "Methodical and precise",
            "language": "Clear and structured",
        },
    )
    _write(
        mind_dir / agent_id / "Buddhi" / "action-plan.jsonld",
        {
            "@context": "https://asisaga.com/ontology/v1/context.jsonld",
            "@type": "AgentActionPlan",
            "name": "Test Agent",
            "role": "Tester",
            "anchor": "Quality",
            "status": "Active",
            "overarchingPurpose": {"description": "Ensure quality"},
            "actionSteps": [{"@type": "Task", "stepId": "T001", "taskName": "Run_Tests"}],
        },
    )

    # ── Ahankara ─────────────────────────────────────────────────────────────
    _write(
        mind_dir / agent_id / "Ahankara" / "ahankara.jsonld",
        {
            "@context": "https://asisaga.com/ontology/v1/context.jsonld",
            "@id": f"https://asisaga.com/agents/{agent_id}/ahankara",
            "@type": "Ahankara",
            "schema_version": "1.0.0",
            "agent_id": agent_id,
            "name": "Test Agent",
            "identity": "Quality Guardian",
            "contextual_axis": "Testing excellence",
            "non_negotiables": ["Correctness"],
            "identity_markers": ["Rigorous", "Systematic"],
            "intellect_constraint": "Must validate before shipping",
        },
    )

    # ── Chitta ───────────────────────────────────────────────────────────────
    _write(
        mind_dir / agent_id / "Chitta" / "chitta.jsonld",
        {
            "@context": "https://asisaga.com/ontology/v1/context.jsonld",
            "@id": f"https://asisaga.com/agents/{agent_id}/chitta",
            "@type": "Chitta",
            "schema_version": "2.0.0",
            "agent_id": agent_id,
            "name": "Test Agent",
            "intelligence_nature": "Investigative clarity",
            "cosmic_intelligence": "Universal quality principle",
            "beyond_identity": "Testing transcends any single role",
            "consciousness_basis": "Awareness of correctness",
        },
    )

    # ── Schemas directory (empty but present) ─────────────────────────────
    (mind_dir / "schemas").mkdir(parents=True, exist_ok=True)
    return mind_dir


# ---------------------------------------------------------------------------
# Test: schema registry
# ---------------------------------------------------------------------------


class TestMindFileSchemas:
    """Tests for the MIND_FILE_SCHEMAS class attribute."""

    def test_all_expected_keys_registered(self) -> None:
        """MIND_FILE_SCHEMAS contains all expected dimension keys."""
        expected = {
            "Manas/state",
            "Buddhi/buddhi.jsonld",
            "Buddhi/action-plan.jsonld",
            "Ahankara/ahankara.jsonld",
            "Chitta/chitta.jsonld",
            "Manas/context/entity",
            "Manas/content/entity",
        }
        assert expected == set(MindLoader.MIND_FILE_SCHEMAS.keys())

    def test_each_entry_has_required_keys(self) -> None:
        """Every schema entry has 'description', 'schema_file', 'required_keys'."""
        for key, schema in MindLoader.MIND_FILE_SCHEMAS.items():
            assert "description" in schema, f"{key}: missing 'description'"
            assert "schema_file" in schema, f"{key}: missing 'schema_file'"
            assert "required_keys" in schema, f"{key}: missing 'required_keys'"
            assert isinstance(schema["required_keys"], set), (
                f"{key}: 'required_keys' must be a set"
            )
            assert len(schema["required_keys"]) > 0, f"{key}: 'required_keys' is empty"

    def test_list_registered_dimensions(self) -> None:
        """list_registered_dimensions returns all 7 schema keys."""
        dims = MindLoader.list_registered_dimensions()
        assert len(dims) == 7
        assert "Buddhi/buddhi.jsonld" in dims


# ---------------------------------------------------------------------------
# Test: get_schemas_dir
# ---------------------------------------------------------------------------


class TestGetSchemasDir:
    def test_returns_schemas_subdir(self, tmp_path: Path) -> None:
        mind_dir = tmp_path / "mind"
        schemas_dir = MindLoader.get_schemas_dir(mind_dir)
        assert schemas_dir == mind_dir / "schemas"

    def test_does_not_require_dir_to_exist(self, tmp_path: Path) -> None:
        mind_dir = tmp_path / "nonexistent"
        schemas_dir = MindLoader.get_schemas_dir(mind_dir)
        assert schemas_dir.name == "schemas"


# ---------------------------------------------------------------------------
# Test: _resolve_schema_key
# ---------------------------------------------------------------------------


class TestResolveSchemaKey:
    def test_manas_state_file(self) -> None:
        assert MindLoader._resolve_schema_key("Manas", "ceo.jsonld") == "Manas/state"

    def test_manas_state_arbitrary_agent(self) -> None:
        assert MindLoader._resolve_schema_key("Manas", "my-agent.jsonld") == "Manas/state"

    def test_buddhi_buddhi(self) -> None:
        assert MindLoader._resolve_schema_key("Buddhi", "buddhi.jsonld") == "Buddhi/buddhi.jsonld"

    def test_buddhi_action_plan(self) -> None:
        assert (
            MindLoader._resolve_schema_key("Buddhi", "action-plan.jsonld")
            == "Buddhi/action-plan.jsonld"
        )

    def test_ahankara(self) -> None:
        assert (
            MindLoader._resolve_schema_key("Ahankara", "ahankara.jsonld")
            == "Ahankara/ahankara.jsonld"
        )

    def test_chitta(self) -> None:
        assert (
            MindLoader._resolve_schema_key("Chitta", "chitta.jsonld")
            == "Chitta/chitta.jsonld"
        )

    def test_manas_context_entity(self) -> None:
        assert (
            MindLoader._resolve_schema_key("Manas/context", "company.jsonld")
            == "Manas/context/entity"
        )

    def test_manas_content_entity(self) -> None:
        assert (
            MindLoader._resolve_schema_key("Manas/content", "business.jsonld")
            == "Manas/content/entity"
        )


# ---------------------------------------------------------------------------
# Test: load_mind_file
# ---------------------------------------------------------------------------


class TestLoadMindFile:
    def test_load_buddhi(self, tmp_path: Path) -> None:
        mind_dir = _make_mind_dir(tmp_path)
        doc = MindLoader.load_mind_file(mind_dir, AGENT_ID, "Buddhi", "buddhi.jsonld")
        assert doc["@type"] == "Buddhi"
        assert doc["agent_id"] == AGENT_ID

    def test_load_ahankara(self, tmp_path: Path) -> None:
        mind_dir = _make_mind_dir(tmp_path)
        doc = MindLoader.load_mind_file(mind_dir, AGENT_ID, "Ahankara", "ahankara.jsonld")
        assert doc["@type"] == "Ahankara"
        assert "identity" in doc
        assert "contextual_axis" in doc
        assert "non_negotiables" in doc
        assert "identity_markers" in doc
        assert "intellect_constraint" in doc

    def test_load_chitta(self, tmp_path: Path) -> None:
        mind_dir = _make_mind_dir(tmp_path)
        doc = MindLoader.load_mind_file(mind_dir, AGENT_ID, "Chitta", "chitta.jsonld")
        assert doc["@type"] == "Chitta"
        for field in ("intelligence_nature", "cosmic_intelligence", "beyond_identity", "consciousness_basis"):
            assert field in doc, f"Chitta missing '{field}'"

    def test_load_action_plan(self, tmp_path: Path) -> None:
        mind_dir = _make_mind_dir(tmp_path)
        doc = MindLoader.load_mind_file(mind_dir, AGENT_ID, "Buddhi", "action-plan.jsonld")
        assert "@type" in doc
        assert "actionSteps" in doc
        assert len(doc["actionSteps"]) >= 1

    def test_load_manas_state(self, tmp_path: Path) -> None:
        mind_dir = _make_mind_dir(tmp_path)
        doc = MindLoader.load_mind_file(
            mind_dir, AGENT_ID, "Manas", f"{AGENT_ID}.jsonld"
        )
        assert doc["@type"] == "Manas"
        assert "context" in doc
        assert "content" in doc

    def test_load_manas_context_entity(self, tmp_path: Path) -> None:
        mind_dir = _make_mind_dir(tmp_path)
        doc = MindLoader.load_mind_file(
            mind_dir, AGENT_ID, "Manas/context", "company.jsonld"
        )
        assert "agent_perspective" in doc
        assert "domain_knowledge" in doc
        assert "skills" in doc

    def test_load_manas_content_entity(self, tmp_path: Path) -> None:
        mind_dir = _make_mind_dir(tmp_path)
        doc = MindLoader.load_mind_file(
            mind_dir, AGENT_ID, "Manas/content", "company.jsonld"
        )
        assert "agent_perspective" in doc
        assert "perspective" in doc
        assert "current_signals" in doc

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        mind_dir = _make_mind_dir(tmp_path)
        with pytest.raises(FileNotFoundError):
            MindLoader.load_mind_file(
                mind_dir, AGENT_ID, "Buddhi", "nonexistent.jsonld"
            )

    def test_invalid_json_raises_value_error(self, tmp_path: Path) -> None:
        mind_dir = _make_mind_dir(tmp_path)
        bad_file = mind_dir / AGENT_ID / "Buddhi" / "bad.jsonld"
        bad_file.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="Unable to parse JSON"):
            MindLoader.load_mind_file(mind_dir, AGENT_ID, "Buddhi", "bad.jsonld")

    def test_missing_required_key_raises_value_error(self, tmp_path: Path) -> None:
        """A document missing schema-required keys raises ValueError."""
        mind_dir = _make_mind_dir(tmp_path)
        # Overwrite buddhi.jsonld with a document missing 'domain_knowledge'
        incomplete = {
            "@context": "https://example.com",
            "@id": "https://example.com/x",
            "@type": "Buddhi",
            "schema_version": "1.0.0",
            "agent_id": AGENT_ID,
            "name": "Test",
            "domain": "Testing",
            # missing: domain_knowledge, skills, persona, language
        }
        (mind_dir / AGENT_ID / "Buddhi" / "buddhi.jsonld").write_text(
            json.dumps(incomplete), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="missing required keys"):
            MindLoader.load_mind_file(mind_dir, AGENT_ID, "Buddhi", "buddhi.jsonld")

    def test_unknown_dimension_no_schema_passes_through(self, tmp_path: Path) -> None:
        """Files under dimensions with no registered schema load without validation."""
        mind_dir = _make_mind_dir(tmp_path)
        custom_file = mind_dir / AGENT_ID / "CustomDimension" / "data.jsonld"
        custom_file.parent.mkdir(parents=True, exist_ok=True)
        custom_file.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        doc = MindLoader.load_mind_file(
            mind_dir, AGENT_ID, "CustomDimension", "data.jsonld"
        )
        assert doc == {"key": "value"}

    def test_multiple_agents_same_mind_dir(self, tmp_path: Path) -> None:
        """MindLoader can load files for multiple agents under the same mind_dir."""
        mind_dir = tmp_path / "mind"
        for agent_id in ("agent-a", "agent-b"):
            _make_mind_dir(tmp_path, agent_id)  # each call populates mind_dir/<agent_id>
        for agent_id in ("agent-a", "agent-b"):
            doc = MindLoader.load_mind_file(
                mind_dir, agent_id, "Buddhi", "buddhi.jsonld"
            )
            assert doc["@type"] == "Buddhi"


# ---------------------------------------------------------------------------
# Test: load_agent_mind
# ---------------------------------------------------------------------------


class TestLoadAgentMind:
    def test_returns_all_four_dimensions(self, tmp_path: Path) -> None:
        mind_dir = _make_mind_dir(tmp_path)
        mind = MindLoader.load_agent_mind(mind_dir, AGENT_ID)
        assert set(mind.keys()) == {"Manas", "Buddhi", "Ahankara", "Chitta"}

    def test_manas_has_context_and_content(self, tmp_path: Path) -> None:
        mind_dir = _make_mind_dir(tmp_path)
        mind = MindLoader.load_agent_mind(mind_dir, AGENT_ID)
        assert "context" in mind["Manas"]
        assert "content" in mind["Manas"]

    def test_buddhi_has_domain_knowledge(self, tmp_path: Path) -> None:
        mind_dir = _make_mind_dir(tmp_path)
        mind = MindLoader.load_agent_mind(mind_dir, AGENT_ID)
        assert isinstance(mind["Buddhi"]["domain_knowledge"], list)
        assert len(mind["Buddhi"]["domain_knowledge"]) >= 1

    def test_ahankara_has_identity(self, tmp_path: Path) -> None:
        mind_dir = _make_mind_dir(tmp_path)
        mind = MindLoader.load_agent_mind(mind_dir, AGENT_ID)
        assert "identity" in mind["Ahankara"]

    def test_chitta_has_intelligence_nature(self, tmp_path: Path) -> None:
        mind_dir = _make_mind_dir(tmp_path)
        mind = MindLoader.load_agent_mind(mind_dir, AGENT_ID)
        assert "intelligence_nature" in mind["Chitta"]

    def test_missing_chitta_file_raises(self, tmp_path: Path) -> None:
        mind_dir = _make_mind_dir(tmp_path)
        (mind_dir / AGENT_ID / "Chitta" / "chitta.jsonld").unlink()
        with pytest.raises(FileNotFoundError):
            MindLoader.load_agent_mind(mind_dir, AGENT_ID)
