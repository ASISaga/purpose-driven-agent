"""
MindLoader — Schema-based file loader for the purpose-driven agent mind.

Each ``PurposeDrivenAgent`` may have a *mind directory* on disk that holds
JSON-LD files structured across four Sanskrit-named dimensions:

- **Manas** (memory) — live agent state with immutable context and mutable
  content layers.
- **Buddhi** (intellect) — legend-derived domain knowledge, skills, persona,
  and language.
- **Ahankara** (identity) — the ego/self-concept that constrains the intellect
  to its domain axis.
- **Chitta** (pure intelligence) — mind without memory; the cosmic substrate
  that transcends identity.

``MindLoader`` is a standalone, side-effect-free class.  It is intentionally
decoupled from any agent registry.  Callers supply the ``mind_dir`` path and
``agent_id`` (the name of the agent's subdirectory) — the loader resolves
paths, loads JSON-LD documents, validates required keys, and returns the
parsed dictionaries.

Typical usage inside a ``PurposeDrivenAgent`` subclass::

    from pathlib import Path
    from purpose_driven_agent.mind_loader import MindLoader

    class BoardroomCEOAgent(PurposeDrivenAgent):
        def __init__(self, mind_dir: Path, **kwargs):
            super().__init__(agent_id="ceo", **kwargs)
            self._mind_dir = mind_dir

        async def initialize(self) -> bool:
            result = await super().initialize()
            mind = MindLoader.load_agent_mind(self._mind_dir, self.agent_id)
            self.buddhi   = mind["Buddhi"]
            self.ahankara = mind["Ahankara"]
            self.chitta   = mind["Chitta"]
            self.manas    = mind["Manas"]
            return result

Mind directory layout expected by ``MindLoader``::

    <mind_dir>/
    ├── schemas/                        # JSON Schema definitions (informative)
    │   ├── manas.schema.json
    │   ├── buddhi.schema.json
    │   ├── action-plan.schema.json
    │   ├── ahankara.schema.json
    │   ├── chitta.schema.json
    │   ├── entity-context.schema.json
    │   └── entity-content.schema.json
    └── <agent_id>/
        ├── Manas/
        │   ├── <agent_id>.jsonld       # Full state (context + content layers)
        │   ├── context/
        │   │   ├── company.jsonld
        │   │   └── *.jsonld
        │   └── content/
        │       ├── company.jsonld
        │       └── *.jsonld
        ├── Buddhi/
        │   ├── buddhi.jsonld
        │   └── action-plan.jsonld
        ├── Ahankara/
        │   └── ahankara.jsonld
        └── Chitta/
            └── chitta.jsonld
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# MindLoader
# ---------------------------------------------------------------------------


class MindLoader:
    """Schema-based loader for purpose-driven agent mind files.

    All methods are class methods — there is no instance state.  The caller
    always provides a ``mind_dir`` :class:`~pathlib.Path` which is the root
    directory containing per-agent subdirectories (one per ``agent_id``).

    The loader validates documents against a lightweight in-code schema
    (required-key set).  The authoritative JSON Schema files that live under
    ``<mind_dir>/schemas/`` are provided for tooling and documentation; they
    are not used for runtime validation.
    """

    # ── Schema registry ──────────────────────────────────────────────────────
    # Maps (dimension, filename) key → required keys + schema file reference.
    #
    # Key format: "{Dimension}/{filename}"  e.g. "Buddhi/buddhi.jsonld"
    # Special keys:
    #   "Manas/state"          → {agent_id}.jsonld  (varies per agent)
    #   "Manas/context/entity" → any file in Manas/context/
    #   "Manas/content/entity" → any file in Manas/content/

    MIND_FILE_SCHEMAS: Dict[str, Dict[str, Any]] = {
        "Manas/state": {
            "description": "Agent memory state — context and content layers",
            "schema_file": "manas.schema.json",
            "required_keys": {
                "@context",
                "@id",
                "@type",
                "schema_version",
                "context",
                "context_management",
                "content",
                "content_management",
            },
        },
        "Buddhi/buddhi.jsonld": {
            "description": "Agent intellect — legend-derived domain wisdom",
            "schema_file": "buddhi.schema.json",
            "required_keys": {
                "@context",
                "@id",
                "@type",
                "schema_version",
                "agent_id",
                "name",
                "domain",
                "domain_knowledge",
                "skills",
                "persona",
                "language",
            },
        },
        "Buddhi/action-plan.jsonld": {
            "description": "Agent action plan — steps toward the initial company purpose",
            "schema_file": "action-plan.schema.json",
            "required_keys": {
                "@context",
                "@type",
                "name",
                "role",
                "anchor",
                "status",
                "overarchingPurpose",
                "actionSteps",
            },
        },
        "Ahankara/ahankara.jsonld": {
            "description": "Agent identity — the ego that constrains the intellect",
            "schema_file": "ahankara.schema.json",
            "required_keys": {
                "@context",
                "@id",
                "@type",
                "schema_version",
                "agent_id",
                "name",
                "identity",
                "contextual_axis",
                "non_negotiables",
                "identity_markers",
                "intellect_constraint",
            },
        },
        "Chitta/chitta.jsonld": {
            "description": "Pure intelligence — mind without memory, cosmic substrate",
            "schema_file": "chitta.schema.json",
            "required_keys": {
                "@context",
                "@id",
                "@type",
                "schema_version",
                "agent_id",
                "name",
                "intelligence_nature",
                "cosmic_intelligence",
                "beyond_identity",
                "consciousness_basis",
            },
        },
        "Manas/context/entity": {
            "description": "Immutable entity perspective (context layer)",
            "schema_file": "entity-context.schema.json",
            "required_keys": {
                "@context",
                "@id",
                "@type",
                "name",
                "agent_perspective",
                "legend",
                "domain_knowledge",
                "skills",
                "persona",
                "language",
            },
        },
        "Manas/content/entity": {
            "description": "Mutable entity perspective (content layer)",
            "schema_file": "entity-content.schema.json",
            "required_keys": {
                "@context",
                "@id",
                "@type",
                "name",
                "agent_perspective",
                "legend",
                "perspective",
                "software_interfaces",
                "current_signals",
            },
        },
    }

    # ── Internal helpers ─────────────────────────────────────────────────────

    @classmethod
    def _read_text(cls, path: Path) -> str:
        """Read a file from disk and return its text content, stripped."""
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()

    @classmethod
    def _load_json_document(cls, path: Path) -> Dict[str, Any]:
        """Load a single JSON document from *path*.

        Handles both single-line JSONL and multi-line JSON formats.

        Raises :class:`ValueError` if the file cannot be parsed.
        Raises :class:`FileNotFoundError` if the file does not exist.
        """
        content = cls._read_text(path)
        first_line = content.partition("\n")[0].strip()
        try:
            return json.loads(first_line)
        except json.JSONDecodeError:
            try:
                return json.loads(content)
            except json.JSONDecodeError as full_error:
                raise ValueError(
                    f"Unable to parse JSON document at {path}: "
                    "single-line and full-content parsing both failed"
                ) from full_error

    @classmethod
    def _validate_required_keys(
        cls,
        document: Dict[str, Any],
        required_keys: Set[str],
        label: str,
    ) -> None:
        """Validate that *document* contains all *required_keys*.

        Raises :class:`ValueError` with a descriptive message listing missing
        keys if validation fails.
        """
        missing = required_keys - set(document.keys())
        if missing:
            raise ValueError(
                f"Mind file '{label}' is missing required keys: "
                + ", ".join(sorted(missing))
            )

    @classmethod
    def _resolve_schema_key(cls, dimension: str, filename: str) -> str:
        """Return the ``MIND_FILE_SCHEMAS`` key for a given dimension/filename pair.

        Special cases:
        - ``Manas`` + ``{agent_id}.jsonld`` → ``"Manas/state"``
        - ``Manas/context`` + any ``.jsonld`` → ``"Manas/context/entity"``
        - ``Manas/content`` + any ``.jsonld`` → ``"Manas/content/entity"``
        - Other dimensions use ``"{dimension}/{filename}"`` directly.
        """
        if dimension == "Manas" and filename.endswith(".jsonld") and "/" not in filename:
            return "Manas/state"
        if dimension in ("Manas/context", "Manas/content"):
            layer = "context" if dimension.endswith("context") else "content"
            return f"Manas/{layer}/entity"
        return f"{dimension}/{filename}"

    @classmethod
    def _validate_mind_file(
        cls, key: str, data: Dict[str, Any], label: str
    ) -> None:
        """Validate a mind file document against its registered schema.

        Uses the lightweight required-key validation.  If no schema is
        registered for *key*, the document is returned without validation.
        """
        schema = cls.MIND_FILE_SCHEMAS.get(key)
        if schema is None:
            return
        cls._validate_required_keys(data, schema["required_keys"], label)

    # ── Public API ───────────────────────────────────────────────────────────

    @classmethod
    def get_schemas_dir(cls, mind_dir: Path) -> Path:
        """Return the ``schemas/`` directory inside *mind_dir*.

        The schemas directory holds the authoritative JSON Schema (``.schema.json``)
        files for each mind file type.  It is informative — runtime validation
        uses the in-code ``MIND_FILE_SCHEMAS`` registry, not the JSON files.
        """
        return mind_dir / "schemas"

    @classmethod
    def load_mind_file(
        cls,
        mind_dir: Path,
        agent_id: str,
        dimension: str,
        filename: str,
    ) -> Dict[str, Any]:
        """Load and schema-validate a single mind file for an agent.

        Given a *mind_dir* root, an *agent_id* (directory name), a *dimension*
        name, and a *filename*, this method:

        1. Resolves the absolute path under ``{mind_dir}/{agent_id}/{dimension}/{filename}``.
        2. Loads the JSON-LD document.
        3. Validates it against the registered schema (required keys).
        4. Returns the validated document.

        **Supported dimension / filename combinations:**

        =============================  ===============================
        dimension                      filename
        =============================  ===============================
        ``Buddhi``                     ``buddhi.jsonld``
        ``Buddhi``                     ``action-plan.jsonld``
        ``Ahankara``                   ``ahankara.jsonld``
        ``Chitta``                     ``chitta.jsonld``
        ``Manas``                      ``{agent_id}.jsonld``
        ``Manas/context``              any ``.jsonld``
        ``Manas/content``              any ``.jsonld``
        =============================  ===============================

        Args:
            mind_dir: Root mind directory containing per-agent subdirectories.
            agent_id: Agent identifier — used as the subdirectory name.
            dimension: Mind dimension name.  For Manas sub-layers use
                ``"Manas/context"`` or ``"Manas/content"``.
            filename: JSON-LD filename inside the dimension directory.

        Returns:
            Parsed and schema-validated JSON-LD document as a :class:`dict`.

        Raises:
            :class:`FileNotFoundError` if the file is absent.
            :class:`ValueError` if the file cannot be parsed or fails
                required-key validation.
        """
        path = mind_dir / agent_id / dimension / filename
        document = cls._load_json_document(path)
        schema_key = cls._resolve_schema_key(dimension, filename)
        cls._validate_mind_file(
            schema_key, document, f"{agent_id}/{dimension}/{filename}"
        )
        return document

    @classmethod
    def load_agent_mind(
        cls,
        mind_dir: Path,
        agent_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Load all four mind dimensions for an agent.

        Loads and schema-validates the canonical file for each dimension:

        - ``Manas``    → ``{mind_dir}/{agent_id}/Manas/{agent_id}.jsonld``
        - ``Buddhi``   → ``{mind_dir}/{agent_id}/Buddhi/buddhi.jsonld``
        - ``Ahankara`` → ``{mind_dir}/{agent_id}/Ahankara/ahankara.jsonld``
        - ``Chitta``   → ``{mind_dir}/{agent_id}/Chitta/chitta.jsonld``

        Args:
            mind_dir: Root mind directory containing per-agent subdirectories.
            agent_id: Agent identifier — used as the subdirectory name.

        Returns:
            Dictionary with keys ``"Manas"``, ``"Buddhi"``, ``"Ahankara"``,
            and ``"Chitta"``, each containing the parsed and validated document.

        Raises:
            :class:`FileNotFoundError` if any canonical dimension file is absent.
            :class:`ValueError` if any file fails required-key validation.
        """
        return {
            "Manas": cls.load_mind_file(
                mind_dir, agent_id, "Manas", f"{agent_id}.jsonld"
            ),
            "Buddhi": cls.load_mind_file(
                mind_dir, agent_id, "Buddhi", "buddhi.jsonld"
            ),
            "Ahankara": cls.load_mind_file(
                mind_dir, agent_id, "Ahankara", "ahankara.jsonld"
            ),
            "Chitta": cls.load_mind_file(
                mind_dir, agent_id, "Chitta", "chitta.jsonld"
            ),
        }

    @classmethod
    def list_registered_dimensions(cls) -> List[str]:
        """Return all registered schema keys.

        Useful for introspection and documentation tooling.
        """
        return list(cls.MIND_FILE_SCHEMAS.keys())
