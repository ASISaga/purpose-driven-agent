# PR: feat/mind-schema-loader — Schema-based mind file loading for PurposeDrivenAgent

## Overview

This PR adds a schema-based mind file loading mechanism to the
`purpose-driven-agent` library so that `PurposeDrivenAgent` subclasses
(including boardroom agents that inherit from it) can load, validate, and use
structured JSON-LD mind files from their respective mind directories.

The feature was originally prototyped in
[ASISaga/business-infinity → `BoardroomStateManager`](https://github.com/ASISaga/business-infinity)
and is being moved here so that the canonical loading mechanism lives in the
base `PurposeDrivenAgent`, not in a downstream boardroom-specific class.

---

## Background: the Four-Dimension Mind Model

Each purpose-driven agent may have a **mind directory** with four Sanskrit-named
dimension subdirectories:

| Dimension | Role | Canonical file |
|---|---|---|
| **Manas** (memory) | Live agent state — immutable context + mutable content | `Manas/{agent_id}.jsonld` |
| **Buddhi** (intellect) | Legend-derived domain knowledge, skills, persona, language | `Buddhi/buddhi.jsonld` |
| **Ahankara** (identity) | Ego/self-concept constraining the intellect to its domain | `Ahankara/ahankara.jsonld` |
| **Chitta** (pure intelligence) | Mind without memory — cosmic substrate beyond identity | `Chitta/chitta.jsonld` |

A `schemas/` directory alongside the agent subdirectories holds the
authoritative JSON Schema files for tooling and documentation.

---

## Files to Copy into the Repository

### `src/purpose_driven_agent/mind_loader.py` → `src/purpose_driven_agent/`

A standalone, side-effect-free `MindLoader` class with:

| Item | Description |
|---|---|
| `MIND_FILE_SCHEMAS` | Schema registry — 7 keys mapping dimension/filename to required keys + schema file reference |
| `load_mind_file(mind_dir, agent_id, dimension, filename)` | Load + schema-validate any single mind file |
| `load_agent_mind(mind_dir, agent_id)` | Load all four canonical dimension files at once |
| `get_schemas_dir(mind_dir)` | Return `mind_dir / "schemas"` |
| `list_registered_dimensions()` | Return all registered schema keys (introspection) |

`MindLoader` is intentionally decoupled from any agent registry.  The caller
supplies `mind_dir` and `agent_id`; the loader resolves paths, loads JSON-LD,
validates required keys, and returns parsed dictionaries.

### `tests/test_mind_loader.py` → `tests/`

Self-contained pytest test suite (no business-infinity dependencies).  Uses
`tmp_path` to create minimal but schema-valid mind directories and exercises:

- Schema registry completeness
- `_resolve_schema_key` for all dimension/filename combinations
- `load_mind_file` for every dimension type, including error paths
- `load_agent_mind` for all four dimensions
- Multiple agents under the same `mind_dir`

---

## Changes to Existing Files

### `src/purpose_driven_agent/agent.py`

See [`agent.py.additions.py`](./agent.py.additions.py) for the exact code
fragments.  The changes are:

**1. New import** (add after existing local imports):

```python
from pathlib import Path  # only if not already imported
from purpose_driven_agent.mind_loader import MindLoader
```

**2. New `__init__` parameter** (add after `ml_service`):

```python
def __init__(
    self,
    ...
    ml_service: Optional[IMLService] = None,
    mind_dir: Optional[Path] = None,   # ← NEW
) -> None:
```

And in the body:

```python
#: Optional path to the agent's mind directory.
self.mind_dir: Optional[Path] = mind_dir
```

**3. Four new instance methods** (add after `get_metadata`, before `_perpetual_loop`):

- `get_mind_dir()` → return `self.mind_dir`
- `get_schemas_dir()` → delegate to `MindLoader.get_schemas_dir`
- `load_mind_file(dimension, filename)` → delegate to `MindLoader.load_mind_file`
- `load_agent_mind()` → delegate to `MindLoader.load_agent_mind`

---

## Typical Usage (after the PR is merged)

### Direct usage in a subclass

```python
from pathlib import Path
from purpose_driven_agent import GenericPurposeDrivenAgent

class BoardroomCEOAgent(GenericPurposeDrivenAgent):
    def __init__(self, mind_dir: Path, **kwargs):
        super().__init__(
            agent_id="ceo",
            purpose="Vision & Strategy: architect the future of ASI Saga",
            adapter_name="ceo",
            mind_dir=mind_dir,
        )

    async def initialize(self) -> bool:
        result = await super().initialize()
        mind = self.load_agent_mind()
        self.buddhi   = mind["Buddhi"]   # domain knowledge, skills, persona
        self.ahankara = mind["Ahankara"] # identity, contextual axis
        self.chitta   = mind["Chitta"]   # pure intelligence, cosmic substrate
        self.manas    = mind["Manas"]    # live state: context + content layers
        return result
```

### Load a single file

```python
action_plan = agent.load_mind_file("Buddhi", "action-plan.jsonld")
company_ctx = agent.load_mind_file("Manas/context", "company.jsonld")
```

### Use MindLoader directly (without a PurposeDrivenAgent instance)

```python
from pathlib import Path
from purpose_driven_agent.mind_loader import MindLoader

mind_dir = Path("boardroom/mind")
buddhi = MindLoader.load_mind_file(mind_dir, "ceo", "Buddhi", "buddhi.jsonld")
mind   = MindLoader.load_agent_mind(mind_dir, "cto")
```

---

## How to Apply This PR

1. Copy `src/purpose_driven_agent/mind_loader.py` → `src/purpose_driven_agent/`
2. Copy `tests/test_mind_loader.py` → `tests/`
3. Apply the three changes in `agent.py.additions.py` to `src/purpose_driven_agent/agent.py`
4. Run tests: `pytest tests/test_mind_loader.py -v`

---

## Relationship to business-infinity

After this PR lands in `purpose-driven-agent`:

- `BoardroomStateManager` in `business-infinity` retains the schema registry
  and the `load_mind_file` / `load_agent_mind` methods that are tested against
  the actual boardroom mind files in `boardroom/mind/`.
- Boardroom agents, once they inherit from `PurposeDrivenAgent`, will be able to
  call `self.load_agent_mind()` directly, with `mind_dir` pointing to their
  respective repository's `boardroom/mind/` directory.
- The `MindLoader` class is the shared building block — `BoardroomStateManager`
  can delegate to it and re-use its schema registry and required-key validation
  instead of duplicating them.

---

## References

→ **Source (business-infinity)**: `src/business_infinity/boardroom.py` — `BoardroomStateManager._MIND_FILE_SCHEMAS`, `load_mind_file`, `load_agent_mind`  
→ **Tests (business-infinity)**: `tests/test_workflows.py` — `TestMindFileSchemas`  
→ **Mind file schemas**: `boardroom/mind/schemas/` in business-infinity  
→ **Boardroom agents spec**: `.github/specs/boardroom-agents.md` in business-infinity  
