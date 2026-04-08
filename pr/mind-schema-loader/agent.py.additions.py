"""
Additions for src/purpose_driven_agent/agent.py
================================================

This file contains the exact code fragments to add to ``PurposeDrivenAgent``
in ``agent.py``.  It is **not** a standalone module — copy each section into
the correct position in agent.py as described by the inline comments.

PR: feat/mind-schema-loader — schema-based mind file loading for PurposeDrivenAgent
"""
from __future__ import annotations

# ── STEP 1: New import ────────────────────────────────────────────────────────
# Add these lines at the top of agent.py alongside the other local imports
# (e.g. after "from purpose_driven_agent.ml_interface import ...").
#
from pathlib import Path
from purpose_driven_agent.mind_loader import MindLoader


# ── STEP 2: New __init__ parameter ───────────────────────────────────────────
# In PurposeDrivenAgent.__init__(...), add this parameter to the signature:
#
#     mind_dir: Optional[Path] = None,
#
# Full updated signature (add the new parameter after ``ml_service``):
#
#   def __init__(
#       self,
#       agent_id: str,
#       purpose: str,
#       ...
#       ml_service: Optional[IMLService] = None,
#       mind_dir: Optional[Path] = None,   # ← ADD THIS
#   ) -> None:
#
# Then, in the body of __init__, add the following line after the
# ``self.ml_service = ...`` assignment:
#
#   #: Optional path to the agent's mind directory.
#   #: Set to a :class:`~pathlib.Path` to enable mind file loading via
#   #: :meth:`load_mind_file` and :meth:`load_agent_mind`.
#   self.mind_dir: Optional[Path] = mind_dir


# ── STEP 3: New instance methods ─────────────────────────────────────────────
# Copy each of the four methods below verbatim into the PurposeDrivenAgent
# class body.  A good place is after the ``get_metadata`` method, before the
# private lifecycle helpers (``_perpetual_loop``, ``_awaken``, etc.).
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  IMPORTANT: do NOT copy the _MindLoaderMethodsMixin class declaration.  ║
# ║  Copy only the def bodies below and paste them into PurposeDrivenAgent. ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_mind_dir(self) -> Optional[Path]:
    """Return the agent's mind directory, or ``None`` if not configured.

    The mind directory is set via the ``mind_dir`` constructor parameter.
    Subclasses (e.g. boardroom agents) typically receive this path from
    their repository layout, pointing to a directory with the structure::

        <mind_dir>/
        ├── schemas/
        └── <agent_id>/
            ├── Manas/
            ├── Buddhi/
            ├── Ahankara/
            └── Chitta/
    """
    return self.mind_dir


def get_schemas_dir(self) -> Optional[Path]:
    """Return the schemas directory inside the agent's mind directory.

    Returns ``None`` if :attr:`mind_dir` is not configured.

    The schemas directory holds the authoritative JSON Schema files for
    each mind file type (``manas.schema.json``, ``buddhi.schema.json``,
    etc.).  These are informative — runtime validation uses the in-code
    ``MindLoader.MIND_FILE_SCHEMAS`` registry.
    """
    if self.mind_dir is None:
        return None
    return MindLoader.get_schemas_dir(self.mind_dir)


def load_mind_file(self, dimension: str, filename: str) -> dict:
    """Load and schema-validate a single mind file for this agent.

    Delegates to :meth:`~purpose_driven_agent.mind_loader.MindLoader.load_mind_file`
    using ``self.mind_dir`` and ``self.agent_id``.

    Args:
        dimension: Mind dimension name (``"Buddhi"``, ``"Ahankara"``,
            ``"Chitta"``, ``"Manas"``, ``"Manas/context"``,
            ``"Manas/content"``).
        filename: JSON-LD filename inside the dimension directory.

    Returns:
        Parsed and schema-validated document as a :class:`dict`.

    Raises:
        :class:`RuntimeError` if :attr:`mind_dir` is not configured.
        :class:`FileNotFoundError` if the file is absent.
        :class:`ValueError` if the file fails required-key validation.
    """
    if self.mind_dir is None:
        raise RuntimeError(
            f"Agent '{self.agent_id}' has no mind_dir configured. "
            "Pass mind_dir=<Path> to the constructor."
        )
    return MindLoader.load_mind_file(self.mind_dir, self.agent_id, dimension, filename)


def load_agent_mind(self) -> dict:
    """Load all four mind dimensions for this agent.

    Delegates to :meth:`~purpose_driven_agent.mind_loader.MindLoader.load_agent_mind`
    using ``self.mind_dir`` and ``self.agent_id``.

    Returns a dict with keys ``"Manas"``, ``"Buddhi"``, ``"Ahankara"``,
    and ``"Chitta"``, each containing the parsed and validated document.

    Raises:
        :class:`RuntimeError` if :attr:`mind_dir` is not configured.
        :class:`FileNotFoundError` if any dimension file is absent.
        :class:`ValueError` if any file fails required-key validation.
    """
    if self.mind_dir is None:
        raise RuntimeError(
            f"Agent '{self.agent_id}' has no mind_dir configured. "
            "Pass mind_dir=<Path> to the constructor."
        )
    return MindLoader.load_agent_mind(self.mind_dir, self.agent_id)
