# Boardroom Mind

The `boardroom/mind/` directory holds the **initial mind** of the Business Infinity boardroom — both individual members and the collective. It is the first-principles knowledge and memory substrate that is hydrated into each agent at initialisation and thereafter maintained by the agent itself.

> **Manas** (Sanskrit: मनस्) — memory; the working, evolving mind.  
> **Buddhi** (Sanskrit: बुद्धि) — intellect; the stable, discriminating intelligence.  
> **Ahankara** (Sanskrit: अहंकार) — identity/ego; the sense of self that gives the intellect its contextual axis.  
> **Chitta** (Sanskrit: चित्त) — pure intelligence; mind without memory, cosmic intelligence that connects to the basis of creation.

---

## Architecture

Each of the eight boardroom members has a subdirectory under `boardroom/mind/{agent_id}/`:

```
boardroom/mind/
├── {agent_id}/
│   ├── Manas/                        # Memory — the agent's JSON-LD state file
│   │   ├── {agent_id}.jsonld         # Full agent state (context + content layers)
│   │   ├── context/                  # Immutable perspective on each entity
│   │   │   ├── company.jsonld        # Agent's fixed knowledge of ASI Saga
│   │   │   └── business-infinity.jsonld  # Agent's fixed knowledge of the product
│   │   └── content/                  # Mutable perspective on each entity
│   │       ├── company.jsonld        # Agent's current signals about ASI Saga
│   │       └── business-infinity.jsonld  # Agent's current signals about the product
│   ├── Buddhi/                       # Intellect — legend-derived domain layer
│   │   ├── buddhi.jsonld             # Legend's domain_knowledge, skills, persona, language
│   │   └── action-plan.jsonld        # Agent's action plan toward the initial company purpose
│   ├── Ahankara/                     # Identity — the ego that constrains the intellect
│   │   └── ahankara.jsonld           # Identity, contextual axis, non-negotiables, intellect constraint
│   └── Chitta/                       # Pure intelligence — mind without memory
│       └── chitta.jsonld             # Universal principles, cosmic connection, beyond identity
└── collective/                       # Shared boardroom mind (no individual owner)
    ├── boardroom.jsonld              # Collective consciousness, resonance ledger, directives
    ├── company.jsonld                # ASI Saga entity — full enriched manifest
    ├── business-infinity.jsonld      # Business Infinity product — @graph JSON-LD (5 records)
    ├── environment.jsonld            # Infrastructure manifest (Azure / GitHub)
    ├── mvp.jsonld                    # MVP phase and milestone records — @graph JSON-LD
    └── orchestration.jsonld          # Orchestration session configuration
```

### Manas layer (memory)

The Manas file (`{agent_id}.jsonld`) is the live state of the agent. It has two layers:

| Layer | Mutability | Purpose |
|-------|------------|---------|
| `context` | Immutable | Identity, mandate, domain knowledge, skills, persona, language — the agent's constitution |
| `content` | Mutable | Active focus, working memory, spontaneous intent, per-entity perspective state |

The `context/` and `content/` subdirectories hold the agent's perspective on each shared entity (ASI Saga and Business Infinity) in the same two-layer split.

### Buddhi layer (intellect)

`buddhi.jsonld` encodes the legend's domain wisdom as a standalone intellect document. It is the seed used to hydrate the agent and is loaded independently by `BoardroomStateManager.load_agent_buddhi(agent_id)`.

`action-plan.jsonld` captures the agent's action steps toward the initial purpose of the company: development of the Business Infinity MVP. Each step is expressed from the legend's own perspective, persona, and language.

### Ahankara layer (identity)

`ahankara.jsonld` encodes the agent's fundamental sense of identity — the ego that determines the axis along which the Buddhi (intellect) can function. The intellect is always constrained by the identity it serves; Ahankara is what gives that constraint its shape.

Loaded by `BoardroomStateManager.load_agent_ahankara(agent_id)`.

| Field | Description |
|-------|-------------|
| `identity` | The core self-concept — who this legend fundamentally is |
| `contextual_axis` | The axis along which all reasoning flows |
| `non_negotiables` | Identity commitments that cannot be violated without destroying the self |
| `identity_markers` | How this identity is expressed and recognized |
| `intellect_constraint` | How this Ahankara shapes and constrains the Buddhi |

### Chitta layer (pure intelligence)

`chitta.jsonld` encodes the agent's connection to pure, memory-free intelligence — the cosmic dimension that transcends both identity (Ahankara) and the memory-bound intellect (Buddhi). Chitta connects the agent to the basis of creation, to something that simply functions without needing to recall.

Loaded by `BoardroomStateManager.load_agent_chitta(agent_id)`.

| Field | Description |
|-------|-------------|
| `pure_intelligence` | Universal principles accessed without memory — what this legend discovered |
| `cosmic_connection` | How this presence connects to universal intelligence |
| `beyond_identity` | What transcends the Ahankara — the awareness before the persona |
| `consciousness_basis` | Connection to the basis of creation/consciousness within |

---

## The Four Dimensions

```
Chitta  ──  Pure Intelligence   (cosmic, memory-free, connects to consciousness)
   │
Ahankara ── Identity/Ego        (constrains the intellect to its contextual axis)
   │
Buddhi  ──  Intellect           (domain wisdom, skills, discriminating intelligence)
   │
Manas   ──  Memory              (working state, active focus, evolving perspective)
```

The dimensions flow from subtle to gross. Chitta is the deepest — pure intelligence without memory, the cosmic substrate. Ahankara gives that intelligence a specific identity, constraining it to a contextual axis. Buddhi is the intellect that operates within that axis. Manas is the memory — the evolving, active layer that captures moment-to-moment experience.

---

## Agent Roster

| Agent | Legend | Domain | `@id` |
|-------|--------|--------|-------|
| CEO | Steve Jobs | Vision & Strategy | `agent:ceo` |
| CFO | Warren Buffett | Finance & Resources | `agent:cfo` |
| COO | W. Edwards Deming | Operations & Workflow | `agent:coo` |
| CMO | Seth Godin | Remarkability / Tribe Building | `agent:sg_cmo` |
| CHRO | Peter Drucker | People & Culture | `agent:chro` |
| CTO | Alan Turing | Technology & Innovation | `agent:sj_cto` |
| CSO | Sun Tzu | Strategy & Competitive Intelligence | `agent:cso_strategy` |
| Founder | Paul Graham | Prioritization / Survival / Shipping | `agent:pg_founder` |

> **CTO `@id` note**: `agent:sj_cto` uses the initials `sj` as a historical artifact from when Steve Jobs held the CTO role in early prototypes. The identifier is preserved for backward compatibility with governance references. `context.name` is the authoritative field — it reads `Alan Turing`.

---

## Collective Mind

The `collective/` subdirectory holds the shared boardroom consciousness — state that belongs to the whole rather than to any individual agent. All files use the `.jsonld` format; multi-record documents use JSON-LD `@graph`.

| File | Format | Description |
|------|--------|-------------|
| `boardroom.jsonld` | object | Active session, resonance ledger, composite score, and directives |
| `company.jsonld` | object | ASI Saga entity manifest with full context and content enrichment |
| `business-infinity.jsonld` | `@graph` | Business Infinity product records (5 records) |
| `environment.jsonld` | object | Azure / GitHub infrastructure manifest |
| `mvp.jsonld` | `@graph` | MVP phase, configuration, and milestone records (5 records) |
| `orchestration.jsonld` | object | Orchestration session configuration and resonance protocol |

---

## Initial Purpose

The initial purpose of **ASI Saga** is the development of the MVP of **Business Infinity** — an autonomous C-suite boardroom that governs business decisions through purpose-driven debate, resonance scoring, and perpetual orchestration. Every action-plan in every agent's Buddhi is anchored to this purpose.

---

## Schemas

JSON Schema files for each mind file type live in `boardroom/mind/schemas/`:

| Schema file | File type validated |
|-------------|---------------------|
| `manas.schema.json` | `Manas/{agent_id}.jsonld` — agent state (context + content) |
| `buddhi.schema.json` | `Buddhi/buddhi.jsonld` — intellect document |
| `action-plan.schema.json` | `Buddhi/action-plan.jsonld` — action plan |
| `ahankara.schema.json` | `Ahankara/ahankara.jsonld` — identity document |
| `chitta.schema.json` | `Chitta/chitta.jsonld` — pure intelligence document |
| `entity-context.schema.json` | `Manas/context/{entity}.jsonld` — immutable entity perspective |
| `entity-content.schema.json` | `Manas/content/{entity}.jsonld` — mutable entity perspective |

`BoardroomStateManager` uses these schemas to validate mind files when they are loaded via:

- `load_mind_file(agent_id, dimension, filename)` — load and validate any single mind file
- `load_agent_mind(agent_id)` — load all four dimensions (Manas, Buddhi, Ahankara, Chitta)
- `get_schemas_dir()` — return the path to the schemas directory

---

## Per-Agent Readme Files

Each agent directory contains a `Readme.md` documenting its specific files:

| Agent | File |
|-------|------|
| CEO (Steve Jobs) | `boardroom/mind/ceo/Readme.md` |
| CFO (Warren Buffett) | `boardroom/mind/cfo/Readme.md` |
| COO (W. Edwards Deming) | `boardroom/mind/coo/Readme.md` |
| CMO (Seth Godin) | `boardroom/mind/cmo/Readme.md` |
| CHRO (Peter Drucker) | `boardroom/mind/chro/Readme.md` |
| CTO (Alan Turing) | `boardroom/mind/cto/Readme.md` |
| CSO (Sun Tzu) | `boardroom/mind/cso/Readme.md` |
| Founder (Paul Graham) | `boardroom/mind/founder/Readme.md` |

---

## References

→ **Agent spec**: `.github/specs/boardroom-agents.md` — legend archetypes, schemas, validation  
→ **Entity spec**: `.github/specs/boardroom-entities.md` — company and product enrichment  
→ **State manager**: `src/business_infinity/boardroom.py` → `BoardroomStateManager`  
→ **Skill (roster)**: `.github/skills/boardroom-agent-state/SKILL.md`  
→ **Schemas**: `boardroom/mind/schemas/` — JSON Schema files for each dimension  
→ **Repository spec**: `.github/specs/repository.md`


