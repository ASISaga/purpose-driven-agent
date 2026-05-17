# Documentation — purpose-driven-agent

Navigation index for all documentation in this directory.

---

## Guides

| Document | Description |
|---|---|
| [Architecture](architecture.md) | AOS container hierarchy, component design, FAS hosting, routing tag protocol |
| [API Reference](api-reference.md) | Complete class and method reference for all public APIs |
| [Contributing](contributing.md) | Development setup, testing, linting, and pull-request guidelines |

---

## Quick Links

- **Start here:** [README.md](../README.md) — overview, installation, quick-start examples
- **Container layout:** [architecture.md § AOS Container Hierarchy](architecture.md#aos-container-hierarchy)
- **FAS hosting:** [architecture.md § FAS Hosting Adapter](architecture.md#fas-hosting-adapter)
- **Routing tags:** [architecture.md § Routing Tag Protocol](architecture.md#routing-tag-protocol)
- **All classes:** [api-reference.md](api-reference.md)
- **New module — `RoutingMixin`:** [api-reference.md § routing_mixin](api-reference.md#module-purpose_driven_agentrouting_mixin)
- **New module — `hosting`:** [api-reference.md § hosting](api-reference.md#module-purpose_driven_agenthosting)
- **`RoutingClassifier`:** [api-reference.md § aos_mcp_servers.routing](api-reference.md#module-aos_mcp_serversrouting)

---

## Source Layout

```
src/
├── purpose_driven_agent/
│   ├── __init__.py          # Public API exports
│   ├── __main__.py          # python -m purpose_driven_agent
│   ├── agent.py             # PurposeDrivenAgent + GenericPurposeDrivenAgent
│   ├── context_provider.py  # ContextProvider hierarchy
│   ├── context_server.py    # ContextMCPServer
│   ├── hosting.py           # FAS hosting adapter
│   ├── ml_interface.py      # IMLService
│   └── routing_mixin.py     # RoutingMixin
└── aos_mcp_servers/
    └── routing.py           # MCP transports + RoutingClassifier
```
