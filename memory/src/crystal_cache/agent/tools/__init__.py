"""Tool implementations — agent/tools/.

Each module registers its tools via @register_tool at import time.
The tool_registry singleton holds all registrations; the agent
loop and the cognition worker dispatcher both consume from it
(filtered by context).

Modules:
- retrievers.py: content_search, knowledge_search,
  navigation_search, depth_search (the four V3 routers per D-A3)
- memory.py: mem0_recall, mem0_write, crystal_recall,
  crystal_write (the two-memory split per D-A5)
- llm.py: llm_invoke
- cognition.py: cognition_run (the agent's entry point to the
  multi-step cognition workflow per D-A6)
- external.py: web_search, document_upload, decompose

Import order doesn't matter — each module registers independently.
The `agent.tool_registry.import_all_tools()` function provides a
single import point that pulls all five modules in.
"""
