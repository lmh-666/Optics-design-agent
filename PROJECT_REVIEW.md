# Project Review For Job Applications

## Strict Assessment

This project is strong enough to be used as a job-search portfolio project, especially for roles involving AI applications, RAG, backend systems, scientific computing, or domain-specific agents. It has a real domain, non-trivial data, structured constraints, retrieval, ranking, evaluation, and a usable API surface.

However, under a strict large-tech-company standard, it should not be oversold as a general autonomous multi-agent platform. The implementation is best described as a domain workflow agent with both a custom planner-executor path and a LangGraph state-graph path:

- stateful multi-turn interaction
- intent classification
- deterministic planning
- tool execution
- retrieval and ranking
- recommendation explanation
- LangGraph nodes and conditional routing

It now includes graph-based orchestration through LangGraph, but it still does not include durable distributed state, async job queues, production observability, formal evaluation metrics, or production-grade deployment controls.

## Best Positioning

Use this title:

> Domain Workflow Agent and Hybrid Retrieval System for Optical Lens Design

Avoid this title:

> Autonomous Multi-Agent Optical Design Platform

The first title is defensible. The second one invites hard questions about LangGraph, tool autonomy, distributed state, evaluation, safety, and production reliability.

## What To Improve Before GitHub

Priority 1:

- Add README, dependencies, `.env.example`, `.gitignore`.
- Fix visible Chinese encoding issues in comments and demo inputs.
- Remove generated caches and notebook checkpoints from version control.
- Decide whether the full 2456-row data file can be public. If not, publish a small sample dataset.

Priority 2:

- Add at least 3 smoke tests for `/parse_requirement`, `/design_assist`, and `/agent/chat`.
- Add example JSON responses under `examples/`.
- Add screenshots of the front end and generated lens layout.
- Add a lightweight architecture diagram.

Priority 3:

- Replace or augment the rule intent classifier with an LLM-based structured classifier.
- Move session state from process memory to Redis or SQLite/PostgreSQL.
- Add structured logging, request IDs, latency metrics, and error taxonomy.
- Add Dockerfile and GitHub Actions.

## Interview Defense

If asked why both custom workflow and LangGraph are present:

> The project started as a domain-constrained optical design assistant. I first built a lightweight planner-executor-tool-registry architecture because the workflow is mostly deterministic: parse requirement, retrieve candidates, apply hard constraints, rescale aperture/focal length, rerank, run ray tracing, and explain. After validating the workflow, I migrated the orchestration layer to LangGraph so that the same stages are explicit graph nodes with conditional routing. The custom path is useful for debugging and baseline comparison; the LangGraph path is better for standard workflow orchestration and future extension.

If asked what makes it RAG:

> The retrieval source is a structured optical lens database instead of unstructured documents. The system retrieves candidate lens structures based on parsed requirements, then augments the recommendation with domain rules, numerical similarity, hard constraints, ray tracing signals, and explanation. It is closer to structured RAG / retrieval-augmented recommendation than document QA.

If asked what the main technical risk is:

> The biggest risk is evaluation. Optical design quality cannot be fully proven by text parsing and approximate ray spread alone. I handled obvious hard constraints and used ray tracing as a feasibility signal, but production use would require stronger optical metrics, benchmark cases, and comparison with expert-designed baselines.
