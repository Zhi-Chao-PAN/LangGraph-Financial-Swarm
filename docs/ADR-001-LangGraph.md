# ADR 001: Selection of LangGraph for Hierarchical Agent Swarm

## Context
Financial document analysis requires coordinating multiple distinct capabilities:
1. **Precise Information Retrieval** (RAG from PDFs)
2. **Data Analysis & Visualization** (Python Code Execution)
3. **Strategic Oversight** (Planning & Validation)

A flat multi-agent system (e.g., AutoGen's default chat) often leads to circular discussions or loss of focus. We needed a framework that supports **Directed Cyclic Graphs (DCGs)** to enforce business logic (e.g., "Always research before plotting").

## Decision
We selected **LangGraph** (built on LangChain) as the orchestration engine.

## Consequences

### Positive
*   **State Control**: The `AgentState` TypedDict provides a strictly typed, immutable audit log of the entire conversation.
*   **Cyclic Capability**: Allows for "Feedback Loops" (e.g., Supervisor rejects a plot --> Quant retries) which are impossible in linear chains.
*   **Human-in-the-Loop**: LangGraph's architecture natively supports interrupting execution for human approval (critical for future financial compliance).

### Negative
*   **Complexity**: Higher learning curve compared to simple "Chain of Thought".
*   **Verbose**: Requires explicit edge definitions.

## Compliance
This decision aligns with the "Robustness" and "Explainability" goals of the Master's project.
