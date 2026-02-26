"""
Agent reasoning trace visualization.
Captures and renders the multi-agent decision process
for transparency and debugging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class StepType(Enum):
    QUERY_PARSE = "query_parse"
    SUB_QUERY_GEN = "sub_query_gen"
    RETRIEVAL = "retrieval"
    LLM_CALL = "llm_call"
    SYNTHESIS = "synthesis"
    QUALITY_CHECK = "quality_check"
    FALLBACK = "fallback"
    ERROR = "error"


@dataclass
class TraceStep:
    """A single step in the agent reasoning process."""

    step_type: StepType
    agent: str
    description: str
    input_data: str = ""
    output_data: str = ""
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "type": self.step_type.value,
            "agent": self.agent,
            "description": self.description,
            "input": self.input_data[:500],
            "output": self.output_data[:500],
            "duration_ms": round(self.duration_ms, 1),
            "metadata": self.metadata,
        }


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a query."""

    query: str
    steps: list[TraceStep] = field(default_factory=list)
    total_duration_ms: float = 0.0
    _start_time: float = field(default_factory=time.time, repr=False)

    def add_step(
        self,
        step_type: StepType,
        agent: str,
        description: str,
        input_data: str = "",
        output_data: str = "",
        duration_ms: float = 0.0,
        **metadata: object,
    ) -> TraceStep:
        step = TraceStep(
            step_type=step_type,
            agent=agent,
            description=description,
            input_data=input_data,
            output_data=output_data,
            duration_ms=duration_ms,
            metadata=dict(metadata),
        )
        self.steps.append(step)
        return step

    def finish(self) -> None:
        self.total_duration_ms = (time.time() - self._start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "step_count": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_markdown(self) -> str:
        lines = [
            "## 🔍 Reasoning Trace",
            f"Query: {self.query}",
            f"Steps: {len(self.steps)} | Duration: {self.total_duration_ms:.0f}ms",
            "",
        ]

        for i, step in enumerate(self.steps, 1):
            icon = _step_icon(step.step_type)
            lines.append(f"### {icon} Step {i}: {step.description}")
            lines.append(f"Agent: `{step.agent}` | Type: `{step.step_type.value}` | {step.duration_ms:.0f}ms")
            if step.input_data:
                lines.append(f"Input: {step.input_data[:200]}")
            if step.output_data:
                lines.append(f"Output: {step.output_data[:200]}")
            lines.append("")

        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Generate a Mermaid flowchart of the reasoning process."""
        lines = ["graph TD"]

        for i, step in enumerate(self.steps):
            node_id = f"S{i}"
            label = f"{step.agent}: {step.description}"
            shape = _mermaid_shape(step.step_type)
            lines.append(f'    {node_id}{shape[0]}"{label}"{shape[1]}')

            if i > 0:
                prev_id = f"S{i - 1}"
                edge_label = f"{step.duration_ms:.0f}ms"
                lines.append(f"    {prev_id} -->|{edge_label}| {node_id}")

        # Style by type
        for i, step in enumerate(self.steps):
            color = _step_color(step.step_type)
            lines.append(f"    style S{i} fill:{color}")

        return "\n".join(lines)

    @property
    def agent_summary(self) -> dict[str, float]:
        """Total time spent per agent."""
        summary: dict[str, float] = {}
        for step in self.steps:
            summary[step.agent] = summary.get(step.agent, 0) + step.duration_ms
        return summary


def _step_icon(step_type: StepType) -> str:
    icons = {
        StepType.QUERY_PARSE: "📝",
        StepType.SUB_QUERY_GEN: "🔀",
        StepType.RETRIEVAL: "🔎",
        StepType.LLM_CALL: "🤖",
        StepType.SYNTHESIS: "🧩",
        StepType.QUALITY_CHECK: "✅",
        StepType.FALLBACK: "⚠️",
        StepType.ERROR: "❌",
    }
    return icons.get(step_type, "▶️")


def _mermaid_shape(step_type: StepType) -> tuple[str, str]:
    if step_type == StepType.ERROR:
        return ("{{", "}}")
    if step_type in (StepType.RETRIEVAL, StepType.LLM_CALL):
        return ("([", "])")
    if step_type == StepType.SYNTHESIS:
        return ("[[", "]]")
    return ("[", "]")


def _step_color(step_type: StepType) -> str:
    colors = {
        StepType.QUERY_PARSE: "#e3f2fd",
        StepType.SUB_QUERY_GEN: "#f3e5f5",
        StepType.RETRIEVAL: "#e8f5e9",
        StepType.LLM_CALL: "#fff3e0",
        StepType.SYNTHESIS: "#e0f7fa",
        StepType.QUALITY_CHECK: "#f1f8e9",
        StepType.FALLBACK: "#fff8e1",
        StepType.ERROR: "#ffebee",
    }
    return colors.get(step_type, "#fafafa")
