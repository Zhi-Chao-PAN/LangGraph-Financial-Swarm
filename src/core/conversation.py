"""
Conversation memory for multi-turn dialogue support.

Tracks conversation history, detects follow-up questions,
and provides context for the swarm agents.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    """A single conversation turn."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """A conversation session with history."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    max_turns: int = 20

    def add_turn(self, role: str, content: str, **metadata: Any) -> None:
        """Add a turn, evicting oldest if over max."""
        self.turns.append(Turn(role=role, content=content, metadata=metadata))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_context(self, last_n: int = 5) -> str:
        """Get recent conversation context as formatted string."""
        recent = self.turns[-last_n:]
        lines = []
        for t in recent:
            prefix = "User" if t.role == "user" else "Assistant"
            lines.append(f"{prefix}: {t.content}")
        return "\n".join(lines)

    def last_user_question(self) -> str | None:
        """Get the most recent user question."""
        for t in reversed(self.turns):
            if t.role == "user":
                return t.content
        return None

    def last_answer(self) -> str | None:
        """Get the most recent assistant answer."""
        for t in reversed(self.turns):
            if t.role == "assistant":
                return t.content
        return None

    @property
    def turn_count(self) -> int:
        return len(self.turns)


# Follow-up detection keywords/patterns
_FOLLOWUP_INDICATORS = [
    "what about", "how about", "and for", "also",
    "compared to", "versus", "vs", "same for",
    "what was it", "how much was it",
    "the same", "that company", "their",
    "last year", "previous year", "next year",
    "why", "explain", "elaborate", "more detail",
    "break it down", "break down",
]

_PRONOUN_REFS = ["it", "they", "them", "its", "their", "this", "that", "those", "these"]


def is_followup(question: str, conversation: Conversation) -> bool:
    """
    Detect if a question is a follow-up to the previous conversation.

    Heuristics:
    1. Contains follow-up indicator phrases
    2. Starts with a pronoun reference
    3. Very short question (likely contextual)
    4. No company/entity name mentioned (relies on prior context)
    """
    if conversation.turn_count == 0:
        return False

    q_lower = question.lower().strip()

    # Check follow-up indicators
    for indicator in _FOLLOWUP_INDICATORS:
        if indicator in q_lower:
            return True

    # Check if starts with pronoun
    first_word = q_lower.split()[0] if q_lower.split() else ""
    if first_word in _PRONOUN_REFS:
        return True

    # Very short question (< 5 words) after existing conversation
    if len(q_lower.split()) < 5 and conversation.turn_count >= 2:
        return True

    return False


def resolve_followup(question: str, conversation: Conversation) -> str:
    """
    Resolve a follow-up question by injecting context from conversation history.

    Returns an enriched question with prior context prepended.
    """
    if not is_followup(question, conversation):
        return question

    context = conversation.get_context(last_n=3)
    return (
        f"[Conversation context]\n{context}\n\n"
        f"[Current question]\n{question}"
    )


class ConversationManager:
    """Manages multiple conversation sessions."""

    def __init__(self, max_conversations: int = 50) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._max = max_conversations

    def get_or_create(self, conversation_id: str | None = None) -> Conversation:
        """Get existing conversation or create new one."""
        if conversation_id and conversation_id in self._conversations:
            return self._conversations[conversation_id]

        conv = Conversation()
        self._conversations[conv.id] = conv

        # Evict oldest if over limit
        if len(self._conversations) > self._max:
            oldest_id = min(self._conversations, key=lambda k: self._conversations[k].created_at)
            del self._conversations[oldest_id]

        return conv

    def get(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    @property
    def active_count(self) -> int:
        return len(self._conversations)
