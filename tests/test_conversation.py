"""Tests for conversation memory and follow-up detection."""

from src.core.conversation import (
    Conversation,
    ConversationManager,
    is_followup,
    resolve_followup,
)


class TestConversation:
    def test_add_turn(self):
        conv = Conversation()
        conv.add_turn("user", "What is NVIDIA revenue?")
        conv.add_turn("assistant", "$130B in FY2025")
        assert conv.turn_count == 2

    def test_max_turns_eviction(self):
        conv = Conversation(max_turns=3)
        for i in range(5):
            conv.add_turn("user", f"Question {i}")
        assert conv.turn_count == 3
        assert "Question 2" in conv.turns[0].content

    def test_get_context(self):
        conv = Conversation()
        conv.add_turn("user", "Q1")
        conv.add_turn("assistant", "A1")
        conv.add_turn("user", "Q2")
        ctx = conv.get_context(last_n=2)
        assert "Assistant: A1" in ctx
        assert "User: Q2" in ctx

    def test_last_user_question(self):
        conv = Conversation()
        conv.add_turn("user", "First")
        conv.add_turn("assistant", "Reply")
        conv.add_turn("user", "Second")
        assert conv.last_user_question() == "Second"

    def test_last_answer(self):
        conv = Conversation()
        conv.add_turn("user", "Q")
        conv.add_turn("assistant", "A")
        assert conv.last_answer() == "A"

    def test_empty_conversation(self):
        conv = Conversation()
        assert conv.last_user_question() is None
        assert conv.last_answer() is None
        assert conv.turn_count == 0


class TestFollowupDetection:
    def setup_method(self):
        self.conv = Conversation()
        self.conv.add_turn("user", "What was NVIDIA revenue in FY2025?")
        self.conv.add_turn("assistant", "$130,497 million")

    def test_indicator_phrase(self):
        assert is_followup("What about FY2024?", self.conv)
        assert is_followup("How about Apple?", self.conv)
        assert is_followup("And for Microsoft?", self.conv)

    def test_pronoun_start(self):
        assert is_followup("It was higher than expected?", self.conv)
        assert is_followup("Their net income?", self.conv)

    def test_short_question(self):
        assert is_followup("And FY2024?", self.conv)

    def test_not_followup_fresh(self):
        empty = Conversation()
        assert not is_followup("What is NVIDIA revenue?", empty)

    def test_not_followup_explicit(self):
        assert not is_followup("What was Apple total revenue in fiscal year 2023?", self.conv)

    def test_comparison_followup(self):
        assert is_followup("Compared to Microsoft?", self.conv)
        assert is_followup("vs Apple?", self.conv)


class TestResolveFollowup:
    def test_enriches_followup(self):
        conv = Conversation()
        conv.add_turn("user", "NVIDIA revenue FY2025?")
        conv.add_turn("assistant", "$130B")
        result = resolve_followup("What about FY2024?", conv)
        assert "[Conversation context]" in result
        assert "NVIDIA revenue" in result
        assert "What about FY2024?" in result

    def test_passthrough_non_followup(self):
        conv = Conversation()
        result = resolve_followup("What is Apple revenue in FY2023?", conv)
        assert result == "What is Apple revenue in FY2023?"


class TestConversationManager:
    def test_create_new(self):
        mgr = ConversationManager()
        conv = mgr.get_or_create()
        assert conv.turn_count == 0
        assert mgr.active_count == 1

    def test_get_existing(self):
        mgr = ConversationManager()
        conv1 = mgr.get_or_create()
        conv1.add_turn("user", "Hello")
        conv2 = mgr.get_or_create(conv1.id)
        assert conv2.turn_count == 1

    def test_eviction(self):
        mgr = ConversationManager(max_conversations=2)
        c1 = mgr.get_or_create()
        _c2 = mgr.get_or_create()
        _c3 = mgr.get_or_create()
        assert mgr.active_count == 2
        assert mgr.get(c1.id) is None  # evicted

    def test_format_history(self):
        conv = Conversation()
        assert "No conversation history" in conv.format_history()
        conv.add_turn("user", "What is NVIDIA revenue?")
        conv.add_turn("assistant", "$130B")
        hist = conv.format_history()
        assert "2 turns" in hist
        assert "Q1:" in hist
        assert "A2:" in hist

    def test_get_nonexistent(self):
        mgr = ConversationManager()
        assert mgr.get("nonexistent") is None
