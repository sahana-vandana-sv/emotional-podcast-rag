# src/memory.py
# -----------------------------------------------------------
# Conversation memory — stores last N turns so the LLM can
# refer to previous queries ("show me more like the last one")
# -----------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional
from src.config import MAX_HISTORY_TURNS


@dataclass
class Turn:
    """One conversation turn."""
    user_query:        str
    primary_emotion:   str
    recommendations:   list        # list of episode dicts
    assistant_summary: str


class ConversationMemory:
    """
    Rolling window of the last MAX_HISTORY_TURNS turns.

    Usage
    -----
    memory = ConversationMemory()
    memory.add(turn)
    context_str = memory.build_context_string()
    memory.clear()
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self.max_turns = max_turns
        self._history: list = []

    def add(self, turn: Turn) -> None:
        """Append a turn; drop oldest if window is full."""
        self._history.append(turn)
        if len(self._history) > self.max_turns:
            self._history.pop(0)

    def build_context_string(self):
        lines = []
        for turn in self._history:
            lines.append(f"User: {turn.user_query}")
            lines.append(f"Emotion: {turn.primary_emotion}")
            
            # Handle recommendations as a list of strings
            if turn.recommendations:
                if isinstance(turn.recommendations, list):
                    rec_str = ", ".join(turn.recommendations)
                else:
                    rec_str = str(turn.recommendations)
                
                lines.append(f"Recommended: {rec_str}")
        
        return "\n".join(lines)

    def last_turn(self) -> Optional[Turn]:
        """Return most recent turn, or None."""
        return self._history[-1] if self._history else None

    def clear(self) -> None:
        """Wipe all history."""
        self._history = []

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"ConversationMemory(turns={len(self._history)}/{self.max_turns})"