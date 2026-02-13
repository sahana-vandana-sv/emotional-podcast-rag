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

    def build_context_string(self) -> str:
        """
        Return plain-text summary of recent turns for LLM prompt.

        Example
        -------
        Previous conversation:
        [Turn 1] User felt: anxiety | Asked: "I feel stressed about my job"
                 Recommended: "Tim Ferriss on Fear", "Brené Brown on Shame"
        """
        if not self._history:
            return ""

        lines = ["Previous conversation:"]
        for i, turn in enumerate(self._history, 1):
            titles = [
                r.get('metadata', {}).get('episode_title', 'Unknown')
                for r in turn.recommendations
            ]
            lines.append(
                f"[Turn {i}] User felt: {turn.primary_emotion} | "
                f'Asked: "{turn.user_query}"\n'
                f"         Recommended: {', '.join(titles)}"
            )
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