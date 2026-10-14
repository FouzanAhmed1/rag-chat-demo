"""Per-session in-memory conversation history.

The client identifies itself with an X-Session-Id header (or a session_id
field in the chat body). The server keeps the last SESSION_MAX_TURNS turns
per session so follow-up questions can see what came before; sessions that
go quiet for SESSION_TTL_SECONDS are swept on access. History is
process-local and never persisted: a restart clears it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app import config


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class _Session:
    turns: list[Turn] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)


class SessionStore:
    def __init__(
        self, max_turns: int | None = None, ttl_seconds: int | None = None
    ) -> None:
        self.max_turns = (
            config.SESSION_MAX_TURNS if max_turns is None else max_turns
        )
        self.ttl_seconds = (
            config.SESSION_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        )
        self._sessions: dict[str, _Session] = {}

    def _sweep(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, sess in self._sessions.items()
            if now - sess.last_seen > self.ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]

    def history(self, session_id: str) -> list[Turn]:
        """Return this session's turns (oldest first), sweeping expired
        sessions first. Unknown or expired ids return []."""
        if not session_id:
            return []
        self._sweep()
        sess = self._sessions.get(session_id)
        return list(sess.turns) if sess else []

    def record(self, session_id: str, question: str, answer: str) -> None:
        """Append a turn, trimming to max_turns and refreshing the TTL."""
        if not session_id:
            return
        self._sweep()
        sess = self._sessions.setdefault(session_id, _Session())
        sess.turns.append(Turn(question=question, answer=answer))
        sess.turns = sess.turns[-self.max_turns :]
        sess.last_seen = time.time()

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# Process-wide store used by the API routes.
sessions = SessionStore()


def rewrite_query(question: str, history: list[Turn]) -> str:
    """Build the retrieval query from history plus the new question.

    A deliberately simple approach: concatenate the last couple of user
    questions so follow-ups like "how about returns?" keep the topic of
    the earlier turns.
    """
    if not history:
        return question
    recent = " ".join(t.question for t in history[-2:])
    return f"{recent} {question}".strip()


def history_prompt_block(history: list[Turn]) -> str:
    """Render past turns for the LLM prompt context."""
    if not history:
        return ""
    lines = ["Conversation so far:"]
    for turn in history:
        lines.append(f"User: {turn.question}")
        lines.append(f"Assistant: {turn.answer[:500]}")
    return "\n".join(lines)
