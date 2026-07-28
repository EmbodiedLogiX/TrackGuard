from __future__ import annotations

from typing import Callable, List


class UndoHistory:
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self._stack: List = []

    def push(self, snapshot) -> None:
        self._stack.append(snapshot)
        if len(self._stack) > self.capacity:
            self._stack.pop(0)

    def pop(self):
        if not self._stack:
            return None
        return self._stack.pop()

    def can_undo(self) -> bool:
        return bool(self._stack)

    def clear(self) -> None:
        self._stack.clear()

    def guard(self, take_snapshot: Callable):
        self.push(take_snapshot())
