"""Agent definitions for the AI Economist service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Agent:
    """Simple agent participating in the simulation.

    Attributes
    ----------
    agent_id: str
        Unique identifier for the agent.
    position: Tuple[int, int]
        Current coordinates on the grid.
    wealth: float
        Accumulated wealth of the agent.
    skill: float
        Productivity multiplier affecting resource gathering.
    """

    agent_id: str
    position: Tuple[int, int]
    wealth: float = 0.0
    skill: float = 1.0

    def move(self, direction: Tuple[int, int], width: int, height: int) -> None:
        """Move the agent within the environment bounds."""
        x, y = self.position
        dx, dy = direction
        new_x = max(0, min(width - 1, x + dx))
        new_y = max(0, min(height - 1, y + dy))
        self.position = (new_x, new_y)

    def collect(self, amount: float) -> float:
        """Increase the agent's wealth and return the collected amount."""
        gained = amount * self.skill
        self.wealth += gained
        return gained

    def pay_tax(self, amount: float) -> float:
        """Deduct tax from the agent's wealth and return the paid amount."""
        paid = min(amount, self.wealth)
        self.wealth -= paid
        return paid
