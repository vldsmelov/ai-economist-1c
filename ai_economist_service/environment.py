"""Grid environment for the AI Economist simulation."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from .agents import Agent

Action = Tuple[int, int]


@dataclass
class GridEnvironment:
    """A simple resource gathering environment."""

    width: int
    height: int
    resources: Dict[Tuple[int, int], float] = field(default_factory=dict)

    def spawn_resources(self, seed: int, density: float, amount: float) -> None:
        """Populate the grid with resources using the provided density."""
        rng = random.Random(seed)
        self.resources.clear()
        total_cells = self.width * self.height
        resource_cells = int(total_cells * density)
        for _ in range(resource_cells):
            x = rng.randrange(self.width)
            y = rng.randrange(self.height)
            self.resources[(x, y)] = self.resources.get((x, y), 0.0) + amount

    def apply_actions(self, agents: Iterable[Agent], actions: Dict[str, Action]) -> Dict[str, float]:
        """Apply movement actions for each agent and return collected resources."""
        collected: Dict[str, float] = {}
        for agent in agents:
            action = actions.get(agent.agent_id, (0, 0))
            agent.move(action, self.width, self.height)
            gained = self._collect_resource(agent)
            collected[agent.agent_id] = gained
        return collected

    def _collect_resource(self, agent: Agent) -> float:
        amount = self.resources.get(agent.position, 0.0)
        if amount <= 0:
            return 0.0
        gathered = min(1.0, amount)
        self.resources[agent.position] = max(0.0, amount - gathered)
        return agent.collect(gathered)

    def serialize(self) -> Dict[str, object]:
        """Return a JSON-serializable representation of the grid."""
        return {
            "width": self.width,
            "height": self.height,
            "resources": [
                {"x": x, "y": y, "amount": amount}
                for (x, y), amount in self.resources.items()
                if amount > 0
            ],
        }


def random_action_space() -> List[Action]:
    """Return the discrete action space of the environment."""
    return [
        (-1, 0),  # left
        (1, 0),   # right
        (0, -1),  # down
        (0, 1),   # up
        (0, 0),   # stay
    ]
