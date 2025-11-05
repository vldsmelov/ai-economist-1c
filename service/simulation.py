"""High level simulation orchestrator for the AI Economist service."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .agents import Agent
from .environment import GridEnvironment, random_action_space
from .planner import DEFAULT_TAX_POLICY, ProgressiveTaxPolicy


@dataclass
class SimulationConfig:
    width: int = 8
    height: int = 8
    num_agents: int = 4
    steps: int = 30
    resource_density: float = 0.25
    resource_amount: float = 2.0
    seed: Optional[int] = None

    def create_agents(self, rng: random.Random) -> List[Agent]:
        agents: List[Agent] = []
        for idx in range(self.num_agents):
            x = rng.randrange(self.width)
            y = rng.randrange(self.height)
            skill = rng.uniform(0.8, 1.2)
            agents.append(
                Agent(agent_id=f"agent_{idx}", position=(x, y), wealth=0.0, skill=skill)
            )
        return agents


@dataclass
class StepResult:
    step_index: int
    actions: Dict[str, List[int]]
    collected: Dict[str, float]
    taxes: Dict[str, float]
    wealth: Dict[str, float]


@dataclass
class EpisodeResult:
    config: SimulationConfig
    steps: List[StepResult]
    final_wealth: Dict[str, float]

    def summary(self) -> Dict[str, float]:
        return {
            "mean_wealth": sum(self.final_wealth.values()) / len(self.final_wealth),
            "max_wealth": max(self.final_wealth.values()),
            "min_wealth": min(self.final_wealth.values()),
        }


@dataclass
class Simulation:
    config: SimulationConfig = field(default_factory=SimulationConfig)
    tax_policy: ProgressiveTaxPolicy = field(default_factory=lambda: DEFAULT_TAX_POLICY)

    def __post_init__(self) -> None:
        seed = self.config.seed or random.randrange(1_000_000)
        self._rng = random.Random(seed)
        self.environment = GridEnvironment(self.config.width, self.config.height)
        self.environment.spawn_resources(
            seed=seed,
            density=self.config.resource_density,
            amount=self.config.resource_amount,
        )
        self.agents = self.config.create_agents(self._rng)
        self.action_space = random_action_space()

    def run_episode(self, steps: Optional[int] = None, seed: Optional[int] = None) -> EpisodeResult:
        steps_to_run = steps if steps is not None else self.config.steps
        rng = random.Random(seed if seed is not None else self.config.seed)
        results: List[StepResult] = []

        for step_index in range(steps_to_run):
            actions = self._sample_actions(rng)
            collected = self.environment.apply_actions(self.agents, actions)
            taxes = self.tax_policy.collect_taxes(self.agents)
            self.tax_policy.redistribute(self.agents, taxes)
            wealth_snapshot = {agent.agent_id: agent.wealth for agent in self.agents}
            results.append(
                StepResult(
                    step_index=step_index,
                    actions={aid: list(direction) for aid, direction in actions.items()},
                    collected=collected,
                    taxes=taxes,
                    wealth=wealth_snapshot,
                )
            )
        final_wealth = {agent.agent_id: agent.wealth for agent in self.agents}
        return EpisodeResult(config=self.config, steps=results, final_wealth=final_wealth)

    def _sample_actions(self, rng: Optional[random.Random]) -> Dict[str, tuple[int, int]]:
        local_rng = rng or self._rng
        actions: Dict[str, tuple[int, int]] = {}
        for agent in self.agents:
            action = local_rng.choice(self.action_space)
            actions[agent.agent_id] = action
        return actions
