"""Tax planner logic for the AI Economist simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .agents import Agent


@dataclass
class TaxBracket:
    threshold: float
    rate: float


@dataclass
class ProgressiveTaxPolicy:
    """A progressive tax system with piecewise-linear brackets."""

    brackets: List[TaxBracket]
    redistribution_fraction: float = 1.0

    def compute_tax(self, wealth: float) -> float:
        tax_due = 0.0
        remaining = wealth
        lower_threshold = 0.0
        for bracket in self.brackets:
            taxable = max(0.0, min(remaining, bracket.threshold - lower_threshold))
            tax_due += taxable * bracket.rate
            remaining -= taxable
            lower_threshold = bracket.threshold
        if remaining > 0:
            tax_due += remaining * self.brackets[-1].rate
        return tax_due

    def collect_taxes(self, agents: Iterable[Agent]) -> Dict[str, float]:
        taxes: Dict[str, float] = {}
        for agent in agents:
            due = self.compute_tax(agent.wealth)
            taxes[agent.agent_id] = agent.pay_tax(due)
        return taxes

    def redistribute(self, agents: Iterable[Agent], taxes: Dict[str, float]) -> None:
        total_collected = sum(taxes.values())
        if total_collected <= 0:
            return
        redistribution_pool = total_collected * self.redistribution_fraction
        recipients = list(agents)
        per_agent = redistribution_pool / len(recipients)
        for agent in recipients:
            agent.wealth += per_agent


DEFAULT_TAX_POLICY = ProgressiveTaxPolicy(
    brackets=[
        TaxBracket(threshold=5.0, rate=0.1),
        TaxBracket(threshold=10.0, rate=0.2),
        TaxBracket(threshold=20.0, rate=0.3),
    ],
    redistribution_fraction=0.9,
)
