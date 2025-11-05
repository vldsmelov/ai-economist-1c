"""CLI for running an AI Economist episode and printing a summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_economist_service.simulation import Simulation, SimulationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an AI Economist simulation")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to run")
    parser.add_argument("--width", type=int, default=None, help="Grid width")
    parser.add_argument("--height", type=int, default=None, help="Grid height")
    parser.add_argument("--num-agents", type=int, default=None, help="Number of agents")
    parser.add_argument("--resource-density", type=float, default=None, help="Resource density")
    parser.add_argument("--resource-amount", type=float, default=None, help="Resource amount")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save JSON result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig()
    if args.steps is not None:
        config.steps = args.steps
    if args.width is not None:
        config.width = args.width
    if args.height is not None:
        config.height = args.height
    if args.num_agents is not None:
        config.num_agents = args.num_agents
    if args.resource_density is not None:
        config.resource_density = args.resource_density
    if args.resource_amount is not None:
        config.resource_amount = args.resource_amount
    if args.seed is not None:
        config.seed = args.seed

    simulation = Simulation(config=config)
    episode = simulation.run_episode()
    data = {
        "summary": episode.summary(),
        "final_wealth": episode.final_wealth,
    }
    print(json.dumps(data, indent=2, ensure_ascii=False))

    if args.output is not None:
        args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
