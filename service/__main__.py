"""Entry point for running the AI Economist CLI."""
from .simulation import Simulation, SimulationConfig


def main() -> None:
    config = SimulationConfig()
    simulation = Simulation(config=config)
    episode = simulation.run_episode()
    print("Episode summary:", episode.summary())


if __name__ == "__main__":
    main()
