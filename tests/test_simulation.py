from ai_economist_service.simulation import Simulation, SimulationConfig


def test_episode_runs_and_produces_summary():
    config = SimulationConfig(steps=5, num_agents=2, width=4, height=4, seed=42)
    simulation = Simulation(config=config)
    episode = simulation.run_episode()

    summary = episode.summary()
    assert "mean_wealth" in summary
    assert len(episode.steps) == 5
    assert len(episode.final_wealth) == 2
