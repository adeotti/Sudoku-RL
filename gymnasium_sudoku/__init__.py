from gymnasium.envs.registration import register

__all__ = ["Gym_env"]
__version__ = "0.1.5"

register(
    id="sudoku-v0",
    entry_point="gymnasium_sudoku.environment:Gym_env",
)

