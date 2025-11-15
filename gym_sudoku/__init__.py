import environment
from gymnasium.envs.registration import register

__all__ = ["environment"]
__version__ = ("0.1.0")

register(
        id="sudoku",
        entry_point = "environment:Gym_env"
)



