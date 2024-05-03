from src.game.utils import Role
from src.game.simulator import AvalonSimulator
from src.players.random_player import RandomAvalonPlayer

def player_factory(role: Role, index: int) -> RandomAvalonPlayer:
    return RandomAvalonPlayer(role, index)

roles = [
    Role.MERLIN,
    Role.RESISTANCE,
    Role.RESISTANCE,
    Role.SPY,
    Role.SPY,
]

simulator = AvalonSimulator(roles, player_factory, verbose=True)
final_game_state = simulator.run_to_completion()
print(final_game_state.game_stage)