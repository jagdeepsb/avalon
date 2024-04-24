from src.utils import Role
from src.simulator import AvalonSimulator
from src.players.stupid_hardcoded_player import stupid_hardcoded_player_factory

roles = [
    Role.MERLIN,
    Role.RESISTANCE,
    Role.RESISTANCE,
    Role.SPY,
    Role.SPY,
]

simulator = AvalonSimulator(roles, stupid_hardcoded_player_factory, verbose=True)
final_game_state = simulator.run_to_completion()
print(final_game_state.game_stage)