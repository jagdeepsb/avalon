from src.game.utils import Role
from src.game.simulator import AvalonSimulator
from src.players.random_player import RandomAvalonPlayer
from src.players.stupid_hardcoded_player import stupid_hardcoded_player_factory
from src.game.arena import AvalonArena

def random_player_factory(role: Role, index: int) -> RandomAvalonPlayer:
    return RandomAvalonPlayer(role, index)

roles = [
    Role.MERLIN,
    Role.RESISTANCE,
    Role.RESISTANCE,
    Role.SPY,
    Role.SPY,
]

arena = AvalonArena(
    roles=roles,
    player_factory_1=random_player_factory,
    player_factory_2=stupid_hardcoded_player_factory,
    num_games=10000,
)

for win_rates, label in zip(arena.get_win_rates_by_role(), ["Random", "Stupid Hardcoded"]):
    print(f"{label} win rates:")
    for role, win_rate in win_rates.items():
        print(f"{role}: {win_rate}")
    print()
    
for win_rate, label in zip(arena.get_overall_win_rates(), ["Random", "Stupid Hardcoded"]):
    print(f"{label} win rate: {win_rate}")