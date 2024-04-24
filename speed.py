import time
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

last_time = time.time()
count = 0
while True:
    simulator = AvalonSimulator(roles, stupid_hardcoded_player_factory)
    final_game_state = simulator.run_to_completion()
    
    count += 1
    
    # every second, print the number of games played
    if time.time() - last_time >= 1:
        print(f"[SPEED] {count}/s games played")
        last_time = time.time()
        count = 0