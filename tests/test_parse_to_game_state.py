import os
import json
from tqdm import tqdm

from src.utils.constants import DATA_DIR
from src.game.game_state import AvalonGameState
from src.players.mimic_player import get_mimic_player_factory
from src.game.simulator import AvalonSimulator

file_path = os.path.join(DATA_DIR, "games_final.json")
with open(file_path, "r") as f:
    games = json.load(f)

print(f"Found {len(games)} games in games_final.json")

for game in tqdm(games):
    # Load a game state from file
    game_state = AvalonGameState.from_json(game)
    roles = game_state.player_assignments
    
    # Simulate the game using mimic players
    simulator = AvalonSimulator(
        roles,
        get_mimic_player_factory(game_state),
        randomize_player_assignments=False,
        verbose=False
    )
    final_game_state = simulator.run_to_completion()
    
    # Assert that the two game states match
    game_state.assert_equals(final_game_state)
    
print(f"Passed: In all {len(games)} games, all game states match!")