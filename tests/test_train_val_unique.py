import os
import json

from src.utils.constants import DATA_DIR
from src.game.game_state import AvalonGameState

TRAIN_GAMES_PATH = os.path.join(DATA_DIR, "games_train.json")
VAL_GAMES_PATH = os.path.join(DATA_DIR, "games_val.json")
with open(TRAIN_GAMES_PATH, "r") as f:
    train_games = json.load(f)
with open(VAL_GAMES_PATH, "r") as f:
    val_games = json.load(f)

print(f"Found {len(train_games)} train games in {TRAIN_GAMES_PATH}")
print(f"Found {len(val_games)} val games in {VAL_GAMES_PATH}")

train_game_states = [AvalonGameState.from_json(game) for game in train_games]
val_games_states = [AvalonGameState.from_json(game) for game in val_games]

for i, tgs_1 in enumerate(train_game_states):
    for j, tgs_2 in enumerate(train_game_states):
        if i != j:
            try:
                tgs_1.assert_equals(tgs_2)
                raise RuntimeError("The previous line should have raised an AssertionError")
            except AssertionError:
                pass # This is expected
            except Exception as e:
                raise e

print(f"Passed: Train games are unique!")

for i, vgs_1 in enumerate(val_games_states):
    for j, vgs_2 in enumerate(val_games_states):
        if i != j:
            try:
                vgs_1.assert_equals(vgs_2)
                raise RuntimeError("The previous line should have raised an AssertionError")
            except AssertionError:
                pass # This is expected
            except Exception as e:
                raise e
            
print(f"Passed: Val games are unique!")

for tgs in train_game_states:
    for vgs in val_games_states:
        try:
            tgs.assert_equals(vgs)
            raise RuntimeError("The previous line should have raised an AssertionError")
        except AssertionError:
            pass # This is expected
        except Exception as e:
            raise e
    
print(f"Passed: Train games and val games are cross-unique!")