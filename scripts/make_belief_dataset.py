import os
import json
from typing import Tuple, List
import numpy as np
from tqdm import tqdm


from src.utils.constants import DATA_DIR, BELIEF_DATASET_PATH
from src.game.game_state import AvalonGameState
from src.players.mimic_player import get_mimic_player_factory
from src.game.simulator import AvalonSimulator
from src.game.utils import GameStage

# TODO: Make this an arg
INPUT_FILE_PATH = os.path.join(DATA_DIR, "games_final.json")

class BeliefDatasetSimulator(AvalonSimulator):
    
    def run_to_completion(self) -> Tuple[AvalonGameState, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Run the game to completion and return the final game state,
        along with (state, ground truth belief distribution) pairs collected
        """
        
        dataset = []
        while self.game_state.game_stage not in [GameStage.RESISTANCE_WIN, GameStage.SPY_WIN]:
            self.step()
            dataset.append((
                self.game_state.game_state_obs(),
                self.game_state.ground_truth_role_distribution(),
            ))
        return self.game_state, dataset

if __name__ == "__main__":

    with open(INPUT_FILE_PATH, "r") as f:
        games = json.load(f)

    print(f"Found {len(games)} games in {INPUT_FILE_PATH}")

    # Compile dataset
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    for game in tqdm(games):
        # Load a game state from file
        game_state = AvalonGameState.from_json(game)
        roles = game_state.player_assignments
        
        # Simulate the game using mimic players
        simulator = BeliefDatasetSimulator(
            roles,
            get_mimic_player_factory(game_state),
            randomize_player_assignments=False,
            verbose=False
        )
        final_game_state, partial_dset = simulator.run_to_completion()
        dataset.extend(partial_dset)
        
        # Assert that the two game states match
        game_state.assert_equals(final_game_state)
        
    # Post-process and save to file
    game_histories = np.stack([
        data[0] for data in dataset
    ])
    game_beliefs = np.stack([
        data[1] for data in dataset
    ])

    print(game_histories.shape)
    print(game_beliefs.shape)

    np.savez(
        BELIEF_DATASET_PATH,
        game_histories=game_histories,
        game_beliefs=game_beliefs,
    )