import os
import json
from typing import Tuple, List
import numpy as np
from tqdm import tqdm


from src.utils.constants import (
    DATA_DIR,
    RES_TRAIN_BELIEF_DATASET_PATH,
    RES_VAL_BELIEF_DATASET_PATH,
    SPY_TRAIN_BELIEF_DATASET_PATH,
    SPY_VAL_BELIEF_DATASET_PATH,
)
from src.game.game_state import AvalonGameState
from src.players.mimic_player import get_mimic_player_factory
from src.game.simulator import AvalonSimulator
from src.game.utils import GameStage, Role

class BeliefDatasetSimulator(AvalonSimulator):
    
    def run_to_completion(self) -> Tuple[AvalonGameState, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Run the game to completion and return the final game state,
        along with (state, ground truth belief distribution) pairs collected
        """
        
        res_dset, spy_dset = [], []
        while self.game_state.game_stage not in [GameStage.RESISTANCE_WIN, GameStage.SPY_WIN]:
            self.step()
            for i in range(len(self.game_state.player_assignments)):
                if self.game_state.player_assignments[i] == Role.RESISTANCE:
                    res_dset.append((
                        self.game_state.game_state_obs(from_perspective_of_player_index=i),
                        self.game_state.ground_truth_role_distribution(),
                    ))
                elif self.game_state.player_assignments[i] == Role.SPY:
                    spy_dset.append((
                        self.game_state.game_state_obs(from_perspective_of_player_index=i),
                        self.game_state.ground_truth_role_distribution(),
                    ))
        return self.game_state, res_dset, spy_dset
    
def process_dset_and_save(dset: List[Tuple[np.ndarray, np.ndarray]], output_file_path: str):
    game_histories = np.stack([
        data[0] for data in dset
    ])
    game_beliefs = np.stack([
        data[1] for data in dset
    ])

    print(game_histories.shape)
    print(game_beliefs.shape)

    np.savez(
        output_file_path,
        game_histories=game_histories,
        game_beliefs=game_beliefs,
    )
    
def make_dset(input_file_path: str, res_output_file_path: str, spy_output_file_path: str):
    
    with open(input_file_path, "r") as f:
        games = json.load(f)

    print(f"Found {len(games)} games in {input_file_path}")

    # Compile dataset
    res_dset: List[Tuple[np.ndarray, np.ndarray]] = []
    spy_dset: List[Tuple[np.ndarray, np.ndarray]] = []
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
        final_game_state, partial_r_dset, partial_s_dset = simulator.run_to_completion()
        res_dset.extend(partial_r_dset)
        spy_dset.extend(partial_s_dset)
        
        # Assert that the two game states match
        game_state.assert_equals(final_game_state)
        
    # Process and save the dataset
    process_dset_and_save(res_dset, res_output_file_path)
    process_dset_and_save(spy_dset, spy_output_file_path)

if __name__ == "__main__":

    input_file_path = os.path.join(DATA_DIR, "games_train.json")
    res_output_file_path = RES_TRAIN_BELIEF_DATASET_PATH
    spy_output_file_path = SPY_TRAIN_BELIEF_DATASET_PATH
    make_dset(input_file_path, res_output_file_path, spy_output_file_path)

    input_file_path = os.path.join(DATA_DIR, "games_val.json")
    res_output_file_path = RES_VAL_BELIEF_DATASET_PATH
    spy_output_file_path = SPY_VAL_BELIEF_DATASET_PATH
    make_dset(input_file_path, res_output_file_path, spy_output_file_path)
    