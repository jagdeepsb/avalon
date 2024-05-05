import os
import json
import torch
from torch.nn.functional import softmax
import numpy as np
from typing import List

from src.game.utils import Role, GameStage
from src.utils.constants import (
    DATA_DIR,
    MODELS_DIR,
)
from src.game.game_state import AvalonGameState
from src.models.belief_predictor import BeliefPredictor
from src.game.beliefs import all_possible_ordered_role_assignments
from src.players.mimic_player import get_mimic_player_factory
from src.game.simulator import AvalonSimulator

def assignment_to_str(assignment: List[Role]) -> str:
    """
    Convert a role assignment to a string
    """
    out = ""
    for role in assignment:
        if role == Role.RESISTANCE:
            out += "R"
        elif role == Role.SPY:
            out += "S"
        elif role == Role.MERLIN:
            out += "M"
        else:
            raise ValueError(f"Unknown role: {role}")
    return out

def round(x: float, n: int) -> float:
    """
    Round a number to n decimal places
    """
    return np.round(x, n)

def print_useful_belief_information(roles: List[Role], beliefs: np.ndarray):
    """
    Print useful information to help understand the beliefs of the model
    """
    
    # get assignments
    all_assignments = all_possible_ordered_role_assignments(roles)
    
    # get indices of beliefs in descending order
    k = 5
    sorted_indices = np.argsort(beliefs)[::-1]
    for i, idx in enumerate(sorted_indices):
        if i == k:
            break
        print(f"Assignment: {assignment_to_str(all_assignments[idx])}, Belief: {round(beliefs[idx], 4)}")
    
class PrintBeliefsSimulator(AvalonSimulator):
    
    def run_to_completion(self, belief_model: BeliefPredictor, player_index: int) -> AvalonGameState:
        """
        Run the game to completion. Print the beliefs of the player roles at each step from
        the perspective of the player with the given index.
        """
        
        while self.game_state.game_stage not in [GameStage.RESISTANCE_WIN, GameStage.SPY_WIN]:
            self.step()
            obs = self.game_state.game_state_obs(from_perspective_of_player_index=player_index)
            # obs = self.game_state.game_state_obs()
            beliefs = softmax(belief_model(torch.tensor(obs).float().unsqueeze(0))).detach().numpy()[0]
            print_useful_belief_information(self.game_state.player_assignments, beliefs)
        return self.game_state
    
def run_game_from_perspective(
    game_state: AvalonGameState,
    belief_model: BeliefPredictor,
    player_index: int
):
    """
    Run a game from the perspective of a player
    """
    roles = game_state.player_assignments
    simulator = PrintBeliefsSimulator(
        roles,
        get_mimic_player_factory(game_state),
        randomize_player_assignments=False,
        verbose=True
    )
    final_game_state = simulator.run_to_completion(belief_model, player_index)
    return final_game_state
    
if __name__ == "__main__":
    
    # EXPERIMENT_NAME = "belief_tf_16_30_10"
    EXPERIMENT_NAME = "belief_debug_16_30_10"

    # Load Games
    EVAL_GAMES_PATH = os.path.join(DATA_DIR, "games_val.json")
    # EVAL_GAMES_PATH = os.path.join(DATA_DIR, "games_train.json")
    with open(EVAL_GAMES_PATH, "r") as f:
        eval_games = json.load(f)
    eval_game_states = [AvalonGameState.from_json(game) for game in eval_games]
    
    # Load model
    model_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}.pt")
    model: BeliefPredictor = torch.load(model_path)
    model.eval()
    
    # Run a single game from each players perspective
    game_idx = 0
    for player_idx in range(len(eval_game_states[game_idx].player_assignments)):
        print(f"\n\n=========================\n==== Player Index: {player_idx} ====\n=========================\n\n")
        game_state = eval_game_states[game_idx]
        final_game_state = run_game_from_perspective(game_state, model, player_idx)
        print(f"Game Outcome: {final_game_state.game_stage}")