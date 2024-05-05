import os
import json
import torch
from torch.nn.functional import softmax
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm

from src.game.utils import Role, GameStage
from src.utils.constants import (
    DATA_DIR,
    MODELS_DIR,
)
from src.game.game_state import AvalonGameState
from src.models.belief_predictor import BeliefPredictor
from src.game.beliefs import Belief, all_possible_ordered_role_assignments
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

class BeliefsSimulator(AvalonSimulator):
    
    def run_to_completion(self, belief_model: BeliefPredictor, player_index: int) -> Tuple[AvalonGameState, List[Belief]]:
        """
        Run the game to completion. Print the beliefs of the player roles at each step from
        the perspective of the player with the given index.
        """
        beliefs = []
        belief_role_assignments = all_possible_ordered_role_assignments(self.game_state.player_assignments)

        while self.game_state.game_stage not in [GameStage.RESISTANCE_WIN, GameStage.SPY_WIN]:
            self.step()
            obs = self.game_state.game_state_obs(from_perspective_of_player_index=player_index)
            beliefs_probs = softmax(belief_model(torch.tensor(obs).float().unsqueeze(0))).detach().numpy()[0]    
            beliefs.append(Belief(belief_role_assignments, beliefs_probs))
                 
        return self.game_state, beliefs
    
def get_belief_score(belief: Belief, role_assignment: List[Role]) -> float:
    """
    Get the score of a belief for a given role assignment
    """
    total_scores = []
    # For each player, determine how well the belief predicts their true role.
    for player_idx in range(len(role_assignment)):
        player_i_score = belief.marginalize([role_assignment[i] if i == player_idx else Role.UNKNOWN for i in range(len(role_assignment))])
        total_scores.append(player_i_score)
    return np.mean(total_scores)

def get_belief_scores_throughout_game(
    game_state: AvalonGameState,
    belief_model: BeliefPredictor,
    player_index: int,
    n_measurements: int,
) -> Dict[float, List[float]]:
    """
    For n measurements throughout the game, determine how well player_index's belief predicts the true role assignment. Measurements are evenly spaced throughout the game.
    """
    
    final_game_state, beliefs = run_game_from_perspective(game_state, belief_model, player_index, verbose=False)
    role_assignment = final_game_state.player_assignments
    
    out = {}
    for i in range(n_measurements):
        percent = i / (n_measurements - 1)
        idx = int(percent * (len(beliefs) - 1))
        belief = beliefs[idx]
        score = get_belief_score(belief, role_assignment)
        out[round(percent, 2)] = score
    return out

def get_average_belief_scores(
    belief_scores: List[Dict[float, List[float]]]
) -> Dict[float, List[float]]:
    """
    Given a list of belief scores, average them to get a single belief score for each percent.
    """
    out = {}
    for belief_score in belief_scores:
        for percent, score in belief_score.items():
            if percent not in out:
                out[percent] = []
            out[percent].append(score)
    for percent in out:
        out[percent] = np.mean(out[percent])
    return out

def get_average_belief_scores_by_role(
    game_states: List[AvalonGameState],
    belief_model: BeliefPredictor,
    n_measurements: int,
) -> Dict[Role, Dict[float, List[float]]]:
    """
    Get the average belief scores for each role in the game. Belief scores measure how well the belief model predicts the true role assignment. Scores range from 0 to 1 where 1 is a perfect prediction.
    
    Assumes all games have the same roles.
    """
    belief_scores = {role: [] for role in game_states[0].player_assignments}
    for game_state in tqdm(game_states):
        for player_index in range(len(game_state.player_assignments)):
            scores = get_belief_scores_throughout_game(game_state, belief_model, player_index, n_measurements)
            role = game_state.player_assignments[player_index]
            belief_scores[role].append(scores)
    out = {}
    for role in belief_scores.keys():
        out[role] = get_average_belief_scores(belief_scores[role])
    return out
    
def run_game_from_perspective(
    game_state: AvalonGameState,
    belief_model: BeliefPredictor,
    player_index: int,
    verbose: bool = True
) -> Tuple[AvalonGameState, List[Belief]]:
    """
    Run a game from the perspective of a player
    """
    roles = game_state.player_assignments
    simulator = BeliefsSimulator(
        roles,
        get_mimic_player_factory(game_state),
        randomize_player_assignments=False,
        verbose=verbose,
    )
    final_game_state, beliefs = simulator.run_to_completion(belief_model, player_index)
    return final_game_state, beliefs
    
if __name__ == "__main__":
    
    # EXPERIMENT_NAME = "belief_tf_16_30_10"
    EXPERIMENT_NAME = "belief_simple_16_30_10"
    
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
    
    # Get average belief scores by role
    n_measurements = 5
    belief_scores = get_average_belief_scores_by_role(eval_game_states, model, n_measurements)
    for role, scores in belief_scores.items():
        print(f"\n\n=========================\n==== Role: {role} ====\n=========================\n\n")
        for percent, score in scores.items():
            print(f"Percent: {percent}, Score: {score}")
    