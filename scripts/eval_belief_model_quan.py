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
from src.players.stupid_hardcoded_player import stupid_hardcoded_player_factory

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
        elif role == Role.UNKNOWN:
            out += "U"
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
            # obs = self.game_state.game_state_obs(from_perspective_of_player_index=player_index)
            # beliefs_probs = softmax(belief_model(torch.tensor(obs).float().unsqueeze(0))).detach().numpy()[0]
            
            # beliefs_probs = Belief.make_uniform(self.game_state.player_assignments).distribution + np.random.normal(0, 0.01, len(belief_role_assignments))
            
            # condition on the fact that the player is resistance
            # belief = Belief(belief_role_assignments, beliefs_probs).condition_on(
            #     [Role.RESISTANCE if i == player_index else Role.UNKNOWN for i in range(len(self.game_state.player_assignments))]
            # )
            
            # condition on the fact that you know who the spies are
            # belief = Belief(belief_role_assignments, beliefs_probs).condition_on(
            #     [Role.SPY if role == Role.SPY else Role.UNKNOWN for role in self.game_state.player_assignments]
            # )
            
            from src.belief_models.trained import TrainedBeliefModel
            from src.belief_models.trivial import GroundTruthBeliefModel
            belief_model = TrainedBeliefModel()
            # belief_model = GroundTruthBeliefModel()
            belief = belief_model(self.game_state, player_index)
            # belief = get_belief_for_player_cheap(self.game_state, player_index, torch.device("cpu"))
            
            beliefs.append(belief)
                 
        return self.game_state, beliefs
    
def get_belief_score(belief: Belief, role_assignment: List[Role]) -> float:
    """
    Get the score of a belief for a given role assignment
    """
    # total_scores = []
    # # For each player, determine how well the belief predicts their true role.
    # for player_idx in range(len(role_assignment)):
    #     player_i_score = belief.marginalize([role_assignment[i] if i == player_idx else Role.UNKNOWN for i in range(len(role_assignment))])
    #     total_scores.append(player_i_score)
    # return np.mean(total_scores)
    
    # return 1 if belief is in top 3
    top_assignments = belief.get_top_k_assignments(1)
    for assignment, _ in top_assignments:
        if assignment == tuple(role_assignment):
            return 1
    return 0
    
    # number of spies correctly identified
    # top_assignment = belief.get_top_k_assignments(1)[0][0]
    # return sum([1 for i in range(len(role_assignment)) if (role_assignment[i] == Role.SPY and top_assignment[i] == Role.SPY)])

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

def get_final_game_state(n: int) -> List[AvalonGameState]:
    import random
    random.seed(42)
    game_states = []
    for i in tqdm(range(n)):
        simulator = AvalonSimulator(
            [Role.MERLIN, Role.RESISTANCE, Role.RESISTANCE, Role.SPY, Role.SPY],
            stupid_hardcoded_player_factory,
            randomize_player_assignments=True,
            verbose=False
        )
        final_game_state = simulator.run_to_completion()
        game_states.append(final_game_state)
    return game_states
    
if __name__ == "__main__":
    
    # EXPERIMENT_NAME = "belief_tf_16_30_10"
    # EXPERIMENT_NAME = "belief_simple_16_30_10"
    # EXPERIMENT_NAME = "belief_debug_16_30_10"
    EXPERIMENT_NAME = "res_belief_16_30_10_v1"
    # EXPERIMENT_NAME = "spy_belief_16_30_10_v2"
    
    # Load Games
    EVAL_GAMES_PATH = os.path.join(DATA_DIR, "games_val.json")
    # EVAL_GAMES_PATH = os.path.join(DATA_DIR, "games_train.json")
    with open(EVAL_GAMES_PATH, "r") as f:
        eval_games = json.load(f)
    eval_game_states = [AvalonGameState.from_json(game) for game in eval_games][:200]
    
    # eval_game_states = get_final_game_state(100)
    
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
    