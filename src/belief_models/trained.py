import torch
from torch.nn.functional import softmax
from typing import Dict
import numpy as np

from src.belief_models.base import BeliefModel
from src.game.utils import Role
from src.game.game_state import AvalonGameState
from src.game.beliefs import Belief, all_possible_ordered_role_assignments
from src.utils.constants import RES_BELIEF_MODEL_PATH, SPY_BELIEF_MODEL_PATH

_MODELS: Dict[str, torch.nn.Module] = {}
_MODEL_DEVICES: Dict[str, torch.device] = {}

class TrainedBeliefModel(BeliefModel):
    def __call__(
        self,
        game_state: AvalonGameState,
        player_index: int,
        device: torch.device = torch.device("cpu")
    ) -> Belief:
        """
        Get the belief of the player with the given index.
        This doesn't use any models, and can be used for testing training.
        Use `belief.distribution` to get the probabilities of each role assignment.
        """
    
        global _MODELS, _MODEL_DEVICES
        role = game_state.player_assignments[player_index]
        
        # get the correct model
        model_path = RES_BELIEF_MODEL_PATH if role == Role.RESISTANCE else SPY_BELIEF_MODEL_PATH
        if model_path not in _MODELS:
            belief_model = torch.load(model_path)
            belief_model.to(device)
            belief_model.eval()
            _MODELS[model_path] = belief_model
            _MODEL_DEVICES[model_path] = device
        else:
            belief_model = _MODELS[model_path]
        
        # move model to correct device if necessary
        if device != _MODEL_DEVICES[model_path]:
            belief_model.to(device)
            _MODEL_DEVICES[model_path] = device
            
        if role == Role.RESISTANCE:
            obs = game_state.game_state_obs(from_perspective_of_player_index=player_index)
            obs = torch.tensor(obs).float().unsqueeze(0).to(device)
            beliefs_probs = softmax(belief_model(obs)).detach().numpy()[0]
            
            belief_role_assignments = all_possible_ordered_role_assignments(game_state.player_assignments)
                
            # condition on the fact that the player is resistance
            belief = Belief(belief_role_assignments, beliefs_probs).condition_on(
                [Role.RESISTANCE if i == player_index else Role.UNKNOWN for i in range(len(game_state.player_assignments))]
            )
            
            # rule out impossible role assignments given the quest history
            belief = belief.condition_on_quest_history(game_state.quest_teams, game_state.quest_votes)
            
        elif role == Role.SPY:
            obs = game_state.game_state_obs(from_perspective_of_player_index=player_index)
            obs = torch.tensor(obs).float().unsqueeze(0).to(device)
            beliefs_probs = softmax(belief_model(obs)).detach().numpy()[0]
            
            belief_role_assignments = all_possible_ordered_role_assignments(game_state.player_assignments)
                
            # condition on the fact that you know who the spies are
            belief = Belief(belief_role_assignments, beliefs_probs).condition_on(
                [Role.SPY if role == Role.SPY else Role.UNKNOWN for role in game_state.player_assignments]
            )
            
        elif role == Role.MERLIN:
            belief = Belief.trivial_distribution(
                game_state.player_assignments
            )
        else:
            raise ValueError(f"Unknown role: {role}")
        
        return belief