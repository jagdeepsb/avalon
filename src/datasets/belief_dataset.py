import numpy as np
import torch
from torch.utils.data import Dataset

class BeliefDataset(Dataset):
    def __init__(self, game_histories: np.ndarray, belief_distributions: np.ndarray):
        """
        game_histories: np.ndarray of shape (n_data_points, n_players, n_history_size, n_features)
        belief_distributions: np.ndarray of shape (n_data_points, distr_size)
        """
        
        # print(game_histories.shape)
        # print(game_histories[3450][2])
        # assert False
        
        assert game_histories.shape[0] == belief_distributions.shape[0]
        self.game_histories = game_histories
        self.belief_distributions = belief_distributions
        

    def __len__(self):
        return self.game_histories.shape[0]

    def __getitem__(self, idx):
        history, belief = self.game_histories[idx], self.belief_distributions[idx]
        
        # history = self.game_histories[0]
        # belief = np.zeros(30)
        # belief[idx] = 1
        
        history_tensor, belief_tensor = torch.tensor(history, dtype=torch.float32), torch.tensor(belief, dtype=torch.float32)
        return history_tensor, belief_tensor