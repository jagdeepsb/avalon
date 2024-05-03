import torch
from torch import nn
import numpy as np
from einops.layers.torch import Rearrange

import torchsummary

def getPositionEncoding(seq_len, d, n=10000) -> torch.Tensor:
    """
    Returns a matrix of shape (seq_len, d) where each row is the
    position encoding of the corresponding row index.
    """
    positional_encoding = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            positional_encoding[k, 2*i] = np.sin(k/denominator)
            positional_encoding[k, 2*i+1] = np.cos(k/denominator)
    positional_encoding = torch.tensor(positional_encoding, dtype=torch.float32)
    return positional_encoding
 
class BeliefPredictor(nn.Module):
    def __init__(
        self,
        encoding_dim: int = 16,
        n_players: int = 5,
        n_history_length: int = 25,
        feature_dim: int = 10,
    ):
        super(BeliefPredictor, self).__init__()
        self.encoding_conv = nn.Conv2d(
            in_channels=feature_dim,
            out_channels=encoding_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        
    def forward(self, x: torch.Tensor):
        
        x = self.encoding_conv(x)
        return x
 
if __name__ == "__main__":
 
    # P = getPositionEncoding(seq_len=4, d=4, n=100)
    # print(P)
    
    model = BeliefPredictor()
    
    torchsummary.summary(model, [(5, 25, 10)])