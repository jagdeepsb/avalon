import torch
from torch import nn
import numpy as np
from einops.layers.torch import Rearrange
# from einops import rearrange as Rearrange

from torchinfo import summary

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

class SelfAttentionTransformer(nn.Module):
    def __init__(
        self,
        n_players: int,
        n_history_length: int, 
        embedding_dim: int, num_heads: int, dropout: float
    ):
        super(SelfAttentionTransformer, self).__init__()
        
        self.n_players, self.n_history_length = n_players, n_history_length
        
        self.multihead_attn_1 = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.multihead_attn_2 = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        
        self.conv_1 = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.conv_2 = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.relu = nn.ReLU()
        
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x: torch.Tensor):
        """
        Input shape: (batch_size, n_players, n_history, feature_dim)
        Output shape: (batch_size, n_players, n_history, feature_dim)
        """
        
        # Attention + LayerNorm + Residual
        x_1 = Rearrange("b n_players n_history d -> b (n_players n_history) d")(x)
        out, attn_matrix = self.multihead_attn_1(x_1, x_1, x_1)
        x_2 = x_1 + self.layer_norm_1(out)
        
        # print(attn_matrix.shape)
        # print(attn_matrix[0, 0:5, 0:5])
        # assert False
        
        # Attention + LayerNorm + Residual
        out, attn_matrix = self.multihead_attn_2(x_2, x_2, x_2)
        out = out + self.layer_norm_2(out)
        
        # Feedforward + Residual
        x_3 = Rearrange(
            "b (n_players n_history) d -> b d n_players n_history",
            n_players=self.n_players,
            n_history=self.n_history_length
        )(out)

        out = self.conv_1(x_3)
        out = self.relu(out)
        out = x_3 + self.conv_2(out)
        
        out = Rearrange("b d n_players n_history -> b n_players n_history d")(out)
        return out


class BeliefPredictor(nn.Module):
    def __init__(
        self,
        encoding_dim: int,
        n_classes: int,
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
        self.positional_encoding_for_players = getPositionEncoding(n_players, encoding_dim)
        self.positional_encoding_for_history = getPositionEncoding(n_history_length, encoding_dim)
        
        
        self.round_conv_1 = nn.Conv2d(
            in_channels=encoding_dim,
            out_channels=encoding_dim,
            kernel_size=(n_players, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        
        self.round_conv_2 = nn.Conv2d(
            in_channels=encoding_dim,
            out_channels=encoding_dim,
            kernel_size=(n_players, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        
        self.player_conv_1 = nn.Conv2d(
            in_channels=encoding_dim,
            out_channels=encoding_dim,
            kernel_size=(1, n_history_length),
            stride=(1, 1),
            padding=(0, 0),
        )
        
        self.player_conv_2 = nn.Conv2d(
            in_channels=encoding_dim,
            out_channels=encoding_dim,
            kernel_size=(1, n_history_length),
            stride=(1, 1),
            padding=(0, 0),
        )
        
        self.relu = nn.ReLU()
        
        self.decoding = nn.Linear(n_players * encoding_dim, n_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor):
        """
        Input shape: (batch_size, n_players, n_history, feature_dim)
        """
        
        # encoding
        out = Rearrange("b n_players n_history d -> b d n_players n_history")(x)
        out = self.encoding_conv(out)
        out = Rearrange("b d n_players n_history -> b n_history n_players d")(out)
        out = out + self.positional_encoding_for_players
        out = Rearrange("b n_history n_players d -> b n_players n_history d")(out)
        out = out + self.positional_encoding_for_history
        x_1 = Rearrange("b n_players n_history d -> b d n_players n_history")(out)
        
        # operations
        out = self.round_conv_1(x_1)
        x_2 = self.relu(out) + x_1 # residual connection
        out = self.player_conv_1(x_2)
        x_3 = self.relu(out) + x_2 # residual connection
        out = self.round_conv_2(x_3)
        x_4 = self.relu(out) + x_3 # residual connection
        out = self.player_conv_2(x_4)
        x_5 = self.relu(out) + x_4 # residual connection
        
        out = torch.einsum("bdph->bdp", x_5)
        
        # decoding
        out = nn.Flatten()(out)
        out = self.decoding(out)
        
        return out

# class BeliefPredictor(nn.Module):
#     """
#     Epoch 9: Avg Loss 0.41714328616785196 Validation Loss 4.3838596116558834
#     Overfit like crazy on train, no interesting progress on val
#     """
#     def __init__(
#         self,
#         encoding_dim: int,
#         n_classes: int,
#         n_players: int = 5,
#         n_history_length: int = 25,
#         feature_dim: int = 10,
#     ):
#         super(BeliefPredictor, self).__init__()
        
#         hidden_layer_dim = 256
#         self.layers = nn.Sequential(
#             nn.Linear(n_players * n_history_length * feature_dim, hidden_layer_dim),
#             nn.GELU(),
#             nn.Linear(hidden_layer_dim, hidden_layer_dim),
#             nn.GELU(),
#             nn.Linear(hidden_layer_dim, n_classes),
#         )
        
#     def forward(self, x: torch.Tensor):
#         """
#         Input shape: (batch_size, n_players, n_history, feature_dim)
#         """
        
#         x = nn.Flatten()(x)
#         x = self.layers(x)
#         return x
 
# class BeliefPredictor(nn.Module):
#     def __init__(
#         self,
#         encoding_dim: int,
#         n_classes: int,
#         n_players: int = 5,
#         n_history_length: int = 25,
#         feature_dim: int = 10,
#     ):
#         super(BeliefPredictor, self).__init__()
#         self.encoding_conv = nn.Conv2d(
#             in_channels=feature_dim,
#             out_channels=encoding_dim,
#             kernel_size=(1, 1),
#             stride=(1, 1),
#             padding=(0, 0),
#         )
#         self.positional_encoding_for_players = getPositionEncoding(n_players, encoding_dim)
#         self.positional_encoding_for_history = getPositionEncoding(n_history_length, encoding_dim)
        
#         self.transformer_1 = SelfAttentionTransformer(
#             n_players=n_players,
#             n_history_length=n_history_length,
#             embedding_dim=encoding_dim,
#             num_heads=4,
#             dropout=0.1
#         )
#         self.transformer_2 = SelfAttentionTransformer(
#             n_players=n_players,
#             n_history_length=n_history_length,
#             embedding_dim=encoding_dim,
#             num_heads=4,
#             dropout=0.1
#         )
#         self.transformer_2 = SelfAttentionTransformer(
#             n_players=n_players,
#             n_history_length=n_history_length,
#             embedding_dim=encoding_dim,
#             num_heads=4,
#             dropout=0.1
#         )
        
        
#         # self.decoding = nn.Linear(n_players * n_history_length * encoding_dim, n_classes)
#         self.decoding = nn.Linear(n_players * n_history_length * feature_dim, n_classes)
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, x: torch.Tensor):
#         """
#         Input shape: (batch_size, n_players, n_history, feature_dim)
#         """
        
#         # encoding
#         # x = Rearrange("b n_players n_history d -> b d n_players n_history")(x)
#         # x = self.encoding_conv(x)
#         # x = Rearrange("b d n_players n_history -> b n_history n_players d")(x)
#         # x = x + self.positional_encoding_for_players
#         # x = Rearrange("b n_history n_players d -> b n_players n_history d")(x)
#         # x = x + self.positional_encoding_for_history
#         # x = Rearrange("b n_players n_history d -> b n_players n_history d")(x)
        
#         # transformers
#         # x = self.transformer_1(x)
        
#         # decoding
#         x = nn.Flatten()(x)
#         x = self.decoding(x)
#         # x = self.softmax(x)
#         return x
 
if __name__ == "__main__":   
    model = BeliefPredictor(encoding_dim=16, n_classes=30)
    summary(model, (32, 5, 25, 10))
    
    x = torch.randn(32, 5, 25, 10)
    y = model(x)
    print(y.shape)
    print(y[0])