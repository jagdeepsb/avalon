import time
import torch
from torch import nn
from tqdm import tqdm
from torchsummary import summary

class SelfAttentionTransformer(nn.Module):
    def __init__(self,):
        super(SelfAttentionTransformer, self).__init__()
        self.multihead_attn_1 = nn.MultiheadAttention(
            embed_dim=10,
            num_heads=2,
            dropout=0.1
        )
        self.multihead_attn_2 = nn.MultiheadAttention(
            embed_dim=10,
            num_heads=2,
            dropout=0.1
        )
        self.layer_norm_1 = nn.LayerNorm(10)
        self.layer_norm_2 = nn.LayerNorm(10)
        self.layer_norm_3 = nn.LayerNorm(10)
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2)
        out, _ = self.multihead_attn_1(x, x, x)
        out = x + self.layer_norm_1(out)
        
        out, _ = self.multihead_attn_2(out, out, out)
        out = out + self.layer_norm_2(out)
        
        out = self.linear(out)
        out = out + self.layer_norm_3(out)
        out = out.permute(1, 0, 2)
        return out

if __name__ == "__main__":
    # transformer_model = nn.Transformer(
    #     nhead=2,
    #     num_encoder_layers=12,
    #     d_model=10,
    #     batch_first=True
    # )
    # src = torch.rand((32, 125, 10))
    # tgt = torch.rand((32, 125, 10))
    
    # start_time = time.time()
    # for i in tqdm(range(100)):
    #     out = transformer_model(src, tgt)
    # print(f"Its/second: {100 / (time.time() - start_time)}")
    
    # print(out.shape)
    
    x = torch.rand((32, 125, 10))
    model = SelfAttentionTransformer()
    start_time = time.time()
    for i in tqdm(range(100)):
        out = model(x)
    print(f"Its/second: {100 / (time.time() - start_time)}")
    print(out.shape)
    
    summary(model, [(125, 10)], depth=5)
    
    
    pass