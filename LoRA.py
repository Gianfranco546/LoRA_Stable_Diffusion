from torch import nn
class LoRA(nn.Module):
    def __init__(self, r, input_dim, output_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, r, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(r, output_dim, bias = False)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x  = self.linear2(x)
        return x 
        
        