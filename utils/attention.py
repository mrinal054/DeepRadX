import torch
import torch.nn as nn

class BasicAttention(nn.Module):
    def __init__(self, feature_dim):
        super(BasicAttention, self).__init__()
        self.attention_fc = nn.Linear(feature_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention_weights = self.sigmoid(self.attention_fc(x))
        return x * attention_weights  # eElement-wise multiplication with attention weights
