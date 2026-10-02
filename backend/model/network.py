import torch.nn as nn

# ========= Neural Network Model =========

class PersistenceNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self,x):
        return self.layers(x)