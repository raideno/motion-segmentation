import torch
import torch.nn as nn

class MLPClassifier(torch.nn.Module):
    def __init__(
        self,
        latent_dim=256,
        hidden_dim=128,
        num_classes=20
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        
        self.start_regression_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        self.end_regression_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # NOTE: x is of shape (B, LatentDimension)
        B, T = x.size()
        
        class_logits = self.classification_head(x)
        start_logits = self.start_regression_head(x)
        end_logits = self.end_regression_head(x)
        
        return class_logits, start_logits, end_logits