import torch
import logging

from src.model import ACTORStyleEncoder

logger = logging.getLogger(__name__)

class TMR(torch.nn.Module):
    def __init__(
        self,
        latent_dim=256,
        pretrained=True
    ):
        super().__init__()
        
        self.model = ACTORStyleEncoder(
            nfeats=263,
            vae=True,
            latent_dim=256,
            ff_size=1024,
            num_layers=6,
            num_heads=4,
            dropout=0.1,
            activation="gelu"
        )
        
        self.latent_dim = latent_dim
        
        if pretrained:
            self.model.load_state_dict(torch.load("/home/nadir/tmr-code/models/tmr_humanml3d_guoh3dfeats/last_weights/motion_encoder.pt"))
            
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.model.eval()
        
    def forward(self, batch):
        # preprocessed_motion, motion, transition_mask = batch
        preprocessed_motion = batch["transformed_motion"]
        motion = batch["motion"]
        transition_mask = batch["annotation"]
        
        encoded = self.model({
            "x": preprocessed_motion.float(),
            # Shape: [batch_size, time]
            "mask": torch.ones(preprocessed_motion.shape[:2], dtype=torch.bool, device=preprocessed_motion.device)
        })
        
        dists = encoded.unbind(1)
        mu, logvar = dists
            
        return mu