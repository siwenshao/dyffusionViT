import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import VisionTransformer

class KoopmanViT(nn.Module):
    def __init__(self, img_size=(221, 42), patch_size=16, in_channels=3, embed_dim=256, num_heads=8, depth=6, koopman_dim=128,  **kwargs):
        super().__init__()
        
        # Vision Transformer for spatial feature extraction
        self.vit = VisionTransformer(
            image_size=img_size, patch_size=patch_size, num_layers=depth,
            num_heads=num_heads, hidden_dim=embed_dim, mlp_dim=embed_dim * 4, num_classes=0
        )
        
        # Koopman Operator
        self.koopman_operator = nn.Linear(embed_dim, koopman_dim, bias=False)
        self.koopman_basis = nn.Linear(koopman_dim, embed_dim)
        
        # Decoder to reconstruct velocity fields
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Velocity fields can be negative
        )
    
    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        x = x.view(batch_size * time_steps, channels, height, width)
        
        # Extract spatial features
        x = self.vit(x)  # (batch*time_steps, embed_dim)
        x = x.view(batch_size, time_steps, -1)  # Reshape back to time series format
        
        # Koopman evolution
        koopman_latent = self.koopman_operator(x)
        evolved_latent = self.koopman_basis(koopman_latent)
        
        # Reshape and decode
        evolved_latent = evolved_latent.view(batch_size * time_steps, -1, height, width)
        out = self.decoder(evolved_latent)
        out = out.view(batch_size, time_steps, channels, height, width)
        
        return out
