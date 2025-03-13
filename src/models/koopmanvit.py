import torch
import torch.nn as nn
import timm
from src.models._base_model import BaseModel  # ✅ Import BaseModel from DYffusion

class KoopmanViT(BaseModel):  # ✅ Inherits from BaseModel
    def __init__(self, img_size=(221, 42), patch_size=16, in_channels=3, embed_dim=256, 
                 num_heads=8, depth=6, koopman_dim=128, with_time_emb=False, **kwargs):  # ✅ Add with_time_emb
        super().__init__(**kwargs)  # ✅ Pass kwargs to BaseModel
        self.with_time_emb = with_time_emb  # ✅ Store the parameter (but do nothing with it)

        self.save_hyperparameters()  # ✅ Store hparams (used in DYffusion)

        # Vision Transformer for spatial feature extraction
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads
        )
        
        # Koopman Operator
        self.koopman_operator = nn.Linear(embed_dim, koopman_dim, bias=False)
        self.koopman_basis = nn.Linear(koopman_dim, embed_dim)

        # Decoder to reconstruct velocity fields
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        x = x.view(batch_size * time_steps, channels, height, width)

        # Extract spatial features
        x = self.vit(x)
        x = x.view(batch_size, time_steps, -1)

        # Koopman evolution
        koopman_latent = self.koopman_operator(x)
        evolved_latent = self.koopman_basis(koopman_latent)

        # Reshape and decode
        evolved_latent = evolved_latent.view(batch_size * time_steps, -1, height, width)
        out = self.decoder(evolved_latent)
        out = out.view(batch_size, time_steps, channels, height, width)

        return out
