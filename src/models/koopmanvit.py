import torch
import torch.nn as nn
import timm
from src.models._base_model import BaseModel  # Import BaseModel from DYffusion

class KoopmanViT(BaseModel):  # Inherits from BaseModel
    def __init__(self, img_size=(221, 42), patch_size=16, in_channels=3, embed_dim=256, 
                 num_heads=8, depth=6, koopman_dim=128, with_time_emb=False, dropout=0.0, **kwargs):  # ✅ Add with_time_emb
        super().__init__(**kwargs)  # Pass kwargs to BaseModel
        self.with_time_emb = with_time_emb  # Store the parameter (but do nothing with it)
        
# In UNet-based models, with_time_emb=True means the model:

# Adds a time encoding to inputs.
# Improves temporal generalization by learning embeddings for time steps.
# In KoopmanViT:

# Time information is handled differently (via Koopman evolution).
# Koopman theory models continuous time evolution explicitly, so adding extra time embeddings might be redundant.

        self.save_hyperparameters()  # Store hparams (used in DYffusion)

        self.dropout = nn.Dropout(dropout) 
        
        # Convert input channels if necessary
        self.pre_vit = nn.Conv2d(in_channels, 3, kernel_size=1) if in_channels != 3 else nn.Identity()

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
            self.dropout,
            nn.Tanh()
        )
    
    def forward(self, x, time=None, **kwargs):  # Accept 'time' argument
        print(f"KoopmanViT received input of shape: {x.shape}")  # ✅ Debugging step
        batch_size = x.shape[0]

        # Check if time_steps is missing
        if x.ndim == 4:
            print("🚨 WARNING: Time dimension is missing! Expected 5D input.")
            time_steps = 1
            x = x.unsqueeze(1)  # Temporary fix if needed
        else:
            time_steps = x.shape[1]  # ✅ Correctly get the number of time steps
        
        batch_size, time_steps, channels, height, width = x.shape
        x = x.view(batch_size * time_steps, channels, height, width)
        
        # Preprocess input for ViT (convert to 3-channel if needed)
        x = self.pre_vit(x)
        
        # Extract spatial features
        x = self.vit(x)  # (batch*time_steps, embed_dim)
        x = x.view(batch_size, time_steps, -1)  # Reshape back to time series format
        
        # Koopman evolution
        x = self.dropout(x)  # Apply dropout before Koopman evolution
        koopman_latent = self.koopman_operator(x)
        evolved_latent = self.koopman_basis(koopman_latent)
        
        # Reshape and decode
        evolved_latent = evolved_latent.view(batch_size * time_steps, -1, height, width)
        out = self.decoder(evolved_latent)
        out = out.view(batch_size, time_steps, channels, height, width)
        
        return out
