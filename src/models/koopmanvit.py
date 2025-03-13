import torch
import torch.nn as nn
import timm
from src.models._base_model import BaseModel  # Import BaseModel from DYffusion

class KoopmanViT(BaseModel):  # Inherits from BaseModel
    def __init__(self, img_size=(221, 42), patch_size=16, in_channels=3, embed_dim=256, 
                 num_heads=8, depth=6, koopman_dim=128, with_time_emb=False, dropout=0.0, **kwargs):  
        super().__init__(**kwargs)  # Pass kwargs to BaseModel
        self.with_time_emb = with_time_emb  
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

        # Decoder to reconstruct velocity fields with upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample again
            nn.Tanh()
        )
    
    def forward(self, x, time=None, **kwargs):  
        print(f"🔍 KoopmanViT received input shape: {x.shape}")  

        # Dynamically handle missing time_steps dimension
        if x.ndim == 4:  
            print("🚨 WARNING: Time dimension is missing! Adding time_steps=1")
            x = x.unsqueeze(1)  

        batch_size, time_steps, channels, height, width = x.shape
        x = x.view(batch_size * time_steps, channels, height, width)

        # Preprocess input for ViT (convert to 3-channel if needed)
        x = self.pre_vit(x)

        # Extract spatial features
        x = self.vit(x)  
        x = x.view(batch_size, time_steps, -1)  

        # Koopman evolution
        x = self.dropout(x)  
        koopman_latent = self.koopman_operator(x)
        evolved_latent = self.koopman_basis(koopman_latent)

        print(f"💡 KoopmanViT latent shape: {evolved_latent.shape}")  

        # Ensure reshaping is valid
        num_features = evolved_latent.shape[-1]
        evolved_latent = evolved_latent.view(batch_size * time_steps, num_features, 1, 1)  

        out = self.decoder(evolved_latent)

        # 🛠 Ensure correct output size
        out = nn.functional.interpolate(out, size=(height, width), mode="bilinear", align_corners=False)

        print(f"📏 Decoder output shape: {out.shape} | Expected: ({batch_size}, {time_steps}, {channels}, {height}, {width})")

        # Ensure reshaping is valid
        num_elements = batch_size * time_steps * channels * height * width
        if out.numel() != num_elements:
            print(f"🚨 WARNING: Decoder output has {out.numel()} elements, expected {num_elements}. Reshaping may be incorrect!")

        out = out.view(batch_size, time_steps, channels, height, width)

        return out
