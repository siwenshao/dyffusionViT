import torch
from src.models.koopmanvit import KoopmanViT
from src.datamodules.physical_systems_benchmark import PhysicalSystemsBenchmarkDataModule

# Load dataset
datamodule = PhysicalSystemsBenchmarkDataModule(
    data_dir="path/to/data",  # Update with actual dataset path
    physical_system="navier-stokes",
    window=1,
    horizon=5,
    prediction_horizon=5,
    num_test_obstacles=1,
)

datamodule.setup()
test_loader = datamodule.test_dataloader()

# Load KoopmanViT model
model = KoopmanViT(
    img_size=(221, 42),  # Matches Navier-Stokes data
    in_channels=3,  # 3 velocity components (u, v, pressure)
    patch_size=16,
    embed_dim=256,
    num_heads=8,
    depth=6,
    koopman_dim=128,
)

model.eval()  # Set to evaluation mode

# Test KoopmanViT
with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch["dynamics"], batch["targets"]
        preds = model(inputs)
        
        print("Predictions shape:", preds.shape)
        print("Targets shape:", targets.shape)
        break  # Only check first batch
