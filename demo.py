import torch
from medverse.lightning_model import LightningModel

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LightningModel.load_from_checkpoint("./Medverse.ckpt", map_location=device).to(device).eval()

# Dummy inputs
# NOTE: The spatial dimensions (H, W, D) of context and target must match
# target_in: [B, 1, H, W, D]
target_in = torch.randn(1, 1, 220, 220, 220, device=device)
# context_in/out: [B, K, 1, H, W, D], K = number of context samples
context_in = torch.randn(1, 3, 1, 220, 220, 220, device=device)
context_out = torch.randn(1, 3, 1, 220, 220, 220, device=device)

# Normalize
target_in = model.normalize_3d_volume(target_in)
context_in = model.normalize_3d_volume(context_in)
context_out = model.normalize_3d_volume(context_out)

# Inference (set forward_l_arg lower if VRAM is limited, min=1)
with torch.no_grad():
    mask = model.autoregressive_inference(target_in, context_in, context_out, forward_l_arg=1)
