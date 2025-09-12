# Universal In-Context Learning for 3D Medical Imaging


Official PyTorch implementation of **Medverse: A Universal Model for Full-Resolution 3D Medical Image Segmentation, Transformation and Enhancement**  

[Paper]((https://arxiv.org/abs/2509.09232).



## 🚀 Running the Model

### 1. Environment Setup

Make sure you have Python and PyTorch installed. All required libraries are listed in the `requirements.txt` file. You can install them using:

```bash
pip install -r requirements.txt
```

### 2. 🖥️ Usage

#### Load a Pretrained Model
Download the model weight [here](https://drive.google.com/file/d/1pycz24JidkOspz2qyUsC1xjfEPBSfQsK/view?usp=sharing).
```python
import torch
from medverse.lightning_model import LightningModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LightningModel.load_from_checkpoint("./Medverse.ckpt", map_location=device).to(device).eval()

```
#### Run Inference
```python
# Dummy inputs
# NOTE: The spatial dimensions (H, W, D) of context and target must match
target_in = torch.randn(1, 1, 220, 220, 220, dtype=torch.float32)  # Target image, shape [batch_size, 1, H, W, D]
context_in = torch.randn(1, 3, 1, 220, 220, 220, dtype=torch.float32)  # Context set images, shape [batch_size, context_size, 1, H, W, D]
context_out = torch.randn(1, 3, 1, 220, 220, 220, dtype=torch.float32)  # Context set labels, shape [batch_size, context_size, 1, H, W, D]

# Normalize
target_in = model.normalize_3d_volume(target_in)
context_in = model.normalize_3d_volume(context_in)
context_out = model.normalize_3d_volume(context_out)

# Inference
with torch.no_grad():
    mask = model.autoregressive_inference(target_in,
                                          context_in,
                                          context_out,
                                          forward_l_arg=1, # min-context size. Lower if GPU memory is limited, min=1.
                                         )
```

## Acknowledgements
This repository is modified from [Neuroverse3D](https://github.com/jiesihu/Neuroverse3D).

