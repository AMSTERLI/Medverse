# Universal In-Context Learning for 3D Medical Imaging


Official PyTorch implementation of **Medverse: A Universal Model for Full-Resolution 3D Medical Image Segmentation, Transformation and Enhancement**  

[Paper](https://arxiv.org/abs/2509.09232).



## 🚀 Running the Model

### 1. Environment Setup

* **Option 1: Using uv (Recommended)**
    This project uses  [uv](https://github.com/astral-sh/uv) for dependency management. :
    ```bash
    # Install uv
    pip install uv
    
    # Sync dependencies (creates a virtual environment and installs packages)
    uv sync
    
    # Activate the environment
    source .venv/bin/activate
    ```

* **Option 2: Using pip**
    You can also install the dependencies listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Usage

#### Load the Pretrained Model
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
                                          forward_l_arg=1, # min-context size. Lower if GPU memory is limited, min=1. No effect on the results.
                                         )
```

## Acknowledgements
This repository was modified from [Neuroverse3D](https://github.com/jiesihu/Neuroverse3D).

## Pan-cancer segmentation MVP

This working copy also contains a segmentation-only `PanCancerMedverse` and a
BCSI-inspired Channel-selective Context-Target Interaction (CCTI) experiment.
The channel ablation modes (`none`, `all`, `random`, `learned`) and the default
MVP settings are documented in `configs/pan_cancer_ccti_mvp.yaml`.  Run
`python scripts/smoke_test_ccti.py` for a small standalone check.

PAOT2 CT pair lists can be converted into organ-aware binary ICL episodes with
`scripts/prepare_paot2_manifest.py`.  The complete organ/label mapping and the
minimal train/evaluate commands are documented in `docs/paot2_training.md`.

## Liver/kidney CT main experiment

The reproducible LiTS + KiTS23 + MSWAL experiment (shared TotalSegmentator ROI,
nnU-Net v2 No-ICL versus PanCancerMedverse ICL + original BAM, CCTI disabled) is
documented in Chinese in `docs/liver_kidney_icl_experiment_zh.md`. Formal
training is protected by separate smoke-test and 2–4-case overfit gates and the
launcher never starts more than one process per GPU.

