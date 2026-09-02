"""Prompt-aware nnU-Net trainers for joint organ and tumor segmentation."""

from __future__ import annotations

from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class RestoreScalarPromptTransform(ImageOnlyTransform):
    """Restore the global 0/1 task code after spatial and intensity augmentation."""

    def get_parameters(self, **data_dict) -> dict:
        image = data_dict["image"]
        if image.shape[0] != 2:
            raise ValueError(f"expected CT+prompt, got {image.shape[0]} channels")
        return {"prompt_code": float(image[1].mean() >= 0.5)}

    def _apply_to_image(self, image: torch.Tensor, **params) -> torch.Tensor:
        image[1].fill_(params["prompt_code"])
        return image


class nnUNetTrainerPromptedFullCT(nnUNetTrainer):
    """Full-CT trainer with stronger foreground sampling and an intact prompt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oversample_foreground_percent = 0.66

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> ComposeTransforms:
        transforms = nnUNetTrainer.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label,
        )
        transforms.transforms.append(RestoreScalarPromptTransform())
        return transforms


class nnUNetTrainerPromptedFullCT_50epochs(nnUNetTrainerPromptedFullCT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 50


class nnUNetTrainerPromptedFullCT_100epochs(nnUNetTrainerPromptedFullCT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 100
