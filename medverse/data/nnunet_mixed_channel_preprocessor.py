"""nnU-Net v2 preprocessor for CT plus a binary organ-mask input channel."""

from __future__ import annotations

from typing import Union

import numpy as np

from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.resampling.default_resampling import (
    compute_new_shape,
    resample_data_or_seg_to_shape,
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class MixedChannelPreprocessor(DefaultPreprocessor):
    """Use continuous interpolation for CT and nearest-neighbor for the organ channel."""

    def run_case_npy(
        self,
        data: np.ndarray,
        seg: Union[np.ndarray, None],
        properties: dict,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str],
    ):
        if isinstance(dataset_json, str):
            import json

            with open(dataset_json, encoding="utf-8") as stream:
                dataset_json = json.load(stream)
        data = data.astype(np.float32)
        if data.shape[0] != 2:
            raise ValueError(f"MixedChannelPreprocessor requires CT+organ channels, got {data.shape[0]}")
        has_seg = seg is not None
        if has_seg:
            if data.shape[1:] != seg.shape[1:]:
                raise ValueError("image/organ/tumor shapes differ before preprocessing")
            seg = np.copy(seg)

        order = [0, *[axis + 1 for axis in plans_manager.transpose_forward]]
        data = data.transpose(order)
        if has_seg:
            seg = seg.transpose(order)
        original_spacing = [properties["spacing"][axis] for axis in plans_manager.transpose_forward]
        properties["shape_before_cropping"] = data.shape[1:]
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties["bbox_used_for_cropping"] = bbox
        properties["shape_after_cropping_and_before_resampling"] = data.shape[1:]

        target_spacing = list(configuration_manager.spacing)
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0], *target_spacing]
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # Channel 0 uses the planned CT normalization. Channel 1 is configured
        # as NoNormalization through dataset.json and remains binary here.
        data = self._normalize(
            data,
            seg,
            configuration_manager,
            plans_manager.foreground_intensity_properties_per_channel,
        )
        ct = configuration_manager.resampling_fn_data(
            data[:1], new_shape, original_spacing, target_spacing
        )
        organ = resample_data_or_seg_to_shape(
            data[1:2],
            new_shape,
            original_spacing,
            target_spacing,
            is_seg=True,
            order=0,
            order_z=0,
        )
        organ = (organ > 0.5).astype(np.float32, copy=False)
        data = np.concatenate((ct.astype(np.float32, copy=False), organ), axis=0)
        seg = configuration_manager.resampling_fn_seg(
            seg, new_shape, original_spacing, target_spacing
        )

        if has_seg:
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect = (
                list(label_manager.foreground_regions)
                if label_manager.has_regions
                else list(label_manager.foreground_labels)
            )
            if label_manager.has_ignore_label:
                collect.append([-1, *label_manager.all_labels])
            properties["class_locations"] = self._sample_foreground_locations(
                seg, collect, verbose=self.verbose
            )
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        seg = seg.astype(np.int16 if np.max(seg) > 127 else np.int8, copy=False)
        return data, seg, properties
