import pytorch_lightning as pl
from .util.shapecheck import ShapeChecker
import torch
from torch.nn import functional as F
import numpy as np
import scipy.ndimage # Added import
import random

import pytorch_lightning as pl
from .models.Medverse import Medverse
from .util.shapecheck import ShapeChecker
from .models.attention_utils.patch_utils import calculate_intersection_matrix

from monai.losses import DiceLoss
from monai.inferers.utils import sliding_window_inference
from monai.utils import BlendMode, PytorchPadMode

# For AR Inference
import itertools
from typing import List, Optional, Sequence, Tuple


class LightningModel(pl.LightningModule):
    """
    We use pytorch lightning to organize our model code
    """

    def __init__(self, hparams):
        super().__init__()
        # save hparams / load hparams
        self.save_hyperparameters(hparams)
        
        # build model
        self.net = Medverse(in_channels = 1, 
                    out_channels = 1, 
                    stages = len(self.hparams.nb_inner_channels), 
                    dim = 2 if self.hparams.data_slice_only else 3,
                    inner_channels = self.hparams.nb_inner_channels,
                    conv_layers_per_stage = self.hparams.nb_conv_layers_per_stage)
        
        
    def forward(self, 
                target_in, 
                context_in = None, # BxLxCinxHxWxD for semantic context
                context_out = None,  # BxLxCinxHxWxD for semantic context
                image_context_in = None, # BxLxCinxHxWxD for NA-ICL input
                image_context_out = None, # BxLxCinxHxWxD for NA-ICL input
                l=3, # Mini-context size for adaptive sequential-parallel context processing
               ):
        
        
            
            
        # run network
        y_pred = self.net(target_in.to(self.device), 
                          context_in=context_in, 
                          context_out=context_out, 
                          image_context_in = image_context_in, # BxLxCinxHxWxD
                          image_context_out = image_context_out, # BxLxCinxHxWxD
                          l = l,
                         )
        return y_pred
    
    
            
    
    
    def normalize_3d_volume(self, target_in: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normalizes a batch of 3D volumes (shape: [..., D, H, W]) to the range [0, 1] independently for each sample.

        Args:
            target_in (torch.Tensor): Input tensor of shape [..., 1, D, H, W], where N is the batch size,
                                      and (D, H, W) are the depth, height, and width of the 3D volume.
            eps (float): A small value to prevent division by zero. Default is 1e-8.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as the input, where each sample is scaled to [0, 1].
        """
        # Ensure the input tensor is of type float32 to avoid integer division issues
        if target_in.dtype != torch.float32:
            target_in = target_in.to(torch.float32)

        # Compute the minimum and maximum values for each sample independently
        # Input shape: [N, 1, D, H, W] → Output shape: [N, 1, 1, 1, 1]
        min_vals = torch.amin(target_in, dim=(-3, -2, -1), keepdim=True)
        max_vals = torch.amax(target_in, dim=(-3, -2, -1), keepdim=True)

        # Compute the dynamic range and prevent division by zero
        dynamic_range = max_vals - min_vals
        dynamic_range[dynamic_range < eps] = eps  # Replace small ranges with eps to avoid division by zero

        # Normalize the input tensor to the range [0, 1]
        normalized = (target_in - min_vals) / dynamic_range

        return normalized
    


    
    def _sliding_window_autoregressive_step(self,
                                            current_target_in,
                                            current_context_in,
                                            current_context_out,
                                            image_context_in_prev,
                                            roi_size, 
                                            sw_batch_size,
                                            overlap,
                                            mode, 
                                            forward_l_param
                                            ):
        """
        
        Helper function to perform a single step of sliding window inference for the autoregressive process.

        This function prepares inputs by stacking them, defines a predictor for MONAI's sliding_window_inference,
        and then invokes the sliding window inference.

        Args:
            current_target_in (torch.Tensor): The target input for the current resolution level.
                                              Shape: [B, C_t, D_iter, H_iter, W_iter]
            current_context_in (torch.Tensor, optional): The semantic context input for the current resolution level.
                                                       Shape: [B, L_c, C_c, D_iter, H_iter, W_iter]
            current_context_out (torch.Tensor, optional): The semantic context output for the current resolution level.
                                                        Shape: [B, L_co, C_co, D_iter, H_iter, W_iter]
            image_context_in_prev (torch.Tensor, optional): The image context (output from the previous autoregressive level).
                                                          Shape: [B, 1, C_ic, D_iter, H_iter, W_iter]
            roi_size (tuple or list): The spatial size of the ROI for sliding window.
            sw_batch_size (int): The batch size for sliding window inference.
            overlap (float): The overlap ratio for sliding window.
            mode (BlendMode): The blending mode for overlapping windows (e.g., BlendMode.GAUSSIAN).
            forward_l_param (int): The 'l' parameter to be passed to the model's forward method.

        Returns:
            torch.Tensor: The prediction output from the sliding window inference for the current level.
        """
        if hasattr(self, 'concat_global_crop') and self.concat_global_crop:
            self.current_context_in_resized = torch.nn.functional.interpolate(current_context_in.squeeze(2), size=(128, 128, 128), mode='trilinear', align_corners=False).unsqueeze(2)
            self.current_context_out_resized = torch.nn.functional.interpolate(current_context_out.squeeze(2), size=(128, 128, 128), mode='trilinear', align_corners=False).unsqueeze(2)


        B_main = current_target_in.shape[0]  # Batch size for the main input (target_in)
        spatial_shape_iter = current_target_in.shape[2:] # Spatial dimensions for the current iteration
        
        inputs_to_stack = []
        # Metadata to help reconstruct original tensor shapes and types from the stacked tensor within the predictor
        channel_meta = {
            "C_target": 0,      # Number of channels in target_in
            "original_L_c": 0,  # Original L dimension of context_in (number of context samples)
            "C_c": 0,           # Number of channels per context_in sample
            "original_L_co": 0, # Original L dimension of context_out
            "C_co": 0,          # Number of channels per context_out sample
            "C_ic": 0           # Number of channels in image_context_in_prev
        }
        flat_channel_counts = [] # Stores the number of channels for each input type *after* potential reshaping (e.g. L*C for contexts)

        # Stack target_in
        inputs_to_stack.append(current_target_in.to(self.device))
        channel_meta["C_target"] = current_target_in.shape[1]
        flat_channel_counts.append(channel_meta["C_target"])

        # Stack context_in if provided
        if current_context_in is not None:
            channel_meta["original_L_c"] = current_context_in.shape[1]
            channel_meta["C_c"] = current_context_in.shape[2]
            # Reshape context_in: [B, L, C, D, H, W] -> [B, L*C, D, H, W] for stacking
            reshaped_context_in = current_context_in.reshape(B_main, -1, *spatial_shape_iter)
            inputs_to_stack.append(reshaped_context_in.to(self.device))
            flat_channel_counts.append(reshaped_context_in.shape[1])
        else:
            flat_channel_counts.append(0) # Placeholder if not present

        # Stack context_out if provided
        if current_context_out is not None:
            channel_meta["original_L_co"] = current_context_out.shape[1]
            channel_meta["C_co"] = current_context_out.shape[2]
            # Reshape context_out: [B, L, C, D, H, W] -> [B, L*C, D, H, W] for stacking
            reshaped_context_out = current_context_out.reshape(B_main, -1, *spatial_shape_iter)
            inputs_to_stack.append(reshaped_context_out.to(self.device))
            flat_channel_counts.append(reshaped_context_out.shape[1])
        else:
            flat_channel_counts.append(0) # Placeholder if not present

        # Stack image_context_in_prev if provided
        if image_context_in_prev is not None:
            channel_meta["C_ic"] = image_context_in_prev.shape[2]
            # Reshape image_context_in: [B, 1, C_ic, D, H, W] -> [B, C_ic, D, H, W] for stacking
            reshaped_img_ctx_in = image_context_in_prev.reshape(B_main, -1, *spatial_shape_iter)
            inputs_to_stack.append(reshaped_img_ctx_in.to(self.device))
            flat_channel_counts.append(reshaped_img_ctx_in.shape[1])
        else:
            flat_channel_counts.append(0) # Placeholder if not present
        
        # Concatenate all inputs along the channel dimension
        stacked_inputs = torch.cat(inputs_to_stack, dim=1)

        # Memoize metadata for use in the nested predictor function (to avoid issues with closures in loops if any)
        memoized_channel_meta = channel_meta
        memoized_flat_channel_counts = flat_channel_counts

        # Define the predictor function that will be called by sliding_window_inference for each window
        def _predictor_for_sliding_window(data_window, network_l_param_inner):
            """ 
            This nested function is called by sliding_window_inference on each window (patch) of the stacked_inputs.
            It unnpacks the window data back into individual model inputs and calls self.forward.
            Args:
                data_window (torch.Tensor): A window from the stacked_inputs.
                                           Shape: [sw_batch_size, C_stacked, D_roi, H_roi, W_roi]
                network_l_param_inner (int): The 'l' parameter for self.forward.
            Returns:
                Output of self.forward for the window.
            """
            current_idx = 0
            # Extract target_window
            target_window = data_window[:, current_idx : current_idx + memoized_flat_channel_counts[0], ...]
            current_idx += memoized_flat_channel_counts[0]

            # Extract and reshape context_in_window if it was part of the stack
            actual_context_in = None
            if memoized_flat_channel_counts[1] > 0:
                context_in_window_flat = data_window[:, current_idx : current_idx + memoized_flat_channel_counts[1], ...]
                actual_context_in = context_in_window_flat.reshape(
                    context_in_window_flat.shape[0], # sw_batch_size
                    memoized_channel_meta["original_L_c"], 
                    memoized_channel_meta["C_c"], 
                    *context_in_window_flat.shape[2:] # roi spatial dimensions
                )
            current_idx += memoized_flat_channel_counts[1]

            # Extract and reshape context_out_window if it was part of the stack
            actual_context_out = None
            if memoized_flat_channel_counts[2] > 0:
                context_out_window_flat = data_window[:, current_idx : current_idx + memoized_flat_channel_counts[2], ...]
                actual_context_out = context_out_window_flat.reshape(
                    context_out_window_flat.shape[0], # sw_batch_size
                    memoized_channel_meta["original_L_co"], 
                    memoized_channel_meta["C_co"], 
                    *context_out_window_flat.shape[2:] # roi spatial dimensions
                )
            current_idx += memoized_flat_channel_counts[2]

            # Extract and reshape image_context_in_window if it was part of the stack
            actual_img_ctx_in = None
            actual_img_ctx_out = None
            if memoized_flat_channel_counts[3] > 0:
                img_ctx_in_window_flat = data_window[:, current_idx : current_idx + memoized_flat_channel_counts[3], ...]
                # Reshape to [B_window, 1, C_model_out, D_roi, H_roi, W_roi]
                # img_ctx_in_window_flat is [B_window, C_model_out, D_roi, H_roi, W_roi]
                actual_img_ctx_out = img_ctx_in_window_flat[:,0:1,:].unsqueeze(1)
                actual_img_ctx_in = img_ctx_in_window_flat[:,1:2,:].unsqueeze(1)
                
            
            if hasattr(self, 'concat_global_crop') and self.concat_global_crop:
                actual_context_in = torch.cat([actual_context_in, self.current_context_in_resized.to(self.device)], dim=1)
                actual_context_out = torch.cat([actual_context_out, self.current_context_out_resized.to(self.device)], dim=1)
                
            
            mask = self.forward(target_in=target_window,
                                context_in=actual_context_in,
                                context_out=actual_context_out,
                                image_context_in=actual_img_ctx_in,
                                image_context_out=actual_img_ctx_out,
                                l=network_l_param_inner
                               )
            # ----plot image
            if self.verbose:
                self.plot_fig(actual_context_in[0], actual_context_out[0], slice_ = 64, name = 'Context', colorbar = True)
                if actual_img_ctx_in is not None:
                    self.plot_pred_fig(target_window, actual_img_ctx_in[0], actual_img_ctx_out[0], slice_ = 64, name = 'PredImgInImgOut', colorbar = True)
                self.plot_pred_fig(target_window, mask, mask, slice_ = 64, name = 'PredOutOut', colorbar = True)

            # Call the model's forward pass with the unpacked and reshaped window inputs
            return mask

        # Perform sliding window inference using MONAI's utility
        output_pred = sliding_window_inference(
            inputs=stacked_inputs, 
            roi_size=roi_size, 
            sw_batch_size=sw_batch_size,
            predictor=_predictor_for_sliding_window, # Custom predictor defined above
            overlap=overlap, 
            mode=mode,
            sigma_scale=0.125, # Standard for Gaussian blending
            padding_mode=PytorchPadMode.CONSTANT, 
            cval=0.0,
            sw_device=self.device, # Device for window operations
            device=self.device,    # Device for the final output
            progress=False, 
            network_l_param_inner=forward_l_param # Pass the 'l' for forward to the predictor
        )
        return output_pred
    

    def autoregressive_inference(self, 
                                 target_in, 
                                 context_in=None, 
                                 context_out=None, 
                                 level=None, 
                                 forward_l_arg=3, 
                                 sw_roi_size=(128,128,128),
                                 sw_overlap=0.1,
                                 sw_blend_mode=BlendMode.GAUSSIAN,
                                 sw_batch_size_val=1,
                                 intensity_clip = True,
                                 verbose = False):
        """
        Performs autoregressive inference by iteratively refining predictions across multiple resolution levels.

        The process involves:
        1. Validating input shapes.
        2. Padding inputs if `level` > 1 to ensure divisibility for downsampling.
        3. Iterating from the coarsest resolution up to the target resolution:
           a. Downsampling `target_in`, `context_in`, `context_out` to the current iteration's resolution.
           b. Using the upsampled prediction from the previous (coarser) level as `image_context_in`.
           c. Calling `_sliding_window_autoregressive_step` for inference at this resolution.
           d. If not the final level, upsampling the current prediction to serve as `image_context_in` for the next level.
        4. Cropping the final prediction if padding was initially applied.

        Args:
            target_in (torch.Tensor): The primary input tensor. Shape: [B, C_t, D, H, W]
            context_in (torch.Tensor, optional): Semantic context input. Shape: [B, L_c, C_c, D, H, W]
            context_out (torch.Tensor, optional): Semantic context output. Shape: [B, L_co, C_co, D, H, W]
            level (int): Number of autoregressive levels. 
                         level=1 means direct inference (no autoregression).
                         level=2 means one autoregressive step (e.g., half-res prediction feeds full-res).
            forward_l_arg (int): The 'l' parameter passed to the model's forward function.
            sw_roi_size (tuple): Spatial size of the ROI for sliding window (e.g., (128, 128, 128)).
            sw_overlap (float): Overlap ratio for sliding window (0.0 to 1.0).
            sw_blend_mode (BlendMode): Blending mode for sliding window (e.g., BlendMode.GAUSSIAN).
            sw_batch_size_val (int): Batch size for sliding window operations.

        Returns:
            torch.Tensor: The final predicted output tensor. Shape: [B, C_out, D_orig, H_orig, W_orig]
        """
        if level is None:
            max_axis = torch.tensor(target_in.shape).max()
            level = int(np.ceil(np.log2(max_axis / 128)) + 1)
            
        self.verbose = verbose
        
        B_orig, C_t_orig, D_orig, H_orig, W_orig = target_in.shape
        # Input shape validation
        if context_in is not None and \
           (context_in.shape[0]!=B_orig or context_in.shape[3:]!=(D_orig,H_orig,W_orig)):
            raise ValueError("Target and context_in mismatch in batch or spatial dimensions.")
        if context_out is not None and \
           (context_out.shape[0]!=B_orig or context_out.shape[3:]!=(D_orig,H_orig,W_orig)):
            raise ValueError("Target and context_out mismatch in batch or spatial dimensions.")
        if not all(s > 0 for s in sw_roi_size):
             raise ValueError("sw_roi_size must have positive components.")

        
        original_spatial_shape_tuple = (D_orig, H_orig, W_orig)
        
        # Calculate padded shape for the highest resolution if multi-level processing is needed 
        current_padded_target_in, current_padded_context_in, current_padded_context_out, \
            padded_hires_spatial_shape_tuple, padded_hires_spatial_shape_list,padding_applied = self._pad_the_image(original_spatial_shape_tuple, 
                                                                                                                                        level, 
                                                                                                                                        target_in, 
                                                                                                                                        context_in, 
                                                                                                                                        context_out)
        
        # Prepare for autoregressive loop
        image_context_feedback = None # Stores the output of level l-1, to be used as image_context_in for level l
        image_context_feedback_in = None
        final_prediction = None

        # Autoregressive loop: from coarsest (l_loop=1) to finest (l_loop=level)
        for l_loop in range(1, level + 1):
            # Determine the spatial dimensions for inputs at the current iteration l_loop
            if l_loop == level: # Final level, use the (potentially padded) highest resolution
                current_iter_input_spatial_shape = list(padded_hires_spatial_shape_tuple)
            else:
                # For intermediate levels, downsample from the highest padded resolution
                scale_val = 2**(level - l_loop) # e.g. level=3, l_loop=1 -> scale_val=4 (1/4 res)
                                               #      level=3, l_loop=2 -> scale_val=2 (1/2 res)
                current_iter_input_spatial_shape = [s // scale_val for s in padded_hires_spatial_shape_list]
                
            # Prepare image_context_in for the model from the previous level's upsampled output
            model_image_context_in = None
            model_image_context_in_in = None
            if image_context_feedback is not None:
                model_image_context_in = self._check_size_image_context(image_context_feedback, current_iter_input_spatial_shape)
            if image_context_feedback_in is not None:   
                model_image_context_in_in = self._check_size_image_context(image_context_feedback_in, current_iter_input_spatial_shape)
             
             # Downsample (interpolate) the (padded) original inputs to the current iteration's spatial size
            original_spatial_for_this_iter_step, input_target_to_slider, \
                input_context_in_to_slider, input_context_out_to_slider, \
                input_img_ctx_to_slider, perform_temporary_upsample = self._downsample_for_autoregressive(
                                                                                                  current_iter_input_spatial_shape, 
                                                                                                  sw_roi_size, 
                                                                                                  current_padded_target_in, 
                                                                                                  current_padded_context_in, 
                                                                                                  current_padded_context_out, 
                                                                                                  model_image_context_in,
                                                                                                  model_image_context_in_in,
                                                                                                )

            # Perform inference for the current level
            current_level_pred_output = self._sliding_window_autoregressive_step(
                current_target_in=input_target_to_slider, 
                current_context_in=input_context_in_to_slider,
                current_context_out=input_context_out_to_slider, 
                image_context_in_prev=input_img_ctx_to_slider,
                roi_size=sw_roi_size, # This is the target ROI size for sliding window, e.g., (128,128,128)
                sw_batch_size=sw_batch_size_val, 
                overlap=sw_overlap,
                mode=sw_blend_mode, 
                forward_l_param=forward_l_arg
            )

            # If inputs were temporarily upsampled, downsample the prediction back to the original iteration shape
            image_context_feedback, image_context_feedback_in, final_prediction = self._resample_image_for_next_loop(
                                                                                      current_level_pred_output, 
                                                                                      input_target_to_slider, 
                                                                                      perform_temporary_upsample,
                                                                                      original_spatial_for_this_iter_step,
                                                                                      level,
                                                                                      l_loop,
                                                                                      padded_hires_spatial_shape_tuple,
                                                                                      padded_hires_spatial_shape_list,
                                                                                      intensity_clip
                                                                                     )
        
        # If padding was applied, crop the final prediction back to the original input spatial dimensions
        if padding_applied and final_prediction is not None:
            final_prediction = final_prediction[:, :, :D_orig, :H_orig, :W_orig]
        
        return final_prediction
    
    def _resample_image_for_next_loop(self, 
                                          current_level_pred_output, 
                                          input_target_to_slider, 
                                          perform_temporary_upsample,
                                          original_spatial_for_this_iter_step,
                                          level,
                                          l_loop,
                                          padded_hires_spatial_shape_tuple,
                                          padded_hires_spatial_shape_list,
                                          intensity_clip
                                         ):
            # If inputs were temporarily upsampled, downsample the prediction back to the original iteration shape
            if perform_temporary_upsample:
                current_level_pred_output = F.interpolate(current_level_pred_output, size=original_spatial_for_this_iter_step, mode='trilinear', align_corners=True)
                input_target_to_slider = F.interpolate(input_target_to_slider, size=original_spatial_for_this_iter_step, mode='trilinear', align_corners=True)

            # If this is not the final level, upsample the prediction to be used as feedback for the next (finer) level
            if l_loop < level:
                next_l_loop = l_loop + 1
                # Determine the spatial dimensions for the *inputs* of the next iteration
                if next_l_loop == level: # Next is final level
                     next_iter_input_spatial_shape = list(padded_hires_spatial_shape_tuple)
                else:
                    next_scale_val = 2**(level - next_l_loop)
                    next_iter_input_spatial_shape = [s // next_scale_val for s in padded_hires_spatial_shape_list]

                # Upsample current prediction to match the input size of the next level

                image_context_feedback = F.interpolate(
                    current_level_pred_output, 
                    size=next_iter_input_spatial_shape,
                    mode='trilinear', 
                    align_corners=True
                )
                image_context_feedback_in = F.interpolate(
                    input_target_to_slider, 
                    size=next_iter_input_spatial_shape,
                    mode='trilinear', 
                    align_corners=True
                )

                if intensity_clip: # Make the output between 0-1
                    image_context_feedback = torch.clamp(image_context_feedback, min=0.0, max=1.0)

                final_prediction = None
            else: # This was the final level (l_loop == level)
                final_prediction = current_level_pred_output
                image_context_feedback, image_context_feedback_in = None, None
            
            return image_context_feedback, image_context_feedback_in, final_prediction
    
    
    def _downsample_for_autoregressive(self, 
                                          current_iter_input_spatial_shape, 
                                          sw_roi_size, 
                                          current_padded_target_in, 
                                          current_padded_context_in, 
                                          current_padded_context_out, 
                                          model_image_context_in,
                                          model_image_context_in_in):
        '''
        Here we consider that if the size of the current level is smaller than sw_roi_size, sw_roi_size is used instead of the current size. This is to enable the model to capture clearer data information.
        '''
        # Downsample (interpolate) the (padded) original inputs to the current iteration's spatial size
        perform_temporary_upsample = any(s_iter < s_roi for s_iter, s_roi in zip(list(current_iter_input_spatial_shape), sw_roi_size))
        if not perform_temporary_upsample:
            iter_target_in = F.interpolate(current_padded_target_in, size=current_iter_input_spatial_shape, mode='trilinear', align_corners=True)

            iter_context_in = None
            if current_padded_context_in is not None:
                B_c, L_c, C_c, D_pc, H_pc, W_pc = current_padded_context_in.shape
                # Reshape for interpolation: [B,L,C,D,H,W] -> [B*L,C,D,H,W]
                reshaped_for_interp_ctx_in = current_padded_context_in.reshape(B_c*L_c, C_c, D_pc, H_pc, W_pc)
                interp_reshaped_ctx_in = F.interpolate(reshaped_for_interp_ctx_in, size=current_iter_input_spatial_shape, mode='trilinear', align_corners=True)
                # Reshape back: [B*L,C,D_iter,H_iter,W_iter] -> [B,L,C,D_iter,H_iter,W_iter]
                iter_context_in = interp_reshaped_ctx_in.reshape(B_c, L_c, C_c, *current_iter_input_spatial_shape)

            iter_context_out = None
            if current_padded_context_out is not None:
                B_co, L_co, C_co, D_pco, H_pco, W_pco = current_padded_context_out.shape
                reshaped_for_interp_ctx_out = current_padded_context_out.reshape(B_co*L_co, C_co, D_pco, H_pco, W_pco)
                interp_reshaped_ctx_out = F.interpolate(reshaped_for_interp_ctx_out, size=current_iter_input_spatial_shape, mode='trilinear', align_corners=True)
                iter_context_out = interp_reshaped_ctx_out.reshape(B_co, L_co, C_co, *current_iter_input_spatial_shape)

        else:
            target_upsample_shape = list(sw_roi_size)
            iter_target_in = F.interpolate(current_padded_target_in, size=target_upsample_shape, mode='trilinear', align_corners=True)

            if current_padded_context_in is not None:
                B_c, L_c, C_c, D_iter, H_iter, W_iter = current_padded_context_in.shape
                reshaped_for_interp_ctx_in = current_padded_context_in.reshape(B_c * L_c, C_c, D_iter, H_iter, W_iter)
                reshaped_for_interp_ctx_in = F.interpolate(reshaped_for_interp_ctx_in, size=target_upsample_shape, mode='trilinear', align_corners=True)
                iter_context_in = reshaped_for_interp_ctx_in.reshape(B_c, L_c, C_c, *target_upsample_shape)

            if current_padded_context_out is not None:
                B_co, L_co, C_co, D_iter, H_iter, W_iter = current_padded_context_out.shape
                reshaped_for_interp_ctx_out = current_padded_context_out.reshape(B_co * L_co, C_co, D_iter, H_iter, W_iter)
                reshaped_for_interp_ctx_out = F.interpolate(reshaped_for_interp_ctx_out, size=target_upsample_shape, mode='trilinear', align_corners=True)
                iter_context_out = reshaped_for_interp_ctx_out.reshape(B_co, L_co, C_co, *target_upsample_shape)

            if model_image_context_in is not None:
                model_image_context_in = F.interpolate(model_image_context_in, size=target_upsample_shape, mode='trilinear', align_corners=True)
                model_image_context_in_in = F.interpolate(model_image_context_in_in, size=target_upsample_shape, mode='trilinear', align_corners=True)


        # --- Potentially upsample inputs to sw_roi_size if they are smaller ---
        original_spatial_for_this_iter_step = list(current_iter_input_spatial_shape)
        input_target_to_slider = iter_target_in
        input_context_in_to_slider = iter_context_in
        input_context_out_to_slider = iter_context_out
        input_img_ctx_to_slider = model_image_context_in
        if model_image_context_in is not None:
            input_img_ctx_to_slider = torch.cat([model_image_context_in, model_image_context_in_in.to(model_image_context_in.device)], dim=1)
        else:
            input_img_ctx_to_slider = None

        # --- End of potential upsampling ---
        return original_spatial_for_this_iter_step, input_target_to_slider, input_context_in_to_slider, input_context_out_to_slider, input_img_ctx_to_slider, perform_temporary_upsample


    def _check_size_image_context(self, image_context_feedback, current_iter_input_spatial_shape):
        # Ensure image_context_feedback has shape [B, 1, C_ic, D_iter, H_iter, W_iter]
        if image_context_feedback.dim() == 4: # Expected [B, C, D, H, W] from interpolate/sliding_window
            model_image_context_in = image_context_feedback.unsqueeze(1) # Add L=1 dimension
        elif image_context_feedback.dim() == 5 and image_context_feedback.shape[1] == 1:
            model_image_context_in = image_context_feedback # Already in [B,1,C,D,H,W]
        else:
            raise ValueError(f"Image context feedback has unexpected shape: {image_context_feedback.shape}")
        # Sanity check for spatial dimensions of the feedback tensor
        if model_image_context_in.shape[2:] != tuple(current_iter_input_spatial_shape):
            raise ValueError(f"Size mismatch for image context feedback: {model_image_context_in.shape[2:]} vs {tuple(current_iter_input_spatial_shape)}")
        return model_image_context_in

    def _pad_the_image(self, original_spatial_shape_tuple, level, target_in, context_in, context_out):
        D_orig, H_orig, W_orig = original_spatial_shape_tuple
        padding_applied = False
        # Calculate padded shape for the highest resolution if multi-level processing is needed
        # This ensures that dimensions are cleanly divisible in all downsampling steps.
        padded_hires_spatial_shape_list = list(original_spatial_shape_tuple)
        if level > 1:
            divisor = 2**(level - 1) # Divisor needed for the coarsest level relative to original
            for i in range(3): # For D, H, W dimensions
                if padded_hires_spatial_shape_list[i] % divisor != 0:
                    padded_hires_spatial_shape_list[i] = (padded_hires_spatial_shape_list[i] // divisor + 1) * divisor

        # Initialize potentially padded versions of inputs
        current_padded_target_in = target_in
        current_padded_context_in = context_in
        current_padded_context_out = context_out

        # Apply padding if the calculated padded shape differs from the original
        if tuple(padded_hires_spatial_shape_list) != original_spatial_shape_tuple:
            padding_applied = True
            pad_d = padded_hires_spatial_shape_list[0] - D_orig
            pad_h = padded_hires_spatial_shape_list[1] - H_orig
            pad_w = padded_hires_spatial_shape_list[2] - W_orig
            # PyTorch F.pad expects padding amounts in (..., W_end, W_start, H_end, H_start, D_end, D_start)
            padding_amounts_torch = (0, pad_w, 0, pad_h, 0, pad_d) # Pad at the end of dimensions

            current_padded_target_in = F.pad(target_in, padding_amounts_torch, mode='constant', value=0)
            if context_in is not None:
                B_c, L_c, C_c, _, _, _ = context_in.shape
                # Reshape context for padding: [B,L,C,D,H,W] -> [B*L,C,D,H,W]
                reshaped_ctx_in = context_in.reshape(B_c*L_c, C_c, D_orig, H_orig, W_orig)
                padded_reshaped_ctx_in = F.pad(reshaped_ctx_in, padding_amounts_torch, mode='constant', value=0)
                # Reshape back: [B*L,C,D_pad,H_pad,W_pad] -> [B,L,C,D_pad,H_pad,W_pad]
                current_padded_context_in = padded_reshaped_ctx_in.reshape(B_c, L_c, C_c, *padded_hires_spatial_shape_list)
            if context_out is not None:
                B_co, L_co, C_co, _, _, _ = context_out.shape
                reshaped_ctx_out = context_out.reshape(B_co*L_co, C_co, D_orig, H_orig, W_orig)
                padded_reshaped_ctx_out = F.pad(reshaped_ctx_out, padding_amounts_torch, mode='constant', value=0)
                current_padded_context_out = padded_reshaped_ctx_out.reshape(B_co, L_co, C_co, *padded_hires_spatial_shape_list)

        padded_hires_spatial_shape_tuple = tuple(padded_hires_spatial_shape_list)

        return current_padded_target_in, current_padded_context_in, current_padded_context_out, padded_hires_spatial_shape_tuple, padded_hires_spatial_shape_list, padding_applied
    def _crop_img(self, image, target_size_int, padded_image_size, patch_size):
        """
        Performs a random crop centered on the image foreground and calculates the intersection matrix.
        The foreground is defined by pixels with intensity > (max - min) / 4 + min.
        If the crop exceeds image boundaries, it's shifted to stay within bounds.
        Assumes image is 3D (H, W, D).
        target_size_int is a single integer for cubic crop size.
        """
        # Ensure image is a NumPy array for calculations
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        H, W, D = padded_image_size
        target_h = target_w = target_d = target_size_int

        if not (H >= target_h and W >= target_w and D >= target_d):
            raise ValueError(f"Padded image size {padded_image_size} must be >= target crop size ({target_h}, {target_w}, {target_d})")

        # 1. Calculate foreground threshold
        min_val = np.min(image)
        max_val = np.max(image)
        threshold = (max_val - min_val) / 4.0 + min_val

        # 2. Find the coordinates of the foreground pixels
        foreground_indices = np.argwhere(image > threshold) # Returns an (N, 3) shaped array where each row is the [h, w, d] coordinates of a foreground pixel

        if foreground_indices.shape[0] > 0:
           # 3. If there is a foreground pixel, randomly select one as the cropping center point
            center_idx = random.randint(0, foreground_indices.shape[0] - 1)
            center_h, center_w, center_d = foreground_indices[center_idx]
            center_h += random.randint(-int(target_size_int/2), int(target_size_int/2))
            center_w += random.randint(-int(target_size_int/2), int(target_size_int/2))
            center_d += random.randint(-int(target_size_int/2), int(target_size_int/2))

            # 4. Calculate the initial clipping start coordinates based on the center point
# Use integer division to ensure integer coordinates
            h_start = center_h - target_h // 2
            w_start = center_w - target_w // 2
            d_start = center_d - target_d // 2

            # 5. Adjust the starting coordinates to ensure the cropping area does not exceed the image boundaries.
# max(0, ...) Ensures it is not less than 0.
# min(..., H - target_h) Ensures the starting point plus the target size is not greater than H.
            h_start = max(0, min(h_start, H - target_h))
            w_start = max(0, min(w_start, W - target_w))
            d_start = max(0, min(d_start, D - target_d))

        else:
            # If there are no foreground pixels (e.g., the image is completely black or has uniform intensity), fall back to completely random cropping
            h_start = random.randint(0, H - target_h)
            w_start = random.randint(0, W - target_w)
            d_start = random.randint(0, D - target_d)

        # 6. Calculate the end coordinates
        h_end = h_start + target_h
        w_end = w_start + target_w
        d_end = d_start + target_d

        # 7. Perform the slicing
# Make sure to use integer indices for slicing
        cropped_image = image[int(h_start):int(h_end), int(w_start):int(w_end), int(d_start):int(d_end)]

        # 8. Calculate the position matrix
        crop_coords = ((h_start, w_start, d_start), (h_end, w_end, d_end))
        grid_division = (patch_size, patch_size, patch_size)

        # Assuming calculate_intersection_matrix is defined outside the class or accessible to self
# If it's a class method, use self.calculate_intersection_matrix(...)
# If it's an external function, call calculate_intersection_matrix(...) directly
        position_matrix = calculate_intersection_matrix( # Or self.calculate_intersection_matrix
            image_size=padded_image_size,
            window_coords=crop_coords,
            grid_division=grid_division
        )
        return cropped_image, crop_coords, position_matrix


