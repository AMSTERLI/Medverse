from dataclasses import dataclass, field
from pydantic import validate_arguments
from .nn.vmap import Vmap, vmap
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *
import random


from .nn.layers import ResidualUnit
from .nn.layers import ConvBlock_target_encoder, ConvBlock_context_c2t,\
        ConvBlock_context_t2c, PairwiseConvAvgModelBlock_c2t, PairwiseConvAvgModelOutput
from .nn.unet_backbone import TargetDecoder, TargetEncoder, ContextUNet

class Medverse(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 stages,
                 dim,
                 inner_channels,
                 conv_layers_per_stage,
                 patch_num = 4,
                 hidden_size=66,
                 img_size=128,
                 use_ccti=False,
                 ccti_mode='learned',
                 ccti_channel_ratio=0.25,
                 ccti_bidirectional=True,
                 ccti_stage_indices=(0, 1, 2),
                ):
        super().__init__()
        self.context_unet = ContextUNet(in_channels = in_channels,
                    out_channels = out_channels, 
                    stages = stages,
                    dim = dim,
                    inner_channels = inner_channels,
                    unet_block_cls_t2c = ConvBlock_context_t2c,
                    unet_block_cls_c2t = ConvBlock_context_c2t,
                    kwargs = {'conv_layers_per_stage':conv_layers_per_stage},
                    patch_num = patch_num,
                    attention_dim = hidden_size,
                    image_size = img_size)
        
        self.target_encoder = TargetEncoder(in_channels = in_channels, 
                    out_channels = out_channels, 
                    stages = stages, 
                    dim = dim,
                    unet_block_cls_t2c = ConvBlock_target_encoder,
                    inner_channels = inner_channels,
                    kwargs = {'conv_layers_per_stage':conv_layers_per_stage})
        
        self.target_decoder = TargetDecoder(in_channels = in_channels, 
                                    out_channels = out_channels, 
                                    stages = stages, 
                                    dim = dim,
                                    inner_channels = inner_channels,
                                    unet_block_cls_c2t = PairwiseConvAvgModelBlock_c2t,
                                    output_block_cls = PairwiseConvAvgModelOutput,
                                    kwargs = {'conv_layers_per_stage':conv_layers_per_stage},
                                    patch_num = patch_num,
                                    attention_dim = hidden_size,
                                    image_size = img_size,
                                    use_ccti=use_ccti,
                                    ccti_mode=ccti_mode,
                                    ccti_channel_ratio=ccti_channel_ratio,
                                    ccti_bidirectional=ccti_bidirectional,
                                    ccti_stage_indices=tuple(ccti_stage_indices))
    
        self.dim = dim
        self.stages = stages
        self.shuffle_mini_context = True
        self.patch_num = patch_num
        self.hidden_size = hidden_size
        self.img_size = img_size
        
        # Generate sincos position embedding
        from .attention_utils.position_embedding_generator import create_position_embedding
        patch_dim = img_size // patch_num
        self.sincos_pos_embed = create_position_embedding(
            img_size=[img_size, img_size, img_size],
            patch_size=[patch_dim, patch_dim, patch_dim],  # Since we're already working with patches
            hidden_size=hidden_size,
            pos_embed_type="sincos",
            spatial_dims=3
        )
        
        # Initialize learnable code parameters
        self.target_code = nn.Parameter(torch.randn(hidden_size) * 0.1)
        self.image_context_code = nn.Parameter(torch.randn(hidden_size) * 0.1)
        self.image_context_embedding = nn.Parameter(torch.randn(inner_channels[0]) * 0.1)
        
    def fuse_feature(self, Features, features, Weight, weight = 1):
        # Fuse the context features into mean context features.
        ori_context_weight = Weight/(Weight+weight)
        new_context_weight = weight/(Weight+weight)
        for ind in range(len(Features)):
            Features[ind] = Features[ind]*ori_context_weight+features[ind]*new_context_weight
        Weight+=weight
        return Features, Weight
    
    def context_sequential_parallel_processing(self, 
                                               context_in, 
                                               context_out, 
                                               l, 
                                               target_features, 
                                               context_pos_embed,
                                               target_pos_embed,
                                               image_context_embedding = None,
                                              ):
        # Context processing.
        # initialize vars
        Weight = 0
        context_features_mean = [0, 0, 0, 0, 0]
        
        # Shuffle mini-context
        context_partition = [(i*l, min(i*l+l,context_in.shape[1])) for i in range(math.ceil(context_in.shape[1]/l))]
        if self.shuffle_mini_context: random.shuffle(context_partition)
        
        # Process the 1 to (n-1) mini-batch
        with torch.no_grad(): # To save GPU memory
            for start_index, end_index in context_partition[:-1]:
                context_features = self.context_unet(
                    context_in[:,start_index:end_index,:].to(next(self.parameters()).device),  
                    context_out[:,start_index:end_index,:].to(next(self.parameters()).device), 
                    target_features,
                    pos_embed_q=context_pos_embed[:,start_index:end_index].to(next(self.parameters()).device),
                    pos_embed_k=target_pos_embed.to(next(self.parameters()).device),
                    image_context_embedding = image_context_embedding,
                ) # list of [BxLxCxHxWxD]

                # update the context_features_mean
                tmp_context_features_mean = [i.mean(dim=1, keepdim=True) for i in context_features] # list of [Bx1xCxHxWxD]
                context_features_mean, Weight = self.fuse_feature(context_features_mean, 
                                                    tmp_context_features_mean, 
                                                    Weight, 
                                                    weight = context_features[0].shape[1])
        # Detach
        context_features_mean = [c.detach() if type(c) is not int else c for c in context_features_mean]
        
        # Process the last mini-batch
        start_index, end_index = context_partition[-1]
        context_features_last = self.context_unet(
            context_in[:,start_index:end_index,:].to(next(self.parameters()).device),
            context_out[:,start_index:end_index,:].to(next(self.parameters()).device), 
            target_features,
            pos_embed_q=context_pos_embed[:,start_index:end_index].to(next(self.parameters()).device),
            pos_embed_k=target_pos_embed.to(next(self.parameters()).device),
            image_context_embedding = image_context_embedding,
        ) # list of [BxLxCxHxWxD]

        # update the context_features_mean & Enlarge gradient from the last mini-context
        tmp_context_features_last_mean = \
            [
            ScaleGradientNTimes.apply(i.mean(dim=1, keepdim=True),len(context_partition)) for i in context_features_last
            ] # list of [Bx1xCxHxWxD]
        
        # Fuse context representation.
        context_features_mean, Weight = self.fuse_feature(context_features_mean, 
                                            tmp_context_features_last_mean, 
                                            Weight, 
                                            weight = context_features_last[0].shape[1])
        return context_features_mean
        
    
    def forward(self, 
                target_in, 
                context_in, # BxLxCinxHxWxD
                context_out,  # BxLxCinxHxWxD
                image_context_in = None, # BxLxCinxHxWxD
                image_context_out = None, # BxLxCinxHxWxD
                retrieval_similarity = None,
                l=3):
        '''
        Args:
            context_in (torch.Tensor): Context input, shape BxLxCinxHxWxD. Could be store in the cpu memory.
            context_out (torch.Tensor): Context output, shape BxLxCoutxHxWxD. Could be store in the cpu memory.
            target_in (torch.Tensor): Target input, shape BxCinxHxWxD
            l (int): Size of mini-context.
            
        Returns:
            torch.Tensor: Target output, shape BxCoutxHxWxD
            
        '''
        
        # Fix the position embedding as sin-cos position embedding
        diag_matrix = torch.eye(self.patch_num**self.dim)
        expanded_matrix = diag_matrix.unsqueeze(0) # Shape: ( 1, 64, 64)
        target_pos_matrix = expanded_matrix.expand(target_in.shape[0], -1, -1) # Shape: (B, 64, 64)
        if context_out is not None:  context_pos_matrix = expanded_matrix.expand(context_in.shape[0], context_in.shape[1], -1, -1) # Shape: (B, L, 64, 64)
        else: context_pos_matrix = None
        if image_context_in is not None: image_context_pos_matrix = expanded_matrix.expand(image_context_in.shape[0], image_context_in.shape[1], -1, -1) # Shape: (B, L, 64, 64)
        else: image_context_pos_matrix = None
        
        
        # Target processing.
        shortcuts, target = self.target_encoder(target_in)
        target_features = [i[1] for i in shortcuts] # list of [BxCxHxWxD]
        
        # Calculate position embeddings with learnable codes
        if target_pos_matrix is None: 
            raise ValueError(f"Position matrix is not input.")
        else:
            target_pos_embed = torch.matmul(target_pos_matrix, self.sincos_pos_embed.to(target_pos_matrix.device)) #[B, patch_num**dim, hidden_size]
            # Add expanded target_code
            target_pos_embed = target_pos_embed + self.target_code.view(1, 1, -1).expand(target_pos_embed.shape).to(target_pos_embed.device)
        
        # Context processing.
        context_pos_embed_mean, context_features_mean = None, None
        if context_pos_matrix is not None:
            context_pos_embed = torch.matmul(context_pos_matrix, self.sincos_pos_embed.to(context_pos_matrix.device)) #[B, L, patch_num**dim, hidden_size]
            context_features_mean = self.context_sequential_parallel_processing(
                                                   context_in, 
                                                   context_out, 
                                                   l, 
                                                   target_features, 
                                                   context_pos_embed,
                                                   target_pos_embed,
                                                  )# list of [Bx1xCxHxWxD]
            context_pos_embed_mean = torch.mean(context_pos_embed, dim=1)
            
        # Image context processing.
        if image_context_pos_matrix is not None: # if input the cropped context
            image_context_pos_embed = torch.matmul(image_context_pos_matrix, self.sincos_pos_embed.to(image_context_pos_matrix.device)) #[B, L, patch_num**dim, hidden_size]
            # Add expanded image_context_code
            image_context_pos_embed = image_context_pos_embed + self.image_context_code.view(1, 1, 1, -1).expand(image_context_pos_embed.shape).to(image_context_pos_embed.device)
            
            image_context_features_mean = self.context_sequential_parallel_processing(
                                               image_context_in, 
                                               image_context_out, 
                                               l, 
                                               target_features, 
                                               image_context_pos_embed,
                                               target_pos_embed,
                                               image_context_embedding = self.image_context_embedding,
                                              )# list of [Bx1xCxHxWxD]
            image_context_pos_embed_mean = torch.mean(image_context_pos_embed, dim=1)
            
            # add feature to context feature list
            context_features_mean = [torch.cat((context_features_mean[i], image_context_features_mean[i]), dim=1) for i in range(len(context_features_mean))] \
                                                            if context_features_mean is not None else image_context_features_mean
            context_pos_embed_mean = torch.cat((context_pos_embed_mean, image_context_pos_embed_mean), dim=0) if context_pos_embed_mean is not None else image_context_pos_embed_mean
        
        # Target decoder processing with context features.
        output = self.target_decoder(
            target, 
            context_features_mean, 
            shortcuts,
            pos_embed_q=target_pos_embed.to(next(self.parameters()).device),
            pos_embed_k=context_pos_embed_mean.to(next(self.parameters()).device),
            retrieval_similarity=retrieval_similarity,
        ) # BxCoutxHxWxD
        
        return output


from torch.autograd import Function

class ScaleGradientNTimes(Function):
    @staticmethod
    def forward(ctx, input_tensor, scale_factor):
        """
        Forward propagation: Identity operation, input is directly output, and save scale_factor to context.
        """
        ctx.scale_factor = scale_factor
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward propagation: Multiply the received gradient by the scale_factor saved in the context.
        """
        return grad_output * ctx.scale_factor, None # Returns two gradients: with respect to input_tensor and scale_factor
    
    
if __name__ == '__main__':
    device = 'cuda'
    unet2d = PairwiseConvAvgModel(dim=2, stages=4, in_channels=2,
                                  out_channels=3, inner_channels=32, conv_layers_per_stage=2).to(device)
    context_in = torch.rand(7, 8, 2, 64, 64).to(device)
    context_out = torch.rand(7, 8, 3, 64, 64).to(device)
    target_in = torch.rand(7, 2, 64, 64).to(device)
    target_out = unet2d(context_in, context_out, target_in)
    print(unet2d)
    print('2d ok')

    unet3d = PairwiseConvAvgModel(dim=3,
                                  stages=4,
                                  in_channels=2,
                                  out_channels=3,
                                  inner_channels=32,
                                  conv_layers_per_stage=2).to(device)
    context_in = torch.rand(7, 8, 2, 64, 64, 32).to(device)
    context_out = torch.rand(7, 8, 3, 64, 64, 32).to(device)
    target_in = torch.rand(7, 2, 64, 64, 32).to(device)
    target_out = unet3d(context_in, context_out, target_in)
    print(unet3d)
    print('3d ok')
