# filename: arch/xception.py
# Complete implementation incorporating DFE  ), DD, and DDA concepts from SDIF paper.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import os
import numpy as np

# Allows duplicate library loading if needed
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

__all__ = ['xception', 'farthest_point_sample_tensor'] # Export factory and FPS util

model_urls = {
    'xception': 'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}

# --- Helper Modules (AdaIN, GRL) ---

class GRL(torch.autograd.Function):
    """ Gradient Reversal Layer """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradient and scale by alpha
        output = grad_output.neg() * ctx.alpha
        return output, None # No gradient needed for alpha

class adaIN(nn.Module):
    """ Adaptive Instance Normalization """
    def __init__(self, eps=1e-5):
        super(adaIN, self).__init__()
        self.eps = eps

    def forward(self, content_feat, gamma, beta):
        """
        Applies AdaIN using style-derived scale (gamma) and shift (beta).
        Args:
            content_feat (Tensor): Content features [B, C, H, W].
            gamma (Tensor): Scale parameter [B, C].
            beta (Tensor): Shift parameter [B, C].
        Returns:
            Tensor: Stylized features [B, C, H, W].
        """
        # Calculate channel-wise mean and variance for content features
        in_mean, in_var = torch.mean(content_feat, dim=[2, 3], keepdim=True), \
                          torch.var(content_feat, dim=[2, 3], keepdim=True)
        # Normalize content features
        out_in = (content_feat - in_mean) / torch.sqrt(in_var + self.eps)
        # Apply style-derived affine transformation
        # Unsqueeze gamma/beta to match feature dimensions [B, C, 1, 1]
        out = out_in * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out

# --- Farthest Point Sampling (Utility Function) ---
# This function is intended for use in the training loop to select diverse styles.
def farthest_point_sample_tensor(point_features, npoint):
    """
    Uses FPS algorithm to sample indices of diverse points from feature vectors.
    Input:
        point_features (Tensor): Features of points to sample from (N, D).
                                  In DDA context, these are concatenated mean/std stats.
        npoint (int): Number of points (styles) to sample.
    Return:
        centroids_features (Tensor): Features of the sampled points [npoint, D].
        centroids_indices (Tensor): Indices of the sampled points in the original tensor [npoint].
    """
    device = point_features.device
    N, D = point_features.shape
    if N < npoint:
        print(f"Warning: Trying to sample {npoint} points but only {N} available. Returning all points.")
        indices = torch.arange(N, device=device)
        return point_features, indices
    if N == 0:
        print("Warning: FPS called with 0 points.")
        return torch.empty((0, D), device=device), torch.empty((0,), dtype=torch.long, device=device)

    centroids_indices = torch.zeros((npoint,), dtype=torch.long, device=device)
    # Initialize distances to a large value
    distance = torch.ones((N,), device=device) * 1e18

    # Randomly select the first point index
    farthest_idx = torch.randint(0, N, (1,), device=device)[0]

    for i in range(npoint):
        centroids_indices[i] = farthest_idx
        # Get the feature vector of the current centroid
        centroid = point_features[farthest_idx, :]
        # Calculate squared Euclidean distance from all points to the current centroid
        dist = torch.sum((point_features - centroid) ** 2, dim=-1)
        # Update the minimum distance found so far for each point
        mask = dist < distance
        distance[mask] = dist[mask]
        # Select the point that is farthest (has the largest minimum distance)
        farthest_idx = torch.argmax(distance, dim=-1)

    sampled_features = point_features[centroids_indices]
    return sampled_features, centroids_indices


# --- Dynamic Feature Extractor (DFE) Module based on  
class DFEModuleFig3(nn.Module):
    """
    DFE Module implementation based on the structure shown in SDIF paper 
    Splits input channels, uses channel attention for the dynamic branch,
    a static conv for the other branch, and fuses the results.
    """
    def __init__(self, inplanes, planes, r=16): # r is the channel reduction ratio for attention
        super(DFEModuleFig3, self).__init__()
        # Adjust channel split if inplanes is odd
        if inplanes % 2 != 0:
             print(f"Warning: DFEModule  received odd inplanes ({inplanes}). Adjusting split.")
             self.channels_Ma = (inplanes + 1) // 2 # Dynamic branch gets potentially one more channel
             self.channels_Mb = inplanes // 2      # Static branch
        else:
             self.channels_Ma = inplanes // 2
             self.channels_Mb = inplanes // 2

        self.inplanes = inplanes
        self.planes = planes
        self.r = r

        # --- Dynamic Branch (Input Ma, Output Z = Attn * Ma) ---
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # MLP for channel attention calculation
        hidden_dim = max(1, self.channels_Ma // r) # Ensure hidden dim is at least 1
        self.fc1 = nn.Conv2d(self.channels_Ma, hidden_dim, 1, bias=False) # Use 1x1 Conv for MLP
        self.relu_att = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_dim, self.channels_Ma, 1, bias=False)
        self.sigmoid = nn.Sigmoid() # To get attention weights between 0 and 1

        # --- Static Branch (Input Mb, Output Z') ---
        self.conv_static = nn.Conv2d(self.channels_Mb, self.channels_Mb, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_static = nn.BatchNorm2d(self.channels_Mb)
        self.relu_static = nn.ReLU(inplace=True)

        # --- Fusion (Concatenate Z and Z', then apply final Conv) ---
        # Input channels = Ma channels + Mb channels = inplanes
        self.conv_fuse = nn.Conv2d(self.channels_Ma + self.channels_Mb, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_fuse = nn.BatchNorm2d(planes)
        self.relu_fuse = nn.ReLU(inplace=True) # Final activation for the module

        self._initialize_weights()

    def _initialize_weights(self):
         # Initialize weights for layers defined in this module
         for m in [self.fc1, self.fc2, self.conv_static, self.conv_fuse]:
              if isinstance(m, nn.Conv2d):
                  n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                  if n > 0: m.weight.data.normal_(0, math.sqrt(2. / n))
                  if m.bias is not None: m.bias.data.zero_()
         for m in [self.bn_static, self.bn_fuse]:
              if isinstance(m, nn.BatchNorm2d):
                  m.weight.data.fill_(1); m.bias.data.zero_()

    def forward(self, x):
        # Split input features along the channel dimension
        Ma = x[:, :self.channels_Ma, :, :] # Features for dynamic branch
        Mb = x[:, self.channels_Ma:, :, :] # Features for static branch

        # --- Dynamic Branch ---
        # Calculate channel attention weights ν(Ma)
        attn_weights = self.avg_pool(Ma)
        attn_weights = self.fc1(attn_weights)
        attn_weights = self.relu_att(attn_weights)
        attn_weights = self.fc2(attn_weights)
        attn_weights = self.sigmoid(attn_weights) # Shape: [B, C_Ma, 1, 1]
        # Apply attention: Z = ν(Ma) ⊗ Ma (element-wise multiplication)
        Z = Ma * attn_weights

        # --- Static Branch ---
        # Calculate Z' = Conv(Mb)
        Z_prime = self.conv_static(Mb)
        Z_prime = self.bn_static(Z_prime)
        Z_prime = self.relu_static(Z_prime)

        # --- Fusion ---
        # Concatenate results from both branches
        out_concat = torch.cat([Z, Z_prime], dim=1) # Shape: [B, C_Ma + C_Mb, H, W]
        # Apply final fusion convolution: F = δ(concat(Z, Z'))
        F = self.conv_fuse(out_concat)
        F = self.bn_fuse(F)
        F = self.relu_fuse(F) # Apply final activation

        return F

# --- Domain Discriminator Module (DD) ---
class Discriminator(nn.Module):
    """ Domain discriminator network used for adversarial training """
    def __init__(self, in_channels=256, num_domains=3):
        super(Discriminator, self).__init__()
        # CNN layers to reduce spatial dimensions and increase channel depth
        self.ad_net = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # Pool to 1x1 spatial size
        )
        # Final fully connected layer for domain classification
        self.fc = nn.Linear(512, num_domains)

    def forward(self, feature, alpha):
        # Apply Gradient Reversal Layer before passing to discriminator
        reversed_feature = GRL.apply(feature, alpha)
        # Pass through the discriminator network
        adversarial_out = self.ad_net(reversed_feature)
        # Flatten the output for the fully connected layer
        adversarial_out = adversarial_out.view(adversarial_out.shape[0], -1)
        # Get domain logits
        adversarial_out = self.fc(adversarial_out)
        return adversarial_out

# --- Separable Convolution & Xception Block ---
class SeparableConv2d(nn.Module):
    """ Depthwise separable convolution operation """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        # Depthwise Convolution: applies spatial filter per input channel
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        # Pointwise Convolution: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x) # Apply depthwise conv
        x = self.pointwise(x) # Apply pointwise conv
        return x

class Block(nn.Module):
    """ Basic building block for Xception, using SeparableConv2d """
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        """
        Args:
            in_filters: Input channels.
            out_filters: Output channels.
            reps: Number of repetitions of SeparableConv within the block.
            strides: Stride for the skip connection and optional MaxPool.
            start_with_relu: Whether the block starts with a ReLU activation.
            grow_first: If True, change channels in the first conv; otherwise in the last.
        """
        super(Block, self).__init__()
        # Define skip connection path if dimensions/channels change or stride > 1
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None # No skip connection needed if identity mapping

        self.relu = nn.ReLU(inplace=True)
        rep = [] # List to hold layers of the main path
        filters = in_filters

        # Build the main path layers
        if grow_first:
            if start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters # Update current filter size
            # Add remaining repetitions
            for _ in range(reps - 1):
                rep.append(self.relu)
                rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
                rep.append(nn.BatchNorm2d(filters))
        else: # Change channels in the last layer
            filters = in_filters # Use input filters for intermediate layers
            for i in range(reps - 1):
                if start_with_relu or i > 0: # Apply ReLU before conv, except maybe the very first one
                    rep.append(self.relu)
                rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
                rep.append(nn.BatchNorm2d(filters))
            # Add the last layer which changes the channel count
            if start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        # Remove initial ReLU if specified
        if not start_with_relu and len(rep) > 0 and isinstance(rep[0], nn.ReLU):
            rep = rep[1:]

        # Apply MaxPool if stride is not 1 (as done in original Xception)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep) # Main path module

    def forward(self, inp):
        x = self.rep(inp) # Output from main path

        # Prepare skip connection
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp # Use input directly if no skip conv needed

        # Ensure spatial dimensions match before adding skip connection
        if x.shape[2:] != skip.shape[2:]:
             # Downsample the skip connection if needed (e.g., due to MaxPool in main path)
             skip = F.adaptive_avg_pool2d(skip, x.shape[2:])

        x += skip # Add residual connection
        return x


# --- Xception Model with DFE (Fig 3), DD, and DDA ---
class Xception(nn.Module):
    """ Xception model modified to incorporate DFE (Fig 3 style), DD, and DDA concepts from the SDIF paper """
    def __init__(self, num_classes=1, num_domains=3, style_dim=256, adain_layer_dim=728, base_style_num=64, dfe_reduction=16):
        """
        Args:
            num_classes: Number of output classes for the main task (e.g., 1 for binary real/fake).
            num_domains: Number of domains for the domain discriminator.
            style_dim: Feature dimension where style statistics (mean, std) are extracted.
            adain_layer_dim: Feature dimension where AdaIN is applied.
            base_style_num: Number of base styles (C in SDIF paper Eq 3) stored for DDA.
            dfe_reduction: Reduction ratio 'r' for the channel attention MLP in DFE modules.
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.style_dim = style_dim
        self.adain_layer_dim = adain_layer_dim
        self.base_style_num = base_style_num

        # --- Backbone Layers ---
        # Entry Flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False) # Output size: (InputSize + 2*Padding - KernelSize)/Stride + 1 => (299+2-3)/2+1 = 150
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False) # (150+2-3)/1+1 = 150
        self.bn2 = nn.BatchNorm2d(64)

        # Middle Flow Blocks (with integrated DFE modules)
        # Block 1: 64->128, stride 2
        self.block1 = Block(64, 128, reps=2, strides=2, start_with_relu=False, grow_first=True)   # 150 -> 75
        # Block 2: 128->256, stride 2
        self.block2 = Block(128, 256, reps=2, strides=2, start_with_relu=True, grow_first=True)  # 75 -> 38, Output Channels = 256

        # DFE Module 1: Takes Block 2 output, output channels = style_dim
        self.conv_dfe1 = DFEModuleFig3(inplanes=256, planes=self.style_dim, r=dfe_reduction)      # Spatial size 38 -> 38, Output Channels = style_dim (e.g., 256)
        # Features after this point are used for Domain Discrimination and Style Statistics

        # Block 3: Takes DFE1 output, 728 channels, stride 2
        self.block3 = Block(self.style_dim, 728, reps=2, strides=2, start_with_relu=True, grow_first=True) # 38 -> 19, Output Channels = 728

        # Repeating Middle Flow Blocks
        self.block4 = Block(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True) # Spatial size 19 -> 19
        self.block5 = Block(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True) # Output Channels = 728

        # DFE Module 2: Takes Block 7 output, output channels = adain_layer_dim
        self.conv_dfe2 = DFEModuleFig3(inplanes=728, planes=self.adain_layer_dim, r=dfe_reduction) # Spatial size 19 -> 19, Output Channels = adain_layer_dim (e.g., 728)
        # AdaIN is applied to the output of this module

        # --- DDA Specific Layers ---
        self.adain_layer = adaIN() # AdaIN module instance
        # MLPs to generate AdaIN parameters (gamma, beta) from aggregated style statistics
        self.gamma_mlp = nn.Linear(self.style_dim, self.adain_layer_dim) # Maps style std deviation to gamma
        self.beta_mlp = nn.Linear(self.style_dim, self.adain_layer_dim)  # Maps style mean to beta
        # Buffers to store base styles (mean and std), updated periodically by the training loop using FPS
        self.register_buffer('mu_base', torch.randn(self.base_style_num, self.style_dim))
        self.register_buffer('sigma_base', torch.randn(self.base_style_num, self.style_dim))
        # Dirichlet distribution for sampling style combination weights (Wn in SDIF paper Eq 3)
        try:
            # Requires base_style_num > 0
            self.dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(self.base_style_num))
        except ValueError as e:
             print(f"Error initializing Dirichlet distribution (base_style_num={self.base_style_num}): {e}. Style aggregation might use uniform weights.")
             self.dirichlet = None # Fallback behavior needed in forward pass

        # --- Continue Backbone (after AdaIN) ---
        # Block 8 takes the stylized features from AdaIN
        self.block8 = Block(self.adain_layer_dim, 728, reps=3, strides=1, start_with_relu=True, grow_first=True) # Spatial size 19 -> 19
        self.block9 = Block(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True) # Output Channels = 728

        # DFE Module 3
        self.conv_dfe3 = DFEModuleFig3(inplanes=728, planes=728, r=dfe_reduction) # Spatial size 19 -> 19, Output Channels = 728

        # Exit Flow Blocks
        # Block 12: 728->1024, stride 2
        self.block12 = Block(728, 1024, reps=2, strides=2, start_with_relu=True, grow_first=False) # 19 -> 10, Output Channels = 1024

        # DFE Module 4
        self.conv_dfe4 = DFEModuleFig3(inplanes=1024, planes=1024, r=dfe_reduction) # Spatial size 10 -> 10, Output Channels = 1024

        # Final Separable Convolutions in Exit Flow
        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, padding=1, bias=False) # Spatial size 10 -> 10
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, stride=1, padding=1, bias=False) # Spatial size 10 -> 10
        self.bn4 = nn.BatchNorm2d(2048)

        # --- Classifier Head ---
        self.dropout = nn.Dropout(0.5) # Dropout before the final layer
        self.fc = nn.Linear(2048, self.num_classes) # Final classification layer

        # --- Domain Discriminator Head ---
        # Takes features from after conv_dfe1 (dimension: style_dim)
        self.dis = Discriminator(in_channels=self.style_dim, num_domains=self.num_domains)

        # Initialize model weights
        self._initialize_weights()

    def _initialize_weights(self):
        """ Initializes weights for the model layers. """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                # Check if this Conv2d belongs to a DFEModuleFig3, skip if so (handled internally)
                is_dfe_submodule = False
                for name, module in m.named_modules():
                     if isinstance(module, DFEModuleFig3):
                         # Check if m is one of the conv layers within DFEModuleFig3
                         if m is module.fc1 or m is module.fc2 or m is module.conv_static or m is module.conv_fuse:
                              is_dfe_submodule = True
                              break
                if is_dfe_submodule:
                     continue

                # Initialize SeparableConv2d sub-layers
                if hasattr(m, 'conv1') and isinstance(m.conv1, nn.Conv2d):
                    n = m.conv1.kernel_size[0] * m.conv1.kernel_size[1] * m.conv1.out_channels
                    if n > 0: m.conv1.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.conv1.bias is not None: m.conv1.bias.data.zero_()
                if hasattr(m, 'pointwise') and isinstance(m.pointwise, nn.Conv2d):
                    n = m.pointwise.kernel_size[0] * m.pointwise.kernel_size[1] * m.pointwise.out_channels
                    if n > 0: m.pointwise.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.pointwise.bias is not None: m.pointwise.bias.data.zero_()
                # Initialize other regular Conv2d layers
                elif not isinstance(m, (SeparableConv2d, DFEModuleFig3)) and not hasattr(m, 'conv1') and not hasattr(m, 'pointwise') :
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if n > 0: m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None: m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                # Check if this BN belongs to a DFEModuleFig3, skip if so (handled internally)
                 is_dfe_submodule = False
                 for name, module in m.named_modules():
                      if isinstance(module, DFEModuleFig3):
                          if m is module.bn_static or m is module.bn_fuse:
                               is_dfe_submodule = True
                               break
                 if is_dfe_submodule:
                      continue
                 # Initialize other BN layers
                 m.weight.data.fill_(1)
                 m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # Initialize Linear layers (MLPs, FC)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        # Explicitly initialize DFE modules using their own method
        for m in self.modules():
            if isinstance(m, DFEModuleFig3):
                 m._initialize_weights()


    def update_styles(self, mu_base_new, sigma_base_new):
        """
        Updates the stored base style statistics (mean and std deviation).
        This should be called by the training loop after running FPS on collected style features.
        """
        if mu_base_new.shape == self.mu_base.shape and sigma_base_new.shape == self.sigma_base.shape:
            self.mu_base.data.copy_(mu_base_new.to(self.mu_base.device))
            self.sigma_base.data.copy_(sigma_base_new.to(self.sigma_base.device))
            # Optional: Add a print statement for confirmation during debugging
            # print(f"Base styles ({self.base_style_num} styles, dim {self.style_dim}) updated in the model.")
        else:
            print(f"Error: Shape mismatch during style update. Model expected mu: {self.mu_base.shape}, sigma: {self.sigma_base.shape}. Received mu: {mu_base_new.shape}, sigma: {sigma_base_new.shape}")


    def forward(self, x, alpha=1.0):
        """
        Performs the forward pass of the model.

        Args:
            x (Tensor): Input batch of images [B, 3, H, W].
            alpha (float): Scaling factor for the Gradient Reversal Layer.

        Returns:
            tuple: A tuple containing:
                - class_pred (Tensor): Logits for the main classification task [B, num_classes].
                - domain_pred (Tensor): Logits from the domain discriminator [B, num_domains].
                - features_for_style_pool (Tensor): Features used for style statistics calculation,
                                                    returned for collection by the training loop [B, style_dim, H', W'].
        """
        B = x.size(0) # Get batch size

        # --- Entry Flow ---
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.conv2(x); x = self.bn2(x); x = self.relu(x)

        # --- Middle Flow Part 1 ---
        x = self.block1(x)
        x = self.block2(x)
        # Features after DFE1 are used for both DD and style calculation
        features_for_style_pool = self.conv_dfe1(x) # Shape: [B, style_dim, H', W']

        # --- Domain Discrimination Branch ---
        # Apply GRL and discriminator to these features
        domain_pred = self.dis(features_for_style_pool, alpha)

        # --- DDA Style Integration ---
        # Sample weights and aggregate base styles only if Dirichlet is valid and styles exist
        if self.dirichlet is not None and self.base_style_num > 0:
            # Sample style combination weights Wn ~ Dirichlet(1/C, ..., 1/C) per batch item
            # Concentration parameter alpha=1 implies uniform probability over the simplex
            # Use torch.ones for concentration parameter, matching paper reference [11] setup
            Wn = self.dirichlet.sample((B,)).to(x.device) # Shape: [B, base_style_num]

            # Aggregate base styles using sampled weights (SDIF Eq 3)
            mu_style = torch.matmul(Wn, self.mu_base)     # Shape: [B, style_dim]
            sigma_style = torch.matmul(Wn, self.sigma_base) # Shape: [B, style_dim]

            # Generate AdaIN parameters gamma and beta using MLPs
            gamma = self.gamma_mlp(sigma_style) # Shape: [B, adain_layer_dim]
            beta = self.beta_mlp(mu_style)     # Shape: [B, adain_layer_dim]
        else:
             # Fallback: If no styles or invalid Dirichlet, use identity transform for AdaIN
             gamma = torch.ones(B, self.adain_layer_dim, device=x.device)
             beta = torch.zeros(B, self.adain_layer_dim, device=x.device)

        # --- Middle Flow Part 2 ---
        # Continue main path using features from DFE1
        x = self.block3(features_for_style_pool)
        x = self.block4(x); x = self.block5(x); x = self.block6(x); x = self.block7(x)
        # Apply DFE2 before AdaIN
        features_before_adain = self.conv_dfe2(x) # Shape: [B, adain_layer_dim, H'', W'']

        # --- Apply AdaIN ---
        # Stylize the features using calculated gamma and beta (SDIF Eq 4)
        stylized_features = self.adain_layer(features_before_adain, gamma, beta)

        # --- Middle Flow Part 3 (Uses stylized features) ---
        x = self.block8(stylized_features)
        x = self.block9(x); x = self.block10(x); x = self.block11(x)
        # Apply DFE3
        x = self.conv_dfe3(x)

        # --- Exit Flow ---
        x = self.block12(x)
        # Apply DFE4
        x = self.conv_dfe4(x)
        # Final separable convolutions
        x = self.conv3(x); x = self.bn3(x); x = self.relu(x)
        x = self.conv4(x); x = self.bn4(x); x = self.relu(x)

        # --- Classifier Head ---
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # Flatten features
        x = x.view(x.size(0), -1)
        # Apply dropout
        x = self.dropout(x)
        # Final classification layer
        class_pred = self.fc(x)

        # Return the required outputs
        return class_pred, domain_pred, features_for_style_pool


# --- Factory Function ---
def xception(pretrained=True, num_classes=1, num_domains=3,
             style_dim=256, adain_layer_dim=728, base_style_num=64, dfe_reduction=16, **kwargs):
    """
    Constructs the Xception model with DFE (Fig 3 style), DD, and DDA.

    Args:
        pretrained (bool): If True, attempts to load ImageNet pretrained weights
                           into the compatible layers of the Xception backbone.
        num_classes (int): Number of output classes for the main classification task.
        num_domains (int): Number of domains for the domain discriminator.
        style_dim (int): Feature dimension where style statistics are extracted.
        adain_layer_dim (int): Feature dimension where AdaIN is applied.
        base_style_num (int): Number of base styles stored for DDA.
        dfe_reduction (int): Reduction ratio 'r' for channel attention in DFE modules.
        **kwargs: Additional arguments (currently unused).

    Returns:
        Xception: The constructed model instance.
    """
    print(f"Creating Xception SDIF model:")
    print(f"  - Pretrained: {pretrained}")
    print(f"  - Main Task Classes: {num_classes}")
    print(f"  - Domain Discriminator Classes: {num_domains}")
    print(f"  - Style Feature Dimension: {style_dim}")
    print(f"  - AdaIN Application Dimension: {adain_layer_dim}")
    print(f"  - Number of Base Styles (DDA): {base_style_num}")
    print(f"  - DFE Attention Reduction Ratio: {dfe_reduction}")

    model = Xception(num_classes=num_classes, num_domains=num_domains,
                     style_dim=style_dim, adain_layer_dim=adain_layer_dim,
                     base_style_num=base_style_num, dfe_reduction=dfe_reduction)
    if pretrained:
        try:
            print(f"Attempting to load pretrained weights from: {model_urls['xception']}")
            # Load the state dictionary from the URL
            pretrained_state_dict = model_zoo.load_url(model_urls['xception'], progress=True)
            current_model_state_dict = model.state_dict()

            # --- Filter Pretrained Weights ---
            # Exclude layers specific to our modifications (DFE, DD, DDA, final FC)
            keys_to_exclude_patterns = ['dis.', 'fc.', 'gamma_mlp.', 'beta_mlp.', '.fc1.', '.fc2.', '.conv_static.', '.conv_fuse.']
            # We also need to handle potential name changes if the original Xception implementation
            # used different naming conventions (e.g., for SeparableConv layers). Check keys carefully.

            pretrained_weights_loaded = {}
            loaded_count = 0
            ignored_count = 0
            shape_mismatch_count = 0

            for k, v in pretrained_state_dict.items():
                is_excluded = any(pattern in k for pattern in keys_to_exclude_patterns)

                if not is_excluded and k in current_model_state_dict:
                    if current_model_state_dict[k].shape == v.shape:
                        pretrained_weights_loaded[k] = v
                        loaded_count += 1
                    else:
                        # Shape mismatch, cannot load this layer
                        print(f"  - Skipping layer {k} due to shape mismatch: Model={current_model_state_dict[k].shape}, Pretrained={v.shape}")
                        shape_mismatch_count += 1
                        ignored_count += 1
                else:
                    # Key not in current model or excluded
                    ignored_count += 1

            print(f"Pretrained weights analysis:")
            print(f"  - Layers Loaded: {loaded_count}")
            print(f"  - Layers Skipped (Excluded/Not Found/Shape Mismatch): {ignored_count}")
            print(f"  - Shape Mismatches: {shape_mismatch_count}")

            if loaded_count > 0:
                # Update the current model's state dict with the filtered pretrained weights
                current_model_state_dict.update(pretrained_weights_loaded)
                # Load the updated state dict into the model (allow missing keys for excluded layers)
                model.load_state_dict(current_model_state_dict, strict=False)
                print("Successfully loaded compatible pretrained weights.")
            else:
                print("Warning: No compatible pretrained weights were found or loaded.")

        except Exception as e:
            print(f"ERROR: Could not load or process pretrained weights: {e}. The model will be trained from scratch.")

    return model

# --- Example Usage Block ---
if __name__ == '__main__':
    # Define model parameters for the example
    num_cls = 1         # Binary classification (real/fake)
    num_dom = 3         # Example: 3 source domains
    style_d = 256       # Dimension for style stats (output of conv_dfe1)
    adain_d = 728       # Dimension where AdaIN is applied (output of conv_dfe2)
    num_styles = 64     # Number of base styles for DDA
    dfe_r = 16          # Reduction ratio for DFE attention MLP

    print("-" * 50)
    print("Example Usage: Instantiating the Xception SDIF Model")
    print("-" * 50)

    # Instantiate the model
    # Set pretrained=False for quick local testing without downloading weights
    model = xception(pretrained=False,
                     num_classes=num_cls,
                     num_domains=num_dom,
                     style_dim=style_d,
                     adain_layer_dim=adain_d,
                     base_style_num=num_styles,
                     dfe_reduction=dfe_r)
    print("\nModel successfully created.")
    # print(model) # Uncomment to print the full model structure

    print("-" * 50)
    print("Testing Forward Pass...")
    print("-" * 50)
    # Create a dummy input batch
    batch_size = 4
    # Standard input size for Xception
    dummy_input = torch.rand(batch_size, 3, 299, 299)
    # Set model to evaluation mode
    model.eval()

    try:
        # Perform a forward pass without calculating gradients
        with torch.no_grad():
            # Alpha for GRL is typically 1.0 during evaluation/inference
            class_out, domain_out, style_features_out = model(dummy_input, alpha=1.0)

        print("Forward pass completed successfully.")
        print(f"Input shape:          {dummy_input.shape}")
        print(f"Class output shape:   {class_out.shape} (Expected: [{batch_size}, {num_cls}])")
        assert class_out.shape == (batch_size, num_cls)
        print(f"Domain output shape:  {domain_out.shape} (Expected: [{batch_size}, {num_dom}])")
        assert domain_out.shape == (batch_size, num_dom)
        print(f"Style features shape: {style_features_out.shape} (Expected: [{batch_size}, {style_d}, H', W'])")
        assert style_features_out.shape[0] == batch_size
        assert style_features_out.shape[1] == style_d

        print("-" * 50)
        print("Simulating Style Update Mechanism (FPS + update_styles)...")
        print("-" * 50)
        # --- This part simulates what the training loop would do ---
        # 1. Calculate style statistics from the features returned by forward pass
        style_means = torch.mean(style_features_out, dim=[2, 3]) # Shape: [B, style_d]
        style_stds = torch.std(style_features_out, dim=[2, 3])   # Shape: [B, style_d]
        # Concatenate mean and stddev for FPS input
        style_stats_collected = torch.cat([style_means, style_stds], dim=1) # Shape: [B, 2 * style_d]
        print(f"Collected style stats (mean||std) shape from batch: {style_stats_collected.shape}")

        # 2. Apply Farthest Point Sampling
        # In practice, you'd collect stats from many batches before running FPS.
        # Here, we run it on the small batch just for demonstration.
        print(f"Running FPS to select {num_styles} base styles...")
        sampled_stats, sampled_indices = farthest_point_sample_tensor(style_stats_collected, npoint=num_styles)
        print(f"Sampled style stats shape after FPS: {sampled_stats.shape}") # Shape: [min(B, num_styles), 2 * style_d]

        # 3. Update the model's style buffers
        # Check if we actually got the required number of styles (might be fewer if B < num_styles)
        if sampled_stats.shape[0] == num_styles:
             # Separate mean and std deviation from the sampled stats
             new_mu = sampled_stats[:, :style_d]     # Shape: [num_styles, style_d]
             new_sigma = sampled_stats[:, style_d:]  # Shape: [num_styles, style_d]

             # Call the model's update method
             model.update_styles(new_mu, new_sigma)
             print("Model's base style buffers updated successfully.")
        elif sampled_stats.shape[0] > 0:
             print(f"Warning: FPS returned {sampled_stats.shape[0]} styles, but model requires {num_styles}. Padding and updating.")
             padding_needed = num_styles - sampled_stats.shape[0]
             padding = torch.zeros((padding_needed, sampled_stats.shape[1]), device=sampled_stats.device)
             sampled_stats_padded = torch.cat([sampled_stats, padding], dim=0)
             new_mu = sampled_stats_padded[:, :style_d]
             new_sigma = sampled_stats_padded[:, style_d:]
             model.update_styles(new_mu, new_sigma)
             print("Model's base style buffers updated with padded styles.")
        else:
             print("Warning: FPS did not return any styles (likely due to insufficient input points). Model styles remain unchanged.")

    except Exception as e:
        # Catch and report any errors during the test run
        print("\n !!! An error occurred during the example run !!!")
        print(e)
        import traceback
        traceback.print_exc()

    print("-" * 50)
    print("Example Usage Finished.")
    print("-" * 50)