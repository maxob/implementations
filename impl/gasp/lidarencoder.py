import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional
import copy  # Added import

# --- Helper Functions ---


def _get_clones(module, N):
    """Creates N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# Helper function to generate reference points (normalized)
def _create_grid_reference_points(spatial_shapes, device):
    """
    Generates grid reference points for each feature map level.

    Args:
        spatial_shapes (list): List of tuples (H, W) for each feature level.
        device (torch.device): Device to create tensors on.

    Returns:
        torch.Tensor: Reference points tensor shape (Sum(H_l*W_l), 2), range [0, 1].
    """
    reference_points_list = []
    for H, W in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            indexing="ij",
        )
        # Normalize to [0, 1]
        ref_y = ref_y / H
        ref_x = ref_x / W
        # Stack and reshape: (H, W, 2) -> (H*W, 2)
        ref = torch.stack((ref_x, ref_y), -1).reshape(-1, 2)
        reference_points_list.append(ref)
    # Concatenate across levels: (Sum(H_l*W_l), 2)
    reference_points = torch.cat(reference_points_list, 0)
    return reference_points


# --- Deformable Attention Module ---


class MultiScaleDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention Module based on Deformable DETR paper.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dims (int): Embedding dimension (C).
            num_heads (int): Number of attention heads (M).
            num_levels (int): Number of feature levels (L).
            num_points (int): Number of sampling points per query per level per head (K).
            dropout (float): Dropout rate.
        """
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, but got {embed_dims} and {num_heads}"
            )

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points
        self.head_dim = embed_dims // num_heads

        # Linear layer to predict sampling offsets (2D for each point)
        self.sampling_offset_proj = nn.Linear(
            embed_dims, self.num_heads * self.num_levels * self.num_points * 2
        )
        # Linear layer to predict attention weights
        self.attention_weight_proj = nn.Linear(
            embed_dims, self.num_heads * self.num_levels * self.num_points
        )
        # Linear layer for value projection
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        # Linear layer for output projection
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize sampling offsets to zero initially
        nn.init.constant_(self.sampling_offset_proj.weight.data, 0.0)
        # Initialize offset biases s.t. points sample identity grid initially
        # This requires knowing the reference points structure, complicated.
        # Simpler init: Set biases also to zero. Model learns from there.
        nn.init.constant_(self.sampling_offset_proj.bias.data, 0.0)

        # Initialize attention weights projection
        nn.init.xavier_uniform_(self.attention_weight_proj.weight.data)
        nn.init.constant_(self.attention_weight_proj.bias.data, 0.0)
        # Initialize value and output projections
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,  # (Batch, NumQuery, C)
        reference_points: torch.Tensor,  # (Batch, NumQuery, NumLevels, 2) or (B, Nq, 2) normalized [0,1]
        value: torch.Tensor,  # (Batch, Sum(H_l*W_l), C)
        value_spatial_shapes: List[tuple],  # List of (H_l, W_l)
        value_level_start_index: torch.Tensor,  # (NumLevels, )
        value_padding_mask: Optional[torch.Tensor] = None,  # (Batch, Sum(H_l*W_l))
    ) -> torch.Tensor:
        """
        Forward pass for Multi-Scale Deformable Attention.
        """
        bs, num_query, _ = query.shape
        bs_val, num_value_points, _ = value.shape  # bs_val should == bs
        assert (value_spatial_shapes is not None) and (value_level_start_index is not None)
        assert sum(H * W for H, W in value_spatial_shapes) == num_value_points

        # Project value features: (B, Sum(HW), C) -> (B, Sum(HW), C)
        value = self.value_proj(value)

        # Apply padding mask to value
        if value_padding_mask is not None:
            value = value.masked_fill(value_padding_mask[..., None], float(0))

        # Reshape value for multi-head: (B, Sum(HW), M, C/M) -> (B, Sum(HW), M, Dv)
        value = value.view(bs, num_value_points, self.num_heads, self.head_dim)

        # Predict sampling offsets: (B, Nq, C) -> (B, Nq, M*L*K*2)
        sampling_offsets = self.sampling_offset_proj(query)
        # Reshape: (B, Nq, M, L, K, 2)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        # Predict attention weights: (B, Nq, C) -> (B, Nq, M*L*K)
        attention_weights = self.attention_weight_proj(query)
        # Reshape: (B, Nq, M, L*K)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        # Apply Softmax: Normalize weights across all points (L*K) for each head
        attention_weights = F.softmax(attention_weights, -1)
        # Reshape: (B, Nq, M, L, K)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        # Prepare reference points
        # reference_points shape: (B, Nq, L, 2) or (B, Nq, 2)
        if reference_points.dim() == 3:
            # If shape is (B, Nq, 2), unsqueeze and repeat for levels
            # Check if num_levels == 1 for this case? No, paper implies p_hat is used for all levels
            reference_points_expanded = reference_points[:, :, None, :].expand(
                -1, -1, self.num_levels, -1
            )  # (B, Nq, L, 2)
        elif reference_points.dim() == 4 and reference_points.shape[2] == self.num_levels:
            reference_points_expanded = reference_points  # Already (B, Nq, L, 2)
        else:
            raise ValueError(
                f"Reference points shape mismatch: expected (B, Nq, {self.num_levels}, 2) or (B, Nq, 2), got {reference_points.shape}"
            )

        # Calculate sampling locations per level based on reference points and offsets
        # Offsets are added to reference points. Clamping might be needed if offsets are large.
        # Add offset to reference points: (B, Nq, 1, L, 1, 2) + (B, Nq, M, L, K, 2) -> (B, Nq, M, L, K, 2)
        # Need reference points shape (B, Nq, 1, L, 1, 2) for broadcasting
        sampling_locations = reference_points_expanded.unsqueeze(2).unsqueeze(4) + sampling_offsets
        # sampling_locations = reference_points_expanded[:, :, None, :, None, :].float() + sampling_offsets # Use float

        # --- Core Sampling Logic using F.grid_sample ---
        # grid_sample expects input (N, C, H, W) and grid (N, H_out, W_out, 2) in range [-1, 1]

        # Reshape value: (B, Sum(HW), M, Dv) -> (B, M, Dv, Sum(HW)) -> Split by level
        value_permuted = value.permute(0, 2, 3, 1)  # (B, M, Dv, Sum(HW))
        value_list = []
        value_level_start_index_list = value_level_start_index.tolist() + [num_value_points]
        for lvl in range(self.num_levels):
            start_idx = value_level_start_index_list[lvl]
            end_idx = value_level_start_index_list[lvl + 1]
            h, w = value_spatial_shapes[lvl]
            # Get value for this level: (B, M, Dv, H*W) -> Reshape to (B*M, Dv, H, W)
            value_l = value_permuted[..., start_idx:end_idx].reshape(
                bs * self.num_heads, self.head_dim, h, w
            )
            value_list.append(value_l)

        # Reshape sampling locations & normalize to [-1, 1] for grid_sample
        # Input locations are normalized [0, 1] + unconstrained offsets. Clamp? Yes.
        sampling_locations = sampling_locations.clamp(0.0, 1.0)  # Clamp normalized coords+offsets
        grid = sampling_locations * 2.0 - 1.0  # Scale to [-1, 1]
        # grid shape: (B, Nq, M, L, K, 2)

        # Sample for each level
        sampled_value_list = []
        for lvl in range(self.num_levels):
            # Grid for level l: (B, Nq, M, K, 2)
            grid_l = grid[:, :, :, lvl, :, :]
            # Reshape grid for grid_sample: (B, Nq, M, K, 2) -> (B*M, Nq, K, 2)
            grid_l = grid_l.permute(0, 2, 1, 3, 4).reshape(
                bs * self.num_heads, num_query, self.num_points, 2
            )

            # Sample using value_list[lvl] which is (B*M, Dv, H, W)
            sampled_value_l = F.grid_sample(
                value_list[lvl], grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            # Output shape: (B*M, Dv, Nq, K)
            sampled_value_list.append(sampled_value_l)

        # Concatenate sampled values across levels: List[(B*M, Dv, Nq, K)] -> (B*M, Dv, Nq, L*K)
        sampled_value = torch.stack(sampled_value_list, dim=-1).flatten(3)  # (B*M, Dv, Nq, L*K)

        # Reshape attention weights: (B, Nq, M, L, K) -> (B*M, Nq, L*K)
        attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
            bs * self.num_heads, num_query, self.num_levels * self.num_points
        )

        # Apply attention weights: (B*M, Dv, Nq, L*K) * (B*M, 1, Nq, L*K) -> sum over L*K
        output = (sampled_value * attention_weights.unsqueeze(1)).sum(-1)  # (B*M, Dv, Nq)

        # Reshape back to (B, Nq, M*Dv) = (B, Nq, C)
        output = output.permute(0, 2, 1).reshape(bs, num_query, self.embed_dims)

        # Final output projection
        output = self.output_proj(output)

        # Apply dropout
        output = self.dropout(output)

        return output


# --- Standard ResNet BasicBlock (Same as before) ---


class BasicBlock(nn.Module):
    """Standard ResNet Basic Block."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# --- Updated LiDAREncoder with functional Attention ---


class LiDAREncoder(nn.Module):
    """
    Implements the LiDAR Encoder architecture from Figure 8,
    using the functional MultiScaleDeformableAttention.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        in_channels = config.bev_input_channels
        fpn_channels = config.fpn_out_channels

        # Positional Embedding (Placeholder - Needs actual implementation e.g., Sine)
        # Generate parameters for different levels based on typical strides 8, 16, 32
        # This is just a placeholder size/concept
        self.pos_embed_L4 = nn.Parameter(
            torch.randn(1, fpn_channels, config.bev_h // 4, config.bev_w // 4)
        )
        self.pos_embed_L8 = nn.Parameter(
            torch.randn(1, fpn_channels, config.bev_h // 8, config.bev_w // 8)
        )
        self.pos_embed_L16 = nn.Parameter(
            torch.randn(1, fpn_channels, config.bev_h // 16, config.bev_w // 16)
        )
        self.pos_embed_list = [self.pos_embed_L4, self.pos_embed_L8, self.pos_embed_L16]
        print("LiDAREncoder: Added placeholder pos_embed parameters per level")

        # Level Embedding (Learnable)
        self.level_embed = nn.Parameter(
            torch.randn(config.num_feature_levels, fpn_channels)
        )  # L, C
        print(f"LiDAREncoder: Added learnable level_embed shape: {self.level_embed.shape}")

        # Initial Convolution
        self.conv1 = nn.Conv2d(
            in_channels, config.backbone_channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(config.backbone_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.current_channels = config.backbone_channels[0]

        # Backbone Stages
        self.layer1 = self._make_layer(
            BasicBlock, config.backbone_channels[0], config.backbone_blocks[0], stride=2
        )  # L/2, W/2
        self.layer2 = self._make_layer(
            BasicBlock, config.backbone_channels[1], config.backbone_blocks[1], stride=2
        )  # L/4, W/4
        self.layer3 = self._make_layer(
            BasicBlock, config.backbone_channels[2], config.backbone_blocks[2], stride=2
        )  # L/8, W/8
        self.layer4 = self._make_layer(
            BasicBlock, config.backbone_channels[3], config.backbone_blocks[3], stride=2
        )  # L/16, W/16

        # Input projection for attention (projects backbone output to fpn_channels)
        self.input_proj = nn.ModuleList()
        # Features used are from layer2, layer3, layer4 outputs
        backbone_out_channels = [
            config.backbone_channels[i] * BasicBlock.expansion for i in range(1, 4)
        ]  # Channels for C3, C4, C5 stages effectively
        for backbone_ch in backbone_out_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(backbone_ch, fpn_channels, kernel_size=1),
                    nn.GroupNorm(32, fpn_channels),  # Optional GroupNorm
                )
            )

        # Multi-Scale Deformable Attention Module Instance
        self.deformable_attention = MultiScaleDeformableAttention(
            embed_dims=fpn_channels,
            num_heads=config.def_attn_heads,
            num_levels=config.num_feature_levels,  # Should be 3
            num_points=config.def_attn_points,
        )

        # FPN Layers
        self.fpn_lat_conv2 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=1)
        self.fpn_lat_conv3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=1)
        self.fpn_lat_conv4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=1)
        self.fpn_out_conv2 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.fpn_out_conv3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.fpn_out_conv4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)

    def _make_layer(self, block, channels, num_blocks, stride=1):
        """Helper function to create a ResNet stage."""
        downsample = None
        if stride != 1 or self.current_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.current_channels,
                    channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channels * block.expansion),
            )
        layers = []
        layers.append(block(self.current_channels, channels, stride, downsample))
        self.current_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.current_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input BEV feature map (Batch, C_in, H, W)
        Returns:
            List[torch.Tensor]: List of FPN output feature maps.
        """
        # Initial Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Backbone Stages
        s1_out = self.layer1(x)  # H/2, W/2 (Not used by Attn/FPN in diagram)
        s2_out = self.layer2(s1_out)  # H/4, W/4
        s3_out = self.layer3(s2_out)  # H/8, W/8
        s4_out = self.layer4(s3_out)  # H/16, W/16

        # Prepare features for MultiScaleDeformableAttention
        features_list = [s2_out, s3_out, s4_out]
        projected_features_flat = []
        spatial_shapes = []
        pos_embeds_flat = []
        masks_flat = []

        for i, feat in enumerate(features_list):
            bs, _, h, w = feat.shape
            spatial_shapes.append((h, w))

            # Project features: (B, C_backbone, H, W) -> (B, C_fpn, H, W)
            proj_feat = self.input_proj[i](feat)
            # Flatten: (B, C_fpn, H, W) -> (B, H*W, C_fpn)
            proj_feat_flat = proj_feat.flatten(2).permute(0, 2, 1)

            # Create mask (assuming no padding)
            mask = torch.zeros((bs, h * w), dtype=torch.bool, device=x.device)
            masks_flat.append(mask)

            # Positional Embedding (Placeholder - needs resize/interpolation)
            pos_embed = F.interpolate(
                self.pos_embed_list[i], size=(h, w), mode="bilinear", align_corners=False
            )
            pos_embed_flat = pos_embed.flatten(2).permute(0, 2, 1)  # (B, H*W, C_fpn)
            pos_embeds_flat.append(pos_embed_flat)

            # Add level embedding to feature
            proj_feat_flat += self.level_embed[i]

            projected_features_flat.append(proj_feat_flat)

        # Concatenate flattened features across levels
        value = torch.cat(projected_features_flat, dim=1)  # (B, Sum(HW), C_fpn)
        value_padding_mask = torch.cat(masks_flat, dim=1)  # (B, Sum(HW))
        pos_embed_cat = torch.cat(pos_embeds_flat, dim=1)  # (B, Sum(HW), C_fpn)

        # Prepare query and reference points for encoder self-attention
        query = value + pos_embed_cat  # Query = Feature + Pos Embed + Level Embed

        # Generate reference points (normalized grid coordinates)
        reference_points_grid = _create_grid_reference_points(
            [(h, w) for h, w in spatial_shapes], device=x.device
        )  # (Sum(HW), 2)
        # Expand to match query shape: (B, Sum(HW), 2)
        reference_points = reference_points_grid.unsqueeze(0).expand(bs, -1, -1)

        # Calculate level start indices
        level_start_index = torch.cat(
            (
                torch.tensor([0], device=x.device),
                torch.tensor([h * w for h, w in spatial_shapes], device=x.device).cumsum(0)[:-1],
            )
        )

        # Apply Multi-Scale Deformable Attention
        # Query and Value are the same in encoder self-attention (with pos/level embeds)
        attn_output = self.deformable_attention(
            query=query,
            reference_points=reference_points,  # Shape (B, Sum(HW), 2)
            value=value,  # Shape (B, Sum(HW), C_fpn)
            value_spatial_shapes=spatial_shapes,
            value_level_start_index=level_start_index,
            value_padding_mask=value_padding_mask,
        )  # Output: (B, Sum(HW), C_fpn)

        # Reshape attention output back to spatial feature maps
        start = 0
        attn_features_list = []
        for i, (h, w) in enumerate(spatial_shapes):
            end = start + h * w
            level_output = attn_output[:, start:end, :]  # (B, H*W, C_fpn)
            level_output = level_output.permute(0, 2, 1).reshape(bs, -1, h, w)  # (B, C_fpn, H, W)
            attn_features_list.append(level_output)
            start = end

        # Features enhanced by attention
        s2_enhanced, s3_enhanced, s4_enhanced = attn_features_list

        # FPN Calculation (using enhanced features)
        p4 = self.fpn_lat_conv4(s4_enhanced)
        p3_lat = self.fpn_lat_conv3(s3_enhanced)
        p2_lat = self.fpn_lat_conv2(s2_enhanced)

        # Top-down pathway
        p3 = p3_lat + F.interpolate(
            p4, size=p3_lat.shape[-2:], mode="bilinear", align_corners=False
        )
        p2 = p2_lat + F.interpolate(
            p3, size=p2_lat.shape[-2:], mode="bilinear", align_corners=False
        )

        # Output convolutions
        p4_out = self.fpn_out_conv4(p4)
        p3_out = self.fpn_out_conv3(p3)
        p2_out = self.fpn_out_conv2(p2)

        return [p2_out, p3_out, p4_out]  # P2, P3, P4 (H/4, H/8, H/16)


# --- Configuration Class (Updated) ---


class LiDAREncoderConfig:
    def __init__(self):
        # Input settings
        self.bev_input_channels = 64
        self.bev_h = 512  # Example BEV height (needs to be divisible by 16)
        self.bev_w = 512  # Example BEV width (needs to be divisible by 16)

        # Backbone settings
        self.backbone_channels = [64, 128, 256, 512]  # C2, C3, C4, C5 output channels effectively
        self.backbone_blocks = [2, 2, 2, 2]  # e.g., ResNet18 structure
        self.num_feature_levels = 3  # Using features from L/4, L/8, L/16 stages (Indices 0, 1, 2)

        # Multi-Scale Deformable Attention settings
        self.def_attn_heads = 8
        self.def_attn_points = 4  # K value

        # FPN settings
        self.fpn_out_channels = 256  # Common dimension C for Attn and FPN


# --- Example Usage ---
if __name__ == "__main__":
    config = LiDAREncoderConfig()
    # Ensure BEV dimensions are divisible by the largest stride factor (16)
    assert config.bev_h % 16 == 0 and config.bev_w % 16 == 0

    model = LiDAREncoder(config).eval()  # Set to eval mode

    batch_size = 2
    dummy_bev_input = torch.randn(batch_size, config.bev_input_channels, config.bev_h, config.bev_w)

    print(f"Input shape: {dummy_bev_input.shape}")
    with torch.no_grad():  # Disable gradients for example usage
        fpn_outputs = model(dummy_bev_input)

    print("\nFPN Output Shapes:")
    expected_shapes = [
        (config.bev_h // 4, config.bev_w // 4),
        (config.bev_h // 8, config.bev_w // 8),
        (config.bev_h // 16, config.bev_w // 16),
    ]
    for i, out in enumerate(fpn_outputs):
        print(f"  Level P{i+2}: {out.shape}")
        assert out.shape[-2:] == expected_shapes[i], f"Shape mismatch for P{i+2}"
        assert out.shape[1] == config.fpn_out_channels, f"Channel mismatch for P{i+2}"

    print("\nOutput resolutions and channels match expected FPN levels.")
    print("\nModel initialization successful.")
