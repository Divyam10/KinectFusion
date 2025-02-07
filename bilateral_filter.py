import torch
import torch.nn.functional as F

def bilateral_filtering(
        depth_image: torch.Tensor,
        kernel_size: int,
        sigma_spatial: float,
        sigma_range: float,
        min_depth: int,
        max_depth: int
):
    radius = kernel_size // 2
    range_factor = 1.0 / (sigma_range ** 2)
    spatial_factor = 1.0 / (sigma_spatial ** 2)

    x = torch.arange(-radius, radius + 1, device=depth_image.device)
    y = torch.arange(-radius, radius + 1, device=depth_image.device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    patches = F.unfold(depth_image[None, None, ...], kernel_size, padding=radius)  # (1, K*K, H*W)
    patches = patches.view(1, kernel_size * kernel_size, *depth_image.shape)  # (1, K*K, H, W)
    center_values = depth_image[None, None, ...]

    # Precompute spatial weights (ensure shape matches unfolded patches)
    spatial_weights = torch.exp(-0.5 * (X ** 2 + Y ** 2) * spatial_factor).view(1, kernel_size * kernel_size, 1, 1)
    range_weights = torch.exp(-0.5 * ((patches - center_values) ** 2) * range_factor)
    weights = spatial_weights * range_weights  # (1, K*K, H, W)
    norm_weights = weights.sum(dim=1, keepdim=True)

    filtered_values = (weights * patches).sum(dim=1, keepdim=True) / norm_weights
    filtered_values = torch.clamp(filtered_values, min_depth, max_depth).squeeze()

    vertex_validity_mask = (filtered_values > 0)

    return filtered_values, vertex_validity_mask
