import torch
import torch.nn.functional as F


class ICP(torch.nn.Module):
    def __init__(self, max_iterations):
        super().__init__()
        self.max_iterations = max_iterations

    def forward(self, depth_source, depth_target, pose, K, optimizer):
        vertices_source = self.compute_vertices(depth_source, K)
        normals_source = self.compute_normals(vertices_source)

        vertices_target = self.compute_vertices(depth_target, K)

        for i in range(self.max_iterations):
            # apply projective data association

            pose = optimizer()

        return pose

    @staticmethod
    def compute_vertices(depth_map, K):
        H, W = depth_map.shape
        device = depth_map.device
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        pixel_grid_v, pixel_grid_u = torch.meshgrid([torch.arange(0, W), torch.arange(0, H)], indexing="xy")
        pixel_grid_u.to(device)
        pixel_grid_v.to(device)

        print(pixel_grid_u.shape, pixel_grid_v.shape)

        pixel_grid_u = ((pixel_grid_u - cx) / fx) * depth_map
        pixel_grid_v = ((pixel_grid_v - cy) / fy) * depth_map

        vertices = torch.stack((pixel_grid_u, pixel_grid_v, depth_map), dim=2)
        return vertices

    @staticmethod
    def compute_normals(self, vertices, normalize_gradients=False):
        H, W, C = vertices.shape

        image = vertices.permute(2,0,1).view(C, 1, H, W)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=vertices.dtype, device=vertices.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=vertices.dtype, device=vertices.device).view(1, 1, 3, 3)

        img_dx = F.conv2d(image, sobel_x, padding=1).view(C, H, W).permute(1, 2, 0)
        img_dy = F.conv2d(image, sobel_y, padding=1).view(C, H, W).permute(1, 2, 0)

        if normalize_gradients:
            mag = torch.sqrt((img_dx ** 2) + (img_dy ** 2) + 1e-8)
            img_dx = img_dx / mag
            img_dy = img_dy / mag

        normals = torch.linalg.cross(img_dx, img_dy)
        normals = normals / (torch.norm(normals, p=2, dim=-1, keepdim=True) + 1e-8)

        vertex_depths = vertices[:, :, -1]
        normals[vertex_depths==0] = 0

        return normals

    def warp_features(self, feature, u, v, mode="bilinear"):
        pass



