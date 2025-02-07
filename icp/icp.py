import torch
import torch.nn.functional as F


class ICP(torch.nn.Module):
    def __init__(self, optimizer=None, occlusion_threshold=750, symmetric_error=False):
        super().__init__()
        self.optimizer = optimizer
        self.occlusion_threshold = occlusion_threshold
        self.symmetric_error = symmetric_error

    def forward(self, depth_source, depth_target, pose, K):
        max_iterations = 1
        if hasattr(self.optimizer, "max_iterations"):
            max_iterations = self.optimizer.max_iterations

        vertices_source = self.compute_vertices(depth_source, K)
        if self.symmetric_error:
            normals_source = self.compute_normals(vertices_source)

        vertices_target = self.compute_vertices(depth_target, K)
        normals_target = self.compute_normals(vertices_target)

        H, W, C = vertices_source.shape
        for i in range(max_iterations):
            R = pose[:3, :3]
            t = pose[:3, -1]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            # x̃ = Rx + t
            # ñ = Rn
            vertices_transformed = torch.matmul(vertices_source, R.T) + t
            if self.symmetric_error:
                normals_transformed = torch.matmul(normals_source, R.T)

            u_transformed = (vertices_transformed[:, :, 0] / vertices_transformed[:, :, 2]) * fx + cx
            v_transformed = (vertices_transformed[:, :, 1] / vertices_transformed[:, :, 2]) * fy + cy

            # projective data association
            u_norm = u_transformed / ((W - 1) / 2) - 1
            v_norm = v_transformed / ((H - 1) / 2) - 1
            # u_norm, v_norm = u_transformed, v_transformed

            uv_grid = torch.cat((u_norm.view(1, H, W, 1), v_norm.view(1, H, W, 1)), dim=-1)

            vertices_target_warped = self.warp_features(vertices_target, uv_grid)
            normals_target_warped = self.warp_features(normals_target, uv_grid)

            normals = normals_target_warped
            if self.symmetric_error:
                normals += normals_transformed
                normals = F.normalize(normals, p=2, dim=-1)

            diff = vertices_transformed - vertices_target_warped
            residuals = (diff * normals).sum(dim=-1)

            mask_source = vertices_source[:, :, 2] <= 0
            mask_target = vertices_target_warped[:, :, 2] <= 0
            out_of_view_pixels = (u_transformed <= 0) | (u_transformed >= W-1) | (v_transformed <= 0) | (v_transformed >= H-1)
            # TODO: Test effects of occlusion mask and estimate how many pixels it is masking on average
            occlusion_mask = diff.norm(p=2, dim=-1) > self.occlusion_threshold
            invalid = occlusion_mask.sum()
            #print("INVALIIIID: ")
            #print(invalid)
            mask = mask_source | mask_target | out_of_view_pixels | occlusion_mask
            # print(torch.sum(mask_source), torch.sum(mask_target), torch.sum(out_of_view_pixels), torch.sum(occlusion_mask))
            # print(mask.shape, torch.sum(mask))
            # Perform linear least squares if no optimizer provided
            if self.optimizer is None:
                vertices_transformed = vertices_transformed.view(-1, 3)
                normals = normals.view(-1, 3)
                residuals = residuals.view(-1, 1)
                mask = mask.view(-1)

                A = torch.matmul(ICP.generate_3d_skew_symmetric_matrix(vertices_transformed.view(-1, 3)), normals.view(-1, 3).unsqueeze(-1)).squeeze(-1)
                A = torch.cat((A, normals), dim=-1)

                b = -residuals

                A[mask] = 0.
                b[mask] = 0.

                # Linear solver for A @ xi = b
                if A.device.type == 'mps':
                    A = A.to("cpu")
                    b = b.to("cpu")
                    optimized_parameters, residuals, rank, _ = torch.linalg.lstsq(A, b)
                    optimized_parameters = optimized_parameters.to("mps")
                else:
                    optimized_parameters, residuals, rank, _ = torch.linalg.lstsq(A, b)
                pose = self.construct_pose_from_parameters(optimized_parameters)
                return pose
            Jf = self.compute_jacobian(vertices_transformed, normals)
            residuals[mask] = 0.
            residuals = residuals.view(H*W, 1, 1)

            # print("loss:", torch.linalg.norm(residuals))

            Jf[mask] = 0.
            Jf = Jf.view(H*W, 1, -1)
            delta_parameters = self.optimizer(residuals, Jf)
            pose = self.exp_se3(delta_parameters) @ pose
        return pose

    @staticmethod
    def compute_vertices(depth_map, K):
        H, W = depth_map.shape
        device = depth_map.device
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        pixel_grid_v, pixel_grid_u = torch.meshgrid([torch.arange(0, W), torch.arange(0, H)], indexing="xy")
        pixel_grid_u = pixel_grid_u.to(device)
        pixel_grid_v = pixel_grid_v.to(device)

        pixel_grid_u = ((pixel_grid_u - cx) / fx) * depth_map
        pixel_grid_v = ((pixel_grid_v - cy) / fy) * depth_map

        vertices = torch.stack((pixel_grid_u, pixel_grid_v, depth_map), dim=2)
        return vertices

    @staticmethod
    def compute_normals(vertices, normalize_gradients=False):
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
        normals = F.normalize(normals, p=2, dim=-1)

        vertex_depths = vertices[:, :, -1]
        normals[vertex_depths==0] = 0

        return normals

    @staticmethod
    def warp_features(feature, uv_grid, mode="bilinear"):
        device = feature.device

        # mps does not yet support F.grid_sample(), it produces a lot of NaNs
        if device.type == "mps":
            uv_grid = uv_grid.detach().cpu()
            feature = feature.detach().cpu()

        padding_mode = "border"
        # if uv_grid.device.type == "mps":
        #     # https://github.com/pytorch/pytorch/issues/125098
        #     padding_mode = "zeros"
        #     uv_grid = uv_grid.clamp(-1, 1)

        feature_warped = F.grid_sample(feature.unsqueeze(0).permute(0, 3, 1, 2), uv_grid, mode=mode, padding_mode=padding_mode, align_corners=True).squeeze()
        return feature_warped.permute(1, 2, 0).to(device)

    @staticmethod
    def compute_jacobian(vertices_transformed, normals):
        H, W, C = vertices_transformed.shape
        J_translation = normals.view(-1, 3)

        J_rotation = torch.matmul(ICP.generate_3d_skew_symmetric_matrix(vertices_transformed.view(-1, 3)), normals.view(-1, 3).unsqueeze(-1)).squeeze(-1)

        Jf = torch.cat((J_rotation, J_translation), dim=-1).view(H, W, -1)
        return Jf

    @staticmethod
    def generate_3d_skew_symmetric_matrix(vector):
        v1, v2, v3 = vector[:, 0], vector[:, 1], vector[:, 2]
        o = torch.zeros_like(v1)

        skew_symmetric_matrix = torch.stack([torch.stack([o, -v3, v2]), torch.stack([v3, o, -v1]), torch.stack([-v2, v1, o])]).permute(2, 0, 1)
        return skew_symmetric_matrix

    @staticmethod
    def exp_se3(xi):
        eps = 1e-8

        w = xi[:3].squeeze()
        v = xi[3:6].squeeze()
        w_hat = torch.tensor([[0., -w[2], w[1]],
                              [w[2], 0., -w[0]],
                              [-w[1], w[0], 0.]]).to(xi)
        w_hat_second = torch.mm(w_hat, w_hat).to(xi)

        theta = torch.norm(w)
        theta_2 = theta ** 2
        theta_3 = theta ** 3
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        eye_3 = torch.eye(3).to(xi)

        if theta <= eps:
            e_w = eye_3
            j = eye_3
        else:
            e_w = eye_3 + w_hat * sin_theta / theta + w_hat_second * (1. - cos_theta) / theta_2
            k1 = (1 - cos_theta) / theta_2
            k2 = (theta - sin_theta) / theta_3
            j = eye_3 + k1 * w_hat + k2 * w_hat_second

        T = torch.eye(4).to(xi)
        T[:3, :3] = e_w
        T[:3, 3] = torch.mv(j, v)

        return T

    @staticmethod
    def construct_pose_from_parameters(parameters):
        pose = torch.tensor([
            [1., parameters[0]*parameters[1]-parameters[2], parameters[0]*parameters[2]+parameters[1], parameters[3]],
            [parameters[2], parameters[0]*parameters[1]*parameters[2]+1, parameters[1]*parameters[2]-parameters[0], parameters[4]],
            [-parameters[1], parameters[0], 1, parameters[5]],
            [0, 0, 0, 1]
        ]).to(parameters.device)
        return pose

