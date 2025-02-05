import torch
import open3d as o3d
from skimage import measure
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


def rigid_transform(xyz, transform):

    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):

    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([
        (np.array([0, 0, 0, im_w, im_w])-cam_intr[0, 2])*np.array([0,
                                                                   max_depth, max_depth, max_depth, max_depth])/cam_intr[0, 0],
        (np.array([0, 0, im_h, 0, im_h])-cam_intr[1, 2])*np.array([0,
                                                                   max_depth, max_depth, max_depth, max_depth])/cam_intr[1, 1],
        np.array([0, max_depth, max_depth, max_depth, max_depth])
    ])
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


def get_vol_bnds(depth_im, cam_intr, cam_pose):
    vol_bnds = np.zeros((3, 2))

    view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:, 0] = np.minimum(
        vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:, 1] = np.maximum(
        vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    return vol_bnds


def get_mesh(vox_grid):
    sdf_numpy = vox_grid.sdf_values.cpu().numpy()
    color_sdf = vox_grid.rgb_values.cpu().numpy()
    voxel_size = 0.02

    verts, faces, norms, vals = measure.marching_cubes(sdf_numpy, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts*voxel_size
    # verts = verts*voxel_size + vox_grid._vol_origin

    y = np.arange(color_sdf.shape[1]) * voxel_size
    x = np.arange(color_sdf.shape[0]) * voxel_size
    z = np.arange(color_sdf.shape[2]) * voxel_size

    r_interpolator = RegularGridInterpolator((x, y, z), color_sdf[..., 0])
    g_interpolator = RegularGridInterpolator((x, y, z), color_sdf[..., 1])
    b_interpolator = RegularGridInterpolator((x, y, z), color_sdf[..., 2])

    r_values = r_interpolator(verts)
    g_values = g_interpolator(verts)
    b_values = b_interpolator(verts)

    vertex_colors = np.stack((b_values, g_values, r_values), axis=-1)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors / 255.0)

    o3d.visualization.draw_geometries([mesh])



class TSDF():

    def __init__(self, vol_dim, intristics, voxel_size=0.01):

        # self._vol_dim = vol_dim
        self._vol_bnds = vol_dim
        self._voxel_size = voxel_size
        self.device = "cuda:0"
        print(self._vol_bnds)
        self._vol_dim = np.ceil(
            (self._vol_bnds[:, 1]-self._vol_bnds[:, 0])/self._voxel_size).copy(order='C').astype(int)
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + \
            self._vol_dim*self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(
            order='C').astype(np.float32)
        print(self._vol_origin)
        print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
            self._vol_dim[0], self._vol_dim[1], self._vol_dim[2],
            self._vol_dim[0]*self._vol_dim[1]*self._vol_dim[2])
        )

        self.fx = intristics[0][0]
        self.fy = intristics[1][1]
        self.cx = intristics[0][2]
        self.cy = intristics[1][2]

        x_grid = torch.arange(0, self._vol_dim[0])
        y_grid = torch.arange(0, self._vol_dim[1])
        z_grid = torch.arange(0, self._vol_dim[2])
        xv, yv, zv = torch.meshgrid([x_grid, y_grid, z_grid])
        self.vox_coords = torch.stack(
            [xv.flatten(), yv.flatten(), zv.flatten()], dim=1).float()

        self.vox_Wcoords = torch.cat([
            (self.vox_coords * voxel_size) + self._vol_origin, torch.ones(len(self.vox_coords), 1, )], dim=1).float().cuda()

        self.vox_coords = torch.cat([
            self.vox_coords, torch.ones(len(self.vox_coords), 1, )], dim=1).float().cuda()

        self.sdf_values = torch.ones(
            (self._vol_dim[0], self._vol_dim[1], self._vol_dim[2])).float().cuda()

        self.rgb_values = torch.zeros(
            (self._vol_dim[0], self._vol_dim[1], self._vol_dim[2], 3)).float().cuda()

        self.weights = torch.zeros(
            (self._vol_dim[0], self._vol_dim[1], self._vol_dim[2])).float().cuda()

        self._vol_origin = torch.from_numpy(self._vol_origin).cuda()
        self._voxel_size = torch.asarray(self._voxel_size).cuda()

    def integrate(self, depth_image, camera_pose, color_img, sdf_trunc=0.03):
        map_width = 640
        map_height = 480
        with torch.no_grad():

            world2cam = torch.inverse(
                torch.from_numpy(camera_pose)).float().cuda()
            pts_camera = torch.matmul(
                world2cam, torch.t(self.vox_Wcoords))
            z_points = pts_camera[2]

            x_pix = torch.round(
                (pts_camera[0] * self.fx)/z_points + self.cx).int()
            y_pix = torch.round(
                (pts_camera[1] * self.fy)/z_points + self.cy).int()

            valid_pix = (x_pix >= 0) & (x_pix < map_width) & (
                y_pix >= 0) & (y_pix < map_height) & (z_points > 0)
            x_coords_valid = self.vox_coords[valid_pix, 0]
            y_coords_valid = self.vox_coords[valid_pix, 1]
            z_coords_valid = self.vox_coords[valid_pix, 2]

            z_pix = pts_camera[2, valid_pix]
            depth_image = depth_image.cuda()
            color_img = color_img.cuda()
            depth_val = depth_image[y_pix[valid_pix], x_pix[valid_pix]]
            rgb_val = color_img[y_pix[valid_pix], x_pix[valid_pix]]

            depth_diff = depth_val - z_pix

            dist = torch.clamp(depth_diff / sdf_trunc, max=1).float()

            valid_pts = (depth_val > 0.) & (depth_diff >= -sdf_trunc)

            x_points_valid = x_coords_valid[valid_pts].int()
            y_points_valid = y_coords_valid[valid_pts].int()
            z_points_valid = z_coords_valid[valid_pts].int()
            valid_dist = dist[valid_pts]
            valid_cols = rgb_val[valid_pts]

            old_sdf = self.sdf_values[x_points_valid, y_points_valid,
                                      z_points_valid]

            old_rgb = self.rgb_values[x_points_valid, y_points_valid,
                                      z_points_valid]

            old_weights = self.weights[x_points_valid, y_points_valid,
                                       z_points_valid]

            self.sdf_values[x_points_valid, y_points_valid,
                            z_points_valid] = ((old_weights * old_sdf) + valid_dist)/(old_weights + 1)

            self.rgb_values[x_points_valid, y_points_valid,
                            z_points_valid] = ((old_weights[:, None] * old_rgb) + valid_cols)/(old_weights[:, None] + 1)

            self.weights[x_points_valid, y_points_valid,
                         z_points_valid] = self.weights[x_points_valid, y_points_valid, z_points_valid] + 1

            print("integrate", self.sdf_values.min())
            torch.cuda.empty_cache()

            return 0

    def get_normals(self):
        """Compute normal volume
        """
        nx, ny, nz = self._vol_dim
        device = self.device
        dx = torch.cat([(self.sdf_values[1:, :, :] - self.sdf_values[:-1, :, :]) /
                       self._voxel_size, torch.zeros(1, ny, nz).to(device)], dim=0)
        dy = torch.cat([(self.sdf_values[:, 1:, :] - self.sdf_values[:, :-1, :]) /
                       self._voxel_size, torch.zeros(nx, 1, nz).to(device)], dim=1)
        dz = torch.cat([(self.sdf_values[:, :, 1:] - self.sdf_values[:, :, :-1]) /
                       self._voxel_size, torch.zeros(nx, ny, 1).to(device)], dim=2)
        norms = torch.stack([dx, dy, dz], -1)
        n = torch.norm(norms, dim=-1)
        outliers_mask = n > 1. / (2 * self._voxel_size)
        norms[outliers_mask] = 0.
        eps = 1e-7
        non_zero_grad = n > eps
        norms[non_zero_grad, :] = norms[non_zero_grad, :] / \
            n[non_zero_grad][:, None]
        return norms

    def get_nn(self, field_vol, coords_w):
        """Get nearest-neigbor values from a given volume
        """
        field_dim = field_vol.shape
        assert len(field_dim) == 3 or len(field_dim) == 4
        vox_coord_float = (
            coords_w - self._vol_origin[None, :]) / self._voxel_size
        vox_coord = torch.floor(vox_coord_float)
        vox_offset = vox_coord_float - vox_coord  # [N, 3]
        vox_coord[vox_offset >= 0.5] += 1.
        vox_coord[:, 0] = torch.clamp(
            vox_coord[:, 0], 0., self._vol_dim[0] - 1)
        vox_coord[:, 1] = torch.clamp(
            vox_coord[:, 1], 0., self._vol_dim[1] - 1)
        vox_coord[:, 2] = torch.clamp(
            vox_coord[:, 2], 0., self._vol_dim[2] - 1)
        vox_coord = vox_coord.long()
        vx, vy, vz = vox_coord[:, 0], vox_coord[:, 1], vox_coord[:, 2]
        v_nn = field_vol[vx, vy, vz]
        return v_nn

    def tril_interp(self, field_vol, coords_w):
        """Get tri-linear interpolated value from a given volume
        """
        field_dim = field_vol.shape
        assert len(field_dim) == 3 or len(field_dim) == 4
        n_pts = coords_w.shape[0]
        vox_coord = torch.floor(
            (coords_w - self._vol_origin[None, :]) / self._voxel_size).long()  # [N, 3]

        # for border points, don't do interpolation
        non_border_mask = (vox_coord[:, 0] < self._vol_dim[0] - 1) & (vox_coord[:, 1] < self._vol_dim[1] - 1) & \
                          (vox_coord[:, 2] < self._vol_dim[2] - 1)
        v_interp = torch.zeros(n_pts) if len(
            field_dim) == 3 else torch.zeros(n_pts, field_vol.shape[-1])
        v_interp = v_interp.to(self.device)
        vx_, vy_, vz_ = vox_coord[~non_border_mask,
                                  0], vox_coord[~non_border_mask, 1], vox_coord[~non_border_mask, 2]
        v_interp[~non_border_mask] = field_vol[vx_, vy_, vz_]

        vx, vy, vz = vox_coord[non_border_mask, 0], vox_coord[non_border_mask,
                                                              1], vox_coord[non_border_mask, 2]  # [N]
        vox_idx = vz + vy * self._vol_dim[-1] + \
            vx * self._vol_dim[-1] * self._vol_dim[-2]
        vertices_coord = self.vox_Wcoords[vox_idx][:, :3]  # [N, 3]
        r = (coords_w[non_border_mask] - vertices_coord) / self._voxel_size
        rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
        if len(field_dim) == 4:
            rx = rx.unsqueeze(1)
            ry = ry.unsqueeze(1)
            rz = rz.unsqueeze(1)
        v000 = field_vol[vx, vy, vz]
        v001 = field_vol[vx, vy, vz+1]
        v010 = field_vol[vx, vy+1, vz]
        v011 = field_vol[vx, vy+1, vz+1]
        v100 = field_vol[vx+1, vy, vz]
        v101 = field_vol[vx+1, vy, vz+1]
        v110 = field_vol[vx+1, vy+1, vz]
        v111 = field_vol[vx+1, vy+1, vz+1]
        v_interp[non_border_mask] = v000 * (1 - rx) * (1 - ry) * (1 - rz) \
            + v001 * (1 - rx) * (1 - ry) * rz \
            + v010 * (1 - rx) * ry * (1 - rz) \
            + v011 * (1 - rx) * ry * rz \
            + v100 * rx * (1 - ry) * (1 - rz) \
            + v101 * rx * (1 - ry) * rz \
            + v110 * rx * ry * (1 - rz) \
            + v111 * rx * ry * rz

        return v_interp

    def get_pts_inside(self, pts, margin=0):
        vox_coord = torch.floor((pts - self._vol_origin[None, :].to(
            self.device)) / self._voxel_size.to(self.device)).long()  # [N, 3]
        valid_pts_mask = (vox_coord[..., 0] >= margin) & (vox_coord[..., 0] < self._vol_dim[0] - margin) \
            & (vox_coord[..., 1] >= margin) & (vox_coord[..., 1] < self._vol_dim[1] - margin) \
            & (vox_coord[..., 2] >= margin) & (vox_coord[..., 2] < self._vol_dim[2] - margin)
        return valid_pts_mask

    @torch.no_grad()
    def render_model(self, c2w, intri, imh, imw, near=500., far=5000., n_samples=192):

        c2w = torch.from_numpy(c2w).float()
        c2w = c2w.to(self.device)
        rays_o, rays_d = self.get_rays(c2w, intri, imh, imw)  # [h, w, 3]
        z_vals = torch.linspace(near, far, n_samples).to(rays_o)  # [n_samples]
        ray_pts_w = (rays_o[:, :, None, :] + rays_d[:, :, None, :] *
                     z_vals[None, None, :, None]).to(self.device)  # [h, w, n_samples, 3]

        tsdf_vals = torch.ones(imh, imw, n_samples).to(self.device)
        valid_ray_pts_mask = self.get_pts_inside(ray_pts_w)
        valid_ray_pts = ray_pts_w[valid_ray_pts_mask]  # [n_valid, 3]
        tsdf_vals[valid_ray_pts_mask] = self.tril_interp(
            self.sdf_values, valid_ray_pts)

        sign_matrix = torch.cat([torch.sign(tsdf_vals[..., :-1] * tsdf_vals[..., 1:]),
                                 torch.ones(imh, imw, 1).to(self.device)], dim=-1)  # [h, w, n_samples]
        cost_matrix = sign_matrix * \
            torch.arange(
                n_samples, 0, -1).float().to(self.device)[None, None, :]  # [h, w, n_samples]

        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        hs, ws = torch.meshgrid(torch.arange(imh), torch.arange(imw))
        mask_pos_to_neg = tsdf_vals[hs, ws, indices] > 0
        inside_vol = self.get_pts_inside(ray_pts_w[hs, ws, indices])
        hit_surface_mask = mask_sign_change & mask_pos_to_neg & inside_vol
        # [n_surf_pts, 3]
        hit_pts = ray_pts_w[hs, ws, indices][hit_surface_mask]

        # compute normals
        norms = self.get_normals()
        surf_tsdf = self.tril_interp(self.sdf_values, hit_pts)  # [n_surf_pts]
        surf_norms = self.get_nn(norms, hit_pts)
        updated_hit_pts = hit_pts - surf_tsdf[:, None] * 0.03 * surf_norms
        valid_mask = self.get_pts_inside(updated_hit_pts)
        hit_pts[valid_mask, :] = updated_hit_pts[valid_mask, :]

        # get depth values
        w2c = torch.inverse(c2w).to(self.device)
        hit_pts_c = (w2c[:3, :3] @ hit_pts.transpose(1, 0)
                     ).transpose(1, 0) + w2c[:3, 3][None, :]
        hit_pts_z = hit_pts_c[:, -1]
        depth_rend = torch.zeros(imh, imw).to(self.device)
        depth_rend[hit_surface_mask] = hit_pts_z

        # vertex map
        vertex_rend = torch.zeros(imh, imw, 3).to(self.device)
        vertex_rend[hit_surface_mask] = hit_pts_c
        # normal map
        surf_norms_c = (w2c[:3, :3] @ surf_norms.transpose(1, 0)
                        ).transpose(1, 0)  # [h, w, 3]
        normal_rend = torch.zeros(imh, imw, 3).to(self.device)
        normal_rend[hit_surface_mask] = surf_norms_c

        if self.rgb_values is not None:
            hit_colors = self.tril_interp(self.rgb_values, hit_pts)
            color_rend = torch.zeros(imh, imw, 3).to(self.device)
            color_rend[hit_surface_mask] = hit_colors
        else:
            color_rend = None

        # import matplotlib.pyplot as plt

        # plt.subplot(1, 3, 1)
        # plt.imshow(depth_rend.cpu().numpy()[-1,1], cmap='gray')
        # plt.title("Depth Map")

        # plt.subplot(1, 3, 2)
        # plt.imshow(normal_rend.cpu().numpy() * 0.5 + 0.5)  # Normalize to [0,1]
        # plt.title("Normal Map")

        # plt.subplot(1, 3, 3)
        # plt.imshow(vertex_rend.cpu().numpy())
        # plt.title("Vertex Map")

        # plt.show()

        return depth_rend, color_rend, vertex_rend, normal_rend, hit_surface_mask

    def render_pyramid(self, c2w, intri, imh, imw, n_pyr=4, near=500., far=5000., n_samples=192):
        K = intri.clone()
        dep_pyr, rgb_pyr, vtx_pyr, nrm_pyr, mask_pyr = [], [], [], [], []
        for l in range(n_pyr):
            dep, rgb, feat, vtx, nrm, mask = self.render_model(
                c2w, K, imh, imw, near=near, far=far, n_samples=n_samples)
            dep_pyr += [dep]
            rgb_pyr += [rgb]
            vtx_pyr += [vtx]
            nrm_pyr += [nrm]
            mask_pyr += [mask]
            imh = imh // 2
            imw = imw // 2
            K /= 2
        return dep_pyr, rgb_pyr, vtx_pyr, nrm_pyr, mask_pyr

    def get_voxel_idx(self, x):

        assert len(x.shape) == 2, print("only accept flattened input!!!")
        x.to(self.device)
        vox_coord = torch.floor(
            (x - self._vol_origin[None, :]) / self._voxel_size)  # [N, 3]
        vx, vy, vz = vox_coord[:, 0], vox_coord[:, 1], vox_coord[:, 2]
        vox_idx = vz + vy * self._vol_dim[-1] + \
            vx * self._vol_dim[-1] * self._vol_dim[-2]
        return vox_idx.long()

    def get_rays(self, c2w, intrinsics, H, W):
        device = self.device
        # c2w = torch.from_numpy(c2w).float()
        # c2w = c2w.to(device)
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                              torch.linspace(0, H - 1, H))
        i = i.t().to(device).reshape(H * W)  # [hw]
        j = j.t().to(device).reshape(H * W)  # [hw]

        dirs = torch.stack([(i - cx) / fx, (j - cy) / fy,
                           torch.ones_like(i)], -1).to(device)  # [hw, 3]
        dirs = dirs.transpose(1, 0)  # [3, hw]
        rays_d = (c2w[:3, :3] @ dirs).transpose(1, 0)  # [hw, 3]
        rays_o = c2w[:3, 3].expand(rays_d.shape)

        return rays_o.reshape(H, W, 3), rays_d.reshape(H, W, 3)




## Usage 
'''

    vol_bnds = get_vol_bnds(....)
    vox_grid = TSDF(vol_bnds, voxel_size=0.02, intristics=cam_intr)

    for depth, pose, img in zip(depths, poses, imgs):

        vox_grid.integrate(depth_im, cam_pose, img)

        depth_rend, color_rend, vertex_rend, normal_rend, hit_surface_mask = vox_grid.render_model(cam_pose, cam_intr, map_width, map_height)


'''