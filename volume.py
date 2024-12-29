import torch
import numpy as np
import cv2


class TSDF():

    def __init__(self, vol_dim=[100, 100, 100], voxel_size=2):

        self.vol_dim = vol_dim
        x_grid = torch.arange(0, self.vol_dim[0])
        y_grid = torch.arange(0, self.vol_dim[1])
        z_grid = torch.arange(0, self.vol_dim[2])
        xv, yv, zv = torch.meshgrid([x_grid, y_grid, z_grid])
        self.vox_coords = torch.stack(
            [xv.flatten(), yv.flatten(), zv.flatten()], dim=1).double()
        self.vox_coords = torch.cat([
            self.vox_coords, torch.ones(len(self.vox_coords), 1, )], dim=1).double()
        self.sdf_values = torch.ones(
            (self.vol_dim[0], self.vol_dim[1], self.vol_dim[2])).double()

        print(self.vox_coords.shape)

    def integrate(self, depth_image, camera_pose, intristics, trunc_value, sdf_trunc):

        pts_camera = torch.matmul(torch.from_numpy(
            camera_pose), torch.t(self.vox_coords))
        z_points = pts_camera[2]
        fx = intristics[0][0]
        fy = intristics[1][1]
        cx = intristics[0][2]
        cy = intristics[1][2]
        x_pix = torch.round((pts_camera[0] * fx)/z_points + cx).long()
        y_pix = torch.round((pts_camera[1] * fy)/z_points + cy).long()

        valid_pix = (x_pix >= 0) & (x_pix < 640) & (
            y_pix >= 0) & (y_pix < 480) & (z_points > 0)
        x_coords_valid = self.vox_coords[valid_pix, 0]
        y_coords_valid = self.vox_coords[valid_pix, 1]
        z_coords_valid = self.vox_coords[valid_pix, 2]
        depth_val = depth_image[y_pix[valid_pix], x_pix[valid_pix]]

        depth_diff = depth_val - z_coords_valid

        valid_sdf = (depth_diff <= trunc_value) & (depth_diff >= -trunc_value)

        dist = torch.clamp(depth_diff / sdf_trunc, max=1)

        valid_pix = (depth_val > 0) & (depth_diff > -sdf_trunc)
        x_points_valid = x_coords_valid[valid_sdf].int()
        y_points_valid = y_coords_valid[valid_sdf].int()
        z_points_valid = z_coords_valid[valid_sdf].int()
        valid_dist = dist[valid_sdf]
        self.sdf_values[x_points_valid, y_points_valid,
                        y_points_valid] = valid_dist
        # depth_values
        # print(pts_camera[2])
        # get gridspace into pixel space.
        return 0


if __name__ == "__main__":

    test = np.random.random(10)
    new = [True, True, True, True, False, True, True, False, False, False]
    print(test[new])
    depth_im = cv2.imread("data/frame-%06d.depth.png" % (0), -1).astype(float)
    depth_im = torch.from_numpy(depth_im)
    depth_im /= 1000.
    depth_im[depth_im == 65.535] = 0
    print(depth_im.shape)
    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
    cam_pose = np.loadtxt("data/frame-%06d.pose.txt" % (0))
    vox_grid = TSDF()
    vox_grid.integrate(depth_im, cam_pose, cam_intr,
                       trunc_value=1, sdf_trunc=1)
