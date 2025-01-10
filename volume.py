import torch
import numpy as np
import cv2
import glob
import open3d as o3d
from skimage import measure
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import time

def rigid_transform(xyz, transform):

  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
  xyz_t_h = np.dot(transform, xyz_h.T).T
  return xyz_t_h[:, :3]

def get_view_frustum(depth_im, cam_intr, cam_pose):
  im_h = depth_im.shape[0]
  im_w = depth_im.shape[1]
  max_depth = np.max(depth_im)
  view_frust_pts = np.array([
    (np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
    (np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
    np.array([0,max_depth,max_depth,max_depth,max_depth])
  ])
  view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
  return view_frust_pts

class TSDF():

    def __init__(self, vol_dim, intristics, voxel_size=0.01):

        # self.vol_dim = vol_dim
        self._vol_bnds = vol_dim
        self._voxel_size = voxel_size
        print(self._vol_bnds)
        self._vol_dim = np.ceil((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).copy(order='C').astype(int)
        self._vol_bnds[:,1] = self._vol_bnds[:,0]+self._vol_dim*self._voxel_size
        self._vol_origin = self._vol_bnds[:,0].copy(order='C').astype(np.float32)
        print( self._vol_origin)
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

    def integrate(self, depth_image, camera_pose, color_img, sdf_trunc=0.03):

        with torch.no_grad():

            world2cam = torch.inverse(torch.from_numpy(camera_pose)).float().cuda()
            pts_camera = torch.matmul(
                world2cam, torch.t(self.vox_Wcoords))
            z_points = pts_camera[2]



            x_pix = torch.round((pts_camera[0] * self.fx)/z_points + self.cx).int()
            y_pix = torch.round((pts_camera[1] * self.fy)/z_points + self.cy).int()



            valid_pix = (x_pix >= 0) & (x_pix < 640) & (
                y_pix >= 0) & (y_pix < 480) & (z_points > 0)
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
                            z_points_valid] = ((old_weights[:, None] * old_rgb) + valid_cols)/(old_weights[:, None]  + 1)

            self.weights[x_points_valid, y_points_valid,
                        z_points_valid] = self.weights[x_points_valid, y_points_valid, z_points_valid] + 1
        
            torch.cuda.empty_cache()

            return 0

    def ray_casting(self, camera_pose, sample_size=200):

        u = torch.arange(0,640).float()
        v = torch.arange(0,480).float()

        x_z = (u - self.cx)/self.fx
        y_z = (v - self.cy)/self.fy
        z = 1
        xv,yv,zv = torch.meshgrid([x_z, y_z, torch.tensor([1.0])])

        direction = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1)  

        direction  = direction/torch.linalg.vector_norm(direction, dim=0)

        print(direction.shape)

        p = self._vol_origin

        total_sample = torch.arange(sample_size) * self._voxel_size
        all_sample_points = total_sample[:, None] * direction[:, None, :]
        all_sample_points = all_sample_points 
        print(all_sample_points.shape)

        # now how to check intersection? what are the dimension of the tsdf 


        world2cam = torch.inverse(torch.from_numpy(camera_pose)).float()
        pts_camera = torch.matmul(
                world2cam, torch.t(self.vox_Wcoords.cpu())).T




        x_max = pts_camera[:,0].max()
        y_max = pts_camera[:,1].max()
        z_max = pts_camera[:,2].max()

        print(x_max,y_max,z_max)

        valid_hits = (all_sample_points[:,:, 0] > 0 ) & (all_sample_points[:,:, 0] < x_max ) & (all_sample_points[:,:, 1] > 0 ) & (all_sample_points[:,:, 1] < y_max ) & (all_sample_points[:,:, 2] > 0 ) & (all_sample_points[:,:,2] < x_max )

        intersection_pts = all_sample_points[valid_hits[:,:, None]]
        
        intersection_pts

        print(intersection_pts.shape)

        exit(1)

        return 0
    



if __name__ == "__main__":

    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
    worldpose= np.identity(4)

        

    print("Estimating voxel volume bounds...")
    n_imgs = 10
    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
    vol_bnds = np.zeros((3,2))
    for i in range(n_imgs):

        depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0  
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))  
        if i == 0:
            worldpose = cam_pose

        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))

    print("Initializing voxel volume...")


    vox_grid = TSDF(vol_bnds, voxel_size=0.02, intristics=cam_intr)
    vox_grid.ray_casting(worldpose)


    path_to_folder = "/home/zeus/masters/3DSMC/project/TSDF/data"

    depths = sorted(glob.glob(path_to_folder + "/" + "*.depth.png"))[:100]
    poses = sorted(glob.glob(path_to_folder + "/" + "*.pose.txt"))[:100]
    imgs = sorted(glob.glob(path_to_folder + "/" + "*.color.jpg"))[:100]


    for depth, pose, img in zip(depths, poses, imgs):
        start_time = time.time()
        depth_im = cv2.imread(depth, -1).astype(float)
        depth_im = torch.from_numpy(depth_im)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0

        cam_pose = np.loadtxt(pose)

        img = cv2.imread(img)
        img = torch.from_numpy(img)
        
        vox_grid.integrate(depth_im, cam_pose, img)
        end_time = time.time()

        print(1/(end_time - start_time))

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
