import torch
import torch.utils.data as data
# from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
import trimesh
from lib.datasets.utils import get_projection_matrix,get_rigid_transformation
import lib.utils.data_utils as if_nerf_dutils

class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split
        
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])
        test_view = cfg.test_view
        view = cfg.training_view if split == 'train' else test_view
        if len(view) == 0:
            view = [0]

        # prepare input images
        i = 0
        i = i + cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame
        if cfg.test_novel_pose:
            i = (i + cfg.num_train_frame) * i_intv
            ni = cfg.num_novel_pose_frame
            if self.human == 'CoreView_390':
                i = 0
 
        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i:i + ni][::i_intv]
        ]).ravel()

        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][i:i + ni][::i_intv]
        ]).ravel()
        self.rawcam_inds = np.array([
            np.array(cfg.raw_view)[view]
            for ims_data in annots['ims'][i:i + ni][::i_intv]
        ]).ravel()

        self.num_cams = len(view)

        joints = np.load(os.path.join(self.data_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        parents = np.load(os.path.join(self.data_root, 'parents.npy'), allow_pickle=True)
        self.parents = np.array(parents)
        self.split = split

    def set_pytorch3d_intrinsic_matrix(self, K, H, W):
        fx = -K[0, 0] * 2.0 / W
        fy = -K[1, 1] * 2.0 / H
        px = -(K[0, 2] - W / 2.0) * 2.0 / W
        py = -(K[1, 2] - H / 2.0) * 2.0 / H
        K = [
            [fx, 0, px, 0],
            [0, fy, py, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
        K = np.array(K)
        return K
    
    def get_mask(self, index):

        msk_path = os.path.join(self.data_root, "training/foregroundSegmentation" + self.ims[index][6:])
        mask = imageio.imread(msk_path)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        msk = mask / 255

        return msk
   
    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, 'vertices',
                                     '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)
        nxyz = np.zeros_like(xyz).astype(np.float32)
        vert = xyz

        boundthr = 0.05
        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)

        if cfg.big_box:
            min_xyz -= boundthr
            max_xyz += boundthr
        else:
            min_xyz[2] -= boundthr
            max_xyz[2] += boundthr

        can_bounds = np.stack([min_xyz, max_xyz], axis=0)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, 'smpl_params',
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= boundthr
            max_xyz += boundthr
        else:
            min_xyz[2] -= boundthr
            max_xyz[2] += boundthr
        bounds = np.stack([min_xyz, max_xyz], axis=0)

        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A = get_rigid_transformation(poses, joints, parents)
        
        params_path = os.path.join(self.data_root, 'smpl_params',
                                   '{}.npy'.format(0))
        paramshape = np.load(params_path, allow_pickle=True).item()

        return can_bounds, bounds, Rh, Th, vert, A, params['poses'].reshape(-1), paramshape['shapes'].reshape(-1)
        
    def get_sampling_points(self, bounds, N_samples):
        min_xyz = bounds[0]
        max_xyz = bounds[1]
        x_vals = np.random.uniform(0, 1, N_samples)
        y_vals = np.random.uniform(0, 1, N_samples)
        z_vals = np.random.uniform(0, 1, N_samples)
        vals = np.stack([x_vals, y_vals, z_vals], axis=1)
        pts = (max_xyz - min_xyz) * vals + min_xyz
        pts = pts.astype(np.float32)
        return pts
        
    def get_bounds(self, xyz):
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz -= 0.05
        max_xyz += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        bounds = bounds.astype(np.float32)
        return bounds
   
    def get_cloth(self, i):
        cloth_path = os.path.join(self.data_root, "codim_mesh", f"{i}.obj")
        cloth_mesh = trimesh.load(cloth_path)
        
        return cloth_mesh.vertices
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, "training", self.ims[index])  # self.data_root
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk = self.get_mask(index)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])

        # Pytorch3D 坐标系 (w2c)
        PYTORCH3D_R = np.array(self.cams['R'][cam_ind])
        PYTORCH_T = np.array(self.cams['T'][cam_ind]) / 1000.
        PYTORCH_RT = np.concatenate([PYTORCH3D_R, PYTORCH_T], axis=1).astype(np.float32)
        # OPENCV(COLMAP) 坐标系 (c2w)
        # 因为pytorch3d相机使用的旋转矩阵为c2w,而位移矩阵为w2c,故上述PYTORCH3D的旋转位移矩阵以w2c表示
        # 3DGS 在渲染时使用COLMAP的坐标系，因此需要将PYTORCH3D的坐标系转换为COLMAP坐标系
        c2w = np.linalg.inv(np.concatenate([PYTORCH_RT, [[0.0, 0.0, 0.0, 1.0]]], axis=0))
        opencv_R =  np.stack([c2w[:3, 0], c2w[:3, 1], c2w[:3, 2]], 1)
        opencv_T = c2w[:3, 3:]
        opencv_RT = np.concatenate([opencv_R, opencv_T], axis=1).astype(np.float32)
        opencv_RT = np.concatenate([opencv_RT, [[0.0, 0.0, 0.0, 1.0]]], axis=0).astype(np.float32)
        c2w = opencv_RT

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)

        if cfg.mask_bkgd:
            img[msk == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1
        K[:2] = K[:2] * cfg.ratio


        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        else:
            #i = int(os.path.basename(img_path)[:-4])
            i = int(os.path.basename(img_path).split('_')[4][:-4])
            frame_index = i

        can_bounds, bounds, Rh, Th, vert, A, smplpose, smplshape = self.prepare_input(i)

        snug_cloth = self.get_cloth(i)
        mask_at_box = np.ones([H, W])
        if self.split != 'train':
            near, far, mask_at_box = if_nerf_dutils.sample_ray(img, msk, K, PYTORCH3D_R, PYTORCH_T, can_bounds)
        pytorch3d_K = self.set_pytorch3d_intrinsic_matrix(K, cfg.H, cfg.W)
        
        # FOV
        fovx = 2 * np.arctan(W / (2 * K[0, 0]))
        fovy = 2 * np.arctan(H / (2 * K[1, 1]))
        # 相机投影矩阵和相机位置 3DGS所需
        # TODO: 手动求NEAR和FAR
        zfar = 100.0 # max(zfar, 100.0)
        znear = 0.01 # min(znear, 0.01)
        world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float()
        projection_matrix = get_projection_matrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1).float()
        # 由于3DGS原始代码默认相机内参中的cx,cy为weight/2,height/2.此处修改为实际相机内参
        projection_matrix[..., 2, 0] = - pytorch3d_K[0, 2]
        projection_matrix[..., 2, 1] = - pytorch3d_K[1, 2]
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0).numpy()
        camera_center = world_view_transform.inverse()[3, :3]
        cam_intrinsics = K
        
        ret = {
            'img': img,
            'msk': msk,
            'vert': vert,
            'A': A,
            'smplpose': smplpose,
            'smplshape': smplshape,
            "fovx": fovx,
            "fovy": fovy,
            "world_view_transform": world_view_transform,
            "c2w": c2w,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center,
            "cam_intrinsics": cam_intrinsics,
            'mask_at_box': mask_at_box,
            'snug_cloth':snug_cloth
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = frame_index - cfg.begin_ith_frame
        #latent_index = newfrmidx
        if cfg.test_novel_pose:
            latent_index = cfg.num_train_frame - 1
        meta = {
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'view_index': cam_ind,
            'cam_ind': cam_ind,
        }
        ret.update(meta)

        R0 = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {'R0_snap': R0, 'Th0_snap': Th, 'K': K.astype(np.float32), 'RT': PYTORCH_RT, 'pytorch3d_K': pytorch3d_K}
        ret.update(meta)
        
        

        return ret

    def __len__(self):
        return len(self.ims)
