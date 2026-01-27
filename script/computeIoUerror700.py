import numpy as np
import trimesh
import cv2
import os
import imageio
import torch
from pytorch3d.renderer.cameras import PerspectiveCameras, OrthographicCameras

from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
import pytorch3d.structures as struct

def set_pytorch3d_intrinsic_matrix(K, H, W):
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

def load_obj2(path):
    model = {}
    pts = []
    faces = []

    with open(path) as file:
        while True:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                pts.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "f":
                faces.append((int(strs[1])-1, int(strs[2])-1, int(strs[3])-1))

        model['pts'] = np.array(pts)
        model['faces'] = np.array(faces)

    return model
    
def IoU_acc(mask, render_depth):
    # mask:[B, H, W], render_depth:[B, H, W]
    intersection = np.bitwise_and(mask, render_depth)
    union = np.bitwise_or(mask, render_depth)

    intersection_num = np.sum(intersection[intersection == 1])
    union_num = np.sum(union[union == 1])

    res = intersection_num / union_num
    return res
    
    
data_root = 'data/magdalena/magdalena2000-allviews'
# data_root = 'data/h36m/S9/Posing'
exp_name = 'anisdf_idg'

# faces_path = os.path.join(data_root, 'templatemodel/modeltri.txt')
# faces = np.loadtxt(faces_path)-1
        
annots_path = os.path.join(data_root, 'annots.npy')
annots = np.load(annots_path, allow_pickle=True).item()
cameras = annots['cams']
ims = annots['ims']

Ks = np.array(cameras['K'])
Rs = np.array(cameras['R'])
Ts = np.array(cameras['T']).transpose(0, 2, 1) / 1000
Ds = np.array(cameras['D'])

height = 1024
width = 1024

#mesh_paths = 'data/magdalena600-allviews/deformedcansmpl/frmmodel'
#mesh_paths = 'data/magdalena2000-allviews/deformedcansmpl/500-600our'
#mesh_paths = sorted(mesh_paths)

# allerr = np.ones((14,300),dtype=np.float64) 
# allerr = np.ones((14,300),dtype=np.float64) 
allerr = np.ones((4,300),dtype=np.float64) 
#for mesh_path in mesh_paths:

sum_acc = 0
count = 0
# for i,view_idx in enumerate(range(0,14)):
# for i,view_idx in enumerate([0,1,2,4,5,7,8,9,10,11,12]):
# for i,view_idx in enumerate([0,5,9,11]):

for i,view_idx in enumerate([1,2,4,7,8,10,12]):
    # renderpath = os.path.join(data_root,
                                   # 'deformedcansmpl/frmmodel/render_view{:d}/'.format(view_idx))
    # os.makedirs(renderpath, exist_ok=True)
    for frame_index in range(700,1000):
        #if frame_index!=485 and frame_index!=484 and frame_index!=535:
        mesh_path = os.path.join("final_paper2/deformedverts{:04d}.obj".format(frame_index))
        # mesh_path = os.path.join(f"final_paper/deformcloth{frame_index}.obj")
        #mesh = trimesh.load(mesh_path)
        # mesh = np.load(mesh_path, allow_pickle=True).item()
        # mesh = trimesh.Trimesh(mesh['vertex'], mesh['triangle'])
        #vertices = mesh.vertices
        model = load_obj2(mesh_path)

        K = Ks[view_idx]
        R = Rs[view_idx]
        T = Ts[view_idx]
        pytorch3d_K = set_pytorch3d_intrinsic_matrix(K, height, width)
        cameras = PerspectiveCameras(device='cuda',
                                     K=pytorch3d_K[None].astype(np.float32),
                                     R=R.T[None].astype(np.float32),
                                     T=T.astype(np.float32))

        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=(height, width),
            faces_per_pixel=100,
        )
        #blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        #    faces_per_pixel=100,
        #lights = PointLights(device='cuda', location=[[0.0, 0.0, -3.0]])
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader()
        )
    #shader=SoftSilhouetteShader(blend_params=blend_params)
        vertex = torch.FloatTensor(model['pts']).cuda()[None]
        triangle = torch.LongTensor(model['faces']).cuda()[None]

        #verts_rgb = torch.ones_like(vertex)  # (1, V, 3)
        #textures = TexturesVertex(verts_features=verts_rgb)

        ppose = struct.Meshes(verts=vertex, faces=triangle)

        silhouette = silhouette_renderer(meshes_world=ppose)
        silhouette = torch.where(silhouette > 0, torch.ones_like(silhouette), torch.zeros_like(silhouette))
        silhouette = silhouette.detach().cpu().numpy()
        mask_render = silhouette.squeeze()[..., 3]

        # renderfilepath = os.path.join(renderpath, '{:d}.png'.format(frame_index))
        # cv2.imwrite(renderfilepath, mask_render*255)
        #mask_render = cv2.imread(renderfilepath)
        #mask_render = mask_render[...,0]/255

        # maskfilepath = os.path.join(data_root, 'mask1/{:d}/'.format(view_idx))
        # mask_path = os.path.join(maskfilepath,
                               # 'image_c_{:d}_f_{:d}.png'.format(view_idx, frame_index))  #
        # mask = cv2.imread(mask_path)
        # mask = mask[...,0]/255
        
        msk_path = os.path.join('data/magdalena/magdalena2000-allviews/training', 'foregroundSegmentation/{:d}/'.format(view_idx)) + 'image_c_{:d}_f_{:d}.jpg'.format(view_idx, frame_index)
        mask = imageio.imread(msk_path)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        mask = mask / 255
        
        acc = IoU_acc(mask.astype(np.bool), mask_render.astype(np.bool))
        # allerr[i, frame_index-700] = acc
        print(view_idx, frame_index, acc)
        sum_acc += acc
        count += 1
    
print(sum_acc / count)        
# print(np.mean(allerr))        
# np.savetxt(os.path.join(data_root, 'allerr_our1114_S4.txt'),allerr)