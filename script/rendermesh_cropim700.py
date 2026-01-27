import numpy as np

import pyrender
import trimesh
import cv2
#import math
import os
#import glob


def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


def make_rotate(rx, ry, rz):

    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R


class Renderer(object):
    def __init__(self, focal_length=1000, height=512, width=512):
        self.height = height
        self.width = width
        self.renderer = pyrender.OffscreenRenderer(height, width)
        self.focal_length = focal_length

    def render(self, vertices, K, R, T, i, viewidx, return_depth=False):
        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180),
                                                      [1, 0, 0])

        self.renderer.viewport_height = self.height
        self.renderer.viewport_width = self.width
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.5, 0.5, 0.5))
        camera_pose = np.eye(4)
        camera = pyrender.camera.IntrinsicsCamera(fx=K[0, 0],
                                                  fy=K[1, 1],
                                                  cx=K[0, 2],
                                                  cy=K[1, 2])
        scene.add(camera, pose=camera_pose)
        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)

        mesh = trimesh.load(mesh_path, process=False)
        # mesh = np.load(mesh_path, allow_pickle=True).item()
        # mesh = trimesh.Trimesh(mesh['vertex'], mesh['triangle'])
        vertices = mesh.vertices
        # vertices = vertices * 0.005
        #i = int(os.path.basename(mesh_path)[:-4])
        # vertices = vertices + np.load('bounds.npy')[i][0]

        vertices = vertices @ R.T + T
        mesh.vertices = vertices

        mesh.apply_transform(rot)
        normals = compute_normal(mesh.vertices, mesh.faces)
        colors = ((0.5 * normals + 0.5) * 255).astype(np.uint8)#173, 174, 232
        #colors = np.array([142, 144, 191]).reshape(1,3).astype(np.uint8)
        #colors = np.tile(colors, mesh.vertices.shape[0]).reshape(-1,3)
        mesh.visual.vertex_colors[:, :3] = colors
        mesh.visual.vertex_colors[0:2421, :3] = np.array([40, 140, 214]).astype(np.uint8)

        trans = [0, 0, 0]

        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.2,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=colors[n % len(colors)])
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh, 'mesh')

        # Use 3 directional lights
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1]) + trans
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1]) + trans
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2]) + trans
        scene.add(light, pose=light_pose)

        # Alpha channel was not working previously need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color, rend_depth = self.renderer.render(
                scene, flags=pyrender.RenderFlags.VERTEX_NORMALS)
        color = color.astype(np.uint8)

        # msk = (rend_depth != 0).astype(np.uint8)
        # color[msk == 0] = 255
        # msk[msk == 1] = 255
        # color = np.concatenate([color, msk[..., None]], axis=2)

        #i = int(os.path.basename(mesh_path)[:-4])

        #img_path = os.path.join(data_root, ims[i]['ims'][5])

        # imf = ims[i-1400]['ims'][viewidx]
        # frmstr = os.path.basename(imf[6:-4]).split('_')
        # newfrmidx = int(frmstr[4])
        # frameidx = newfrmidx + 1400
        # img_path = '../../rddc_dataset/FranziBlue/training/img/img1400/' + str(camidx[viewidx]) + '/' + frmstr[
        #     0] + '_' + frmstr[1] + '_' + frmstr[2] + '_' + frmstr[
        #                3] + '_' + str(frameidx) + '.jpg'
        img_path = os.path.join('data/magdalena/magdalena2000-allviews/training', ims[i]['ims'][viewidx])
        print(img_path)
        #img = cv2.imread(img_path)

        img = cv2.imread(img_path)[..., [2, 1, 0]]
        
        msk_path = os.path.join('data/magdalena/magdalena2000-allviews/training', 'foregroundSegmentation/') + ims[i]['ims'][viewidx][6:]
        mask = cv2.imread(msk_path)
        print(msk_path)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        #mask = mask / 255
        img[mask == 0] = 0
        
        msk = (rend_depth != 0).astype(np.uint8)
        img_render = img.copy()
        #img[msk == 1] = img[msk == 1] * 0.5 + color[msk == 1] * 0.5
        img_render[msk == 1] = color[msk == 1]
        #img = np.concatenate([img, img_render], axis=1)
        #img = img[..., [2, 1, 0]]

        img_render = img_render[..., [2, 1, 0]]
        # path = 'data/magdalena2000-allviews/deformedcansmpl/frmmodel/ourmodel/{:d}'.format(viewidx)
        # #os.system('mkdir -p {}'.format(path))
        # os.makedirs(path, exist_ok=True)
        
        # x, y, w, h = boundbox[viewidx*300+i-700,:]
        # color = color[y:y + h, x:x + w]
        # img_render = img_render[y:y + h, x:x + w]
            
        # cv2.imwrite('{}/{:04d}.png'.format(path, i), img_render)
        
        path = f'./render_mesh/final_paper/{viewidx}'#.format(exp_name)
        print(path)
        #os.system('mkdir -p {}'.format(path))
        os.makedirs(path, exist_ok=True)
        color = color[..., [2, 1, 0]]
        # cv2.imwrite('{}/{:04d}_{}.png'.format(path, i, viewidx), img_render)
        cv2.imwrite('{}/{:04d}_{}.png'.format(path, i, viewidx), img_render)
        
        #img = img[..., [2, 1, 0]]
        #cv2.imwrite('{}/raw_{:04d}_view{:04d}.png'.format(path, i, viewidx), img)


renderer = Renderer(height=1024, width=1024)#Renderer(height=940, width=1285)

data_root = 'data/magdalena/magdalena2000-allviews'#'data/FranziBlue1400/img1400_data'
# data_root = 'data/h36m/S9/Posing'
exp_name = 'anisdf_idg'

boundbox = np.loadtxt('data/result/if_nerf/final_3/allboundbox_700.txt').astype(np.int32)
# boundbox1 = np.loadtxt('../../../neuralbody-deformation-occupancy-fixnerf/data850-1000new/result/if_nerf/female3c/allboundbox150-300_700.txt').astype(np.int)
# boundbox = boundbox + boundbox1

annots_path = os.path.join(data_root, 'annots.npy')
annots = np.load(annots_path, allow_pickle=True).item()
cameras = annots['cams']
ims = annots['ims']

Ks = np.array(cameras['K'])
Rs = np.array(cameras['R'])
Ts = np.array(cameras['T']).transpose(0, 2, 1) / 1000
Ds = np.array(cameras['D'])

camidx = [0,1,2,10,20,30,40,50,60,65,70,75,80,85,90]
viewidx = 1

# camidx = [4,2,7,8,10]#[2,2,7,7,8,8,7]
camidx = [2,7,10]#[2,2,7,7,8,8,7]
# camidx = [4,8]
frmidx = [828,839]#[700,808,832,847,709,745,985]
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# for idx in range(700,1000):#(0,8):#
for idx in range(0,3):#(0,8):#
    # frame_index = idx#frmidx[idx]
    for frame_index in range(700, 1000):
    # frame_index = 721
        viewidx = camidx[idx]
        #if frame_index%50==0:# == 745 or frame_index == 700 or frame_index == 850 or frame_index == 900 or frame_index == 950 or frame_index == 990:
        # mesh_path = os.path.join(f'paper/without_weight/deformcloth{frame_index}.obj')
        mesh_path = os.path.join(f'final_paper2/deformedverts{frame_index:04d}.obj')
        if os.path.exists(mesh_path):
            renderer.render(mesh_path, Ks[viewidx], Rs[viewidx], Ts[viewidx], frame_index, viewidx)
        