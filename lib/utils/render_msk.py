import pyrender
import numpy as np
import trimesh
import cv2

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code borrowed from https://github.com/nkolot/SPIN
    """
    def __init__(self, focal_length=5000, img_res=512):
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_res,
            viewport_height=img_res,
            point_size=1.0)
        self.renderer.viewport_height = img_res
        self.renderer.viewport_width = img_res
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]

    def __call__(self, vertices, faces, 
                 background_color=(255, 255, 255), 
                 baseColorFactor=[0.658, 0.214, 0.0114, 1.0],
                 camera_pose = None,
                 K=None):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0,
            alphaMode='OPAQUE',
            # baseColorFactor=(0.4, 0.4, 0.4, 1.0)
            baseColorFactor=baseColorFactor
            )

        mesh = trimesh.Trimesh(vertices, faces)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        
        # camera_pose = np.eye(4)
        camera_pose = camera_pose
        # camera_pose[:3, 3] = camera_translation
        # camera = pyrender.IntrinsicsCamera(
        #     fx=self.focal_length, fy=self.focal_length,
        #     cx=self.camera_center[0], cy=self.camera_center[1])
        camera = pyrender.camera.IntrinsicsCamera(fx=K[0, 0],
                                                  fy=K[1, 1],
                                                  cx=K[0, 2],
                                                  cy=K[1, 2])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        # color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        full_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        valid_mask = (full_depth > 0)[:,:,None]
        return valid_mask
    

if __name__ == "__main__":
    
    annots = np.load("data/mag/annots.npy", allow_pickle=True).item()
    cams = annots['cams']
    
    mesh = trimesh.load("sim_output/codim_ipc_sim/1_2_mag_700_10_cloth_1.000000_1.000000_mag_700_10_smplx_320_10_2/shell10.obj")
    v = mesh.vertices
    f = mesh.faces
    
    PYTORCH3D_R = np.array(cams['R'][0])
    PYTORCH_T = np.array(cams['T'][0]) / 1000.
    PYTORCH_RT = np.concatenate([PYTORCH3D_R, PYTORCH_T], axis=1).astype(np.float32)
    c2w = np.linalg.inv(np.concatenate([PYTORCH_RT, [[0.0, 0.0, 0.0, 1.0]]], axis=0))
    opengl_R =  np.stack([c2w[:3, 0], -c2w[:3, 1], -c2w[:3, 2]], -1)
    opengl_T = c2w[:3, 3:]
    opengl_RT = np.concatenate([opengl_R, opengl_T], axis=1).astype(np.float32)
    opengl_RT = np.concatenate([opengl_RT, [[0.0, 0.0, 0.0, 1.0]]], axis=0).astype(np.float32)
    c2w = opengl_RT
    
    K = np.array(cams['K'][0])
    
    # vertices_color = np.array([[171, 162, 113, 255]], dtype=np.uint8).repeat(len(v), axis=0)
    # # print(mesh.visual.vertex_colors)
    # mesh.visual.vertex_colors = vertices_color
    # mesh.show()
    
    render = Renderer(img_res=1024)
    img = render(v, f, background_color=(0, 0, 0), K=K, camera_pose=c2w)
    cv2.imwrite("res_1024.jpg", img.astype(np.uint8)*255)
    print(img)