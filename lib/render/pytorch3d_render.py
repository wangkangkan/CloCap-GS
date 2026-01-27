import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    RasterizationSettings,
    BlendParams,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    TexturesVertex
)

class Py3dRender():
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        direction = torch.tensor([[0.0, -1.0, 1.0]], device=self.device)  # 太阳光方向 (从上方和前方照射)
        lights = DirectionalLights(
            device=self.device,
            direction=direction,
            ambient_color=((0.3, 0.3, 0.3),),  # 环境光
            diffuse_color=((2.0, 2.0, 2.0),),  # 漫反射
            # specular_color=((0.5, 0.5, 0.5),)  # 镜面反射
        )

        # 渲染设置
        raster_settings = RasterizationSettings(
            image_size=1024,  # 输出图像大小
            blur_radius=0.0,  # 无模糊
            faces_per_pixel=1,  # 每像素显示的面数
            # background_color=(0.0, 0.0, 0.0)
        )
        blend_params=BlendParams(background_color=(0.0,0.0,0.0))
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = PerspectiveCameras(device=self.device, R=R, T=T)
        # 设置渲染器
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                blend_params=blend_params,
                # lights=lights
            )
        )
        obj_filename = "/home/gugudada/Downloads/untitled.obj"  # 替换为你的 OBJ 文件路径
        mesh = load_objs_as_meshes([obj_filename], device=self.device)
        self.textures = mesh.textures
        self.face = mesh.faces_list()[0]
        
        smpl_obj_filename = "/home/gugudada/deform.obj"
        smpl_mesh = load_objs_as_meshes([smpl_obj_filename], device=self.device)
        self.smpltextures = smpl_mesh.textures
        self.smplface = smpl_mesh.faces_list()[0]
        
    def renderImage(self, batch):
        R, T = torch.split(batch['RT'], [3, 1], dim=-1)
        R = R.transpose(-1, -2)
        T = T.transpose(-1, -2)
  
        cameras = PerspectiveCameras(device=self.device,
                                     K=batch['pytorch3d_K'].float(),
                                     R=R.float(),
                                     T=T[0].float())
        self.renderer.rasterizer.cameras=cameras
        
        meshes = Meshes(
                    verts=[batch["codim_mesh"].squeeze(0).to(self.device).float()],   
                    faces=[self.face],
                    textures=self.textures
            )
        # meshes.textures = self.textures
        smpl_meshes = Meshes(
            verts=[batch["deform_mesh"].squeeze(0).to(self.device).float()],   
            faces=[self.smplface],
            textures=self.smpltextures
        )
        # smpl_meshes.textures = self.smpltextures
        
        joint_mesh = join_meshes_as_scene(meshes=[meshes, smpl_meshes])
        images = self.renderer(joint_mesh)
        
        mask = (np.where(images[0, ..., :3].cpu().numpy() > 0.0, 1.0, 0.0)).astype(np.uint8) 
        
        return images, mask
        
        
# # 设备设置
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 加载网格 (包含 OBJ 文件和材质贴图)
# obj_filename = "/home/gugudada/Downloads/untitled.obj"  # 替换为你的 OBJ 文件路径
# mesh = load_objs_as_meshes([obj_filename], device=device)
# obj_filename = "sim_input/mag_700_300_cloth/dress_reordered_11.obj"
# mesh2 = load_objs_as_meshes([obj_filename], device=device)
# mesh2.textures = mesh.textures

# # 定义摄像机视角
# R, T = look_at_view_transform(2.7, 0, 0)  # 距离，仰角，方位角
# cameras = PerspectiveCameras(device=device, R=R, T=T)

# # 定义光源
# # lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])
# direction = torch.tensor([[0.0, -1.0, 1.0]], device=device)  # 太阳光方向 (从上方和前方照射)
# lights = DirectionalLights(
#     device=device,
#     direction=direction,
#     ambient_color=((0.3, 0.3, 0.3),),  # 环境光
#     diffuse_color=((2.0, 2.0, 2.0),),  # 漫反射
#     # specular_color=((0.5, 0.5, 0.5),)  # 镜面反射
# )

# # 渲染设置
# raster_settings = RasterizationSettings(
#     image_size=1024,  # 输出图像大小
#     blur_radius=0.0,  # 无模糊
#     faces_per_pixel=1  # 每像素显示的面数
# )

# # 设置渲染器
# renderer = MeshRenderer(
#     rasterizer=MeshRasterizer(
#         cameras=cameras,
#         raster_settings=raster_settings
#     ),
#     shader=SoftPhongShader(
#         device=device,
#         cameras=cameras,
#         lights=lights
#     )
# )

# # 渲染图像
# images = renderer(mesh2)

# # 保存渲染结果
# import matplotlib.pyplot as plt
# plt.imshow(images[0, ..., :3].cpu().numpy())
# plt.axis("off")
# plt.show()
