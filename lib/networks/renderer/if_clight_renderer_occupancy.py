import math
import torch
from lib.config import cfg
# from .nerf_net_utils import *
# from ... import embedder
import os
import numpy as np
#import neural_renderer as nr
import cv2
import scipy.io as scio
import trimesh
import torch.nn.functional as F

from pytorch3d.renderer.cameras import PerspectiveCameras, OrthographicCameras
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,SoftPhongShader
)
import pytorch3d.structures as struct
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops import subdivide_meshes
from pytorch3d.structures import Meshes,Pointclouds,join_meshes_as_batch
from lib.networks.gs_network_snug import Network as GNS
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from plyfile import PlyData, PlyElement

# from lib.utils.rotations import matrix_to_quaternion, rotation_6d_to_matrix
from lib.utils.loss import l1_loss, ssim
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

class Renderer:
    def __init__(self, net:GNS):
        self.net = net

        #self.meshrenderer = meshrenderer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # smpl模型三角面
        npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeform/smpltri.txt')) - 1
        self.smplfaces = torch.LongTensor(npfaces).to(self.device)
        self.smplfaces = self.smplfaces[None, :, :]
        
        # 人体模板三角面
        npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/desmpl/desmpltri.txt')) - 1
        self.desmplfaces = torch.LongTensor(npfaces).to(self.device)
        self.desmplfaces = self.desmplfaces[None, :, :]
        
        # 人体模板顶点法向量
        npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/desmpl/desmpl_vfidx.txt')) - 1        
        self.desmpl_vfidx = torch.LongTensor(npfaces).to(self.device)
        
        
        # 服装三角面
        npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/clothes_face.txt')) - 1
        self.clothfaces = torch.LongTensor(npfaces).to(self.device)
        self.clothfaces = self.clothfaces[None, :, :]
        
        # loading template deformation graph
        templateshape_path = os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/clothes_vert.txt')
        templatecloth = np.loadtxt(templateshape_path)
        templatecloth = torch.Tensor(templatecloth).to(self.device)
        self.templatecloth = templatecloth
        
        self.meshface = torch.cat([self.clothfaces, self.desmplfaces+templatecloth.shape[0]], dim=1)
        
        self.l1loss = torch.nn.L1Loss()
        
        # 固定顶点的索引
        attachidx = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/cloth/attachidx.txt')) - 1
        attachidx = torch.LongTensor(attachidx).to(self.device)
        self.attachtag = torch.zeros_like(templatecloth[:,0])
        self.attachtag[attachidx] = 1
        
        self.submesh = subdivide_meshes.SubdivideMeshes()
        
        neighbor_edges = np.load(os.path.join(cfg.train_dataset.data_root, 'edges_pair.npy'))
        self.neighbor_edges = torch.from_numpy(neighbor_edges).to(self.device)

    def prepare_sp_input(self, batch):
        # feature, coordinate, shape, batch size
        sp_input = {}
        sp_input['R'] = batch['R']
        if cfg.train_dataset.human == '0080':
            sp_input['Rh'] = batch['Rh']
        sp_input['Th'] = batch['Th']

        # used for color function
        sp_input['latent_index'] = batch['latent_index']
        sp_input['frame_index'] = batch['frame_index']

        sp_input['vert'] = batch['vert']
        sp_input['A'] = batch['A']
        sp_input['RT'] = batch['RT']
        if "pytorch_RT" in batch.keys():
            sp_input['pytorch_RT'] = batch['pytorch_RT']
        sp_input['K'] = batch['K']
        sp_input['msk'] = batch['msk']
        sp_input['fovx'] = batch['fovx']
        sp_input['fovy'] = batch['fovy']
        sp_input['world_view_transform'] = batch['world_view_transform']
        sp_input['c2w'] = batch['c2w']
        sp_input['full_proj_transform'] = batch['full_proj_transform']
        sp_input['camera_center'] = batch['camera_center']
        sp_input['cam_intrinsics'] = batch['cam_intrinsics']
        
        sp_input['smplpose'] = batch['smplpose']
        sp_input['smplshape'] = batch['smplshape']

        return sp_input 
    
    def computeinterpenetrationloss_posedsmpl(self, graphdeformedverts):

        batch_size = graphdeformedverts.size(0)
        cloth_ptsdist = torch.cdist(graphdeformedverts, self.net.deformation_network.smplgraphdeformedverts, p=2)
        cloth_ptsdistmin = torch.min(cloth_ptsdist, 2)
        cloth_nnvidx = torch.squeeze(cloth_ptsdistmin[1], -1)  # B*P
        
        ptsnum = graphdeformedverts.size(1)
        templatevtnum = self.deformedpersonsmpl.size(1)
        idx = [torch.full([ptsnum], i * templatevtnum, dtype=torch.long) for i in range(batch_size)]
        idx = torch.cat(idx).to(self.device)
        bwidx = cloth_nnvidx.view(-1) + idx
        canpersonsmpl = self.net.deformation_network.smplgraphdeformedverts.view(-1,3)
        selectpersonsmpl = canpersonsmpl[bwidx.long(), :]
        selectpersonsmpl_norm = self.net.deformation_network.deformedsmpl_vertnorm[bwidx.long(), :]
              
        normaldist = (selectpersonsmpl - graphdeformedverts.view(-1,3))*selectpersonsmpl_norm
 
        interpenetration = torch.sum(normaldist,1)
        interpenetrationloss = torch.mean(F.relu(interpenetration),0)
        
        _,trinormal = mesh_face_areas_normals(self.deformedpersonsmpl.view(-1,3), self.desmplfaces[0])
         
        deformedposedsmpl_vertnorm = trinormal[self.desmpl_vfidx,:]

        pospersonsmpl = self.deformedpersonsmpl.view(-1,3)
        selectpospersonsmpl = pospersonsmpl[bwidx.long(), :]
        selectpospersonsmpl_norm = deformedposedsmpl_vertnorm[bwidx.long(), :]
              
        normaldist1 = (selectpospersonsmpl - self.deformedcloth.view(-1,3))*selectpospersonsmpl_norm
 
        interpenetration1 = torch.sum(normaldist1,1)
        interpenetrationloss1 = torch.mean(F.relu(interpenetration1),0)
        
        return interpenetrationloss+interpenetrationloss1

    def render_image_gaussian_rasterizer(
        self, 
        verbose=False,
        bg_color = None,
        sh_deg:int=None,
        return_2d_radii = False,
        quaternions=None,
        return_opacities:bool=False,
        return_colors:bool=False,
        positions:torch.Tensor=None,
        sp_input = None,
        gs_output = None, 
        ):
        """Render an image using the Gaussian Splatting Rasterizer.

        Args:
            nerf_cameras (CamerasWrapper, optional): _description_. Defaults to None.
            camera_indices (int, optional): _description_. Defaults to 0.
            verbose (bool, optional): _description_. Defaults to False.
            bg_color (_type_, optional): _description_. Defaults to None.
            sh_deg (int, optional): _description_. Defaults to None.
            sh_rotations (torch.Tensor, optional): _description_. Defaults to None.
            compute_color_in_rasterizer (bool, optional): _description_. Defaults to False.
            compute_covariance_in_rasterizer (bool, optional): _description_. Defaults to True.
            return_2d_radii (bool, optional): _description_. Defaults to False.
            quaternions (_type_, optional): _description_. Defaults to None.
            use_same_scale_in_all_directions (bool, optional): _description_. Defaults to False.
            return_opacities (bool, optional): _description_. Defaults to False.
            return_colors (bool, optional): _description_. Defaults to False.
            positions (torch.Tensor, optional): _description_. Defaults to None.
            point_colors (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        if bg_color is None:
            bg_color = torch.Tensor([0.0, 0.0, 0.0]).to(self.device)
        
        fov_x = sp_input["fovx"][0]
        fov_y = sp_input["fovy"][0]
        tanfovx = math.tan(fov_x * 0.5)
        tanfovy = math.tan(fov_y * 0.5)
        full_proj_transform = sp_input["full_proj_transform"].squeeze(0)
        camera_center = sp_input["camera_center"][0]
        world_view_transform = sp_input["world_view_transform"].squeeze(0)
        

        raster_settings = GaussianRasterizationSettings(
            image_height=cfg.H,
            image_width=cfg.W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=sh_deg,
            campos=camera_center,
            prefiltered=False,
            debug=False
        )
    
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # shs = gs_output["gs_shs"]
        shs = None
        splat_opacities =  gs_output["gs_opacity"]
        quaternions = gs_output["quaternions"]
        scales = gs_output["gs_scales"]
        # splat_colors = None
        splat_colors = gs_output["gs_shs"]
        cov3D = None
        
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        # screenspace_points = torch.zeros_like(self._points, dtype=self._points.dtype, requires_grad=True, device=self.device) + 0
        screenspace_points = torch.zeros(positions.shape[0], 3, dtype=positions.dtype, requires_grad=True, device=self.device)
        if return_2d_radii:
            try:
                screenspace_points.retain_grad()
            except:
                print("WARNING: return_2d_radii is True, but failed to retain grad of screenspace_points!")
                pass
        means2D = screenspace_points
        
        if verbose:
            print("points", positions.shape)
            print("splat_opacities", splat_opacities.shape)
            print("quaternions", quaternions.shape)
            print("scales", scales.shape)
            print("screenspace_points", screenspace_points.shape)
        
        rendered_image, radii = rasterizer(
            means3D = positions,
            means2D = means2D,
            shs = shs,
            colors_precomp = splat_colors,
            opacities = splat_opacities,
            scales = scales,
            rotations = quaternions,
            cov3D_precomp = cov3D)
        
        if not(return_2d_radii or return_opacities or return_colors):
            return rendered_image.transpose(0, 1).transpose(1, 2)
        
        else:
            outputs = {
                "image": rendered_image.transpose(0, 1).transpose(1, 2),
                "radii": radii,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
            }
            if return_opacities:
                outputs["opacities"] = splat_opacities
            if return_colors:
                outputs["colors"] = splat_colors
        
            return outputs

    def face_normal(self,vertices,faces):
        v = vertices
        f = faces
        vertice_num = v.shape[-1]
        indices_v_num = f.shape[-1]
        # 是否为批量 
        is_batch = False
        if len(v.shape) == (len(f.shape) + 1):
            f = torch.tile(f, [v.shape[0],1, 1])
            f = f.reshape(f.shape[0],-1,1)
            f = f.repeat([1,1,vertice_num]).long()
            is_batch = True
        else:
            f = f.reshape(-1,1)
            f = f.repeat(1,vertice_num).long()
        # 获取mesh的三角形的顶点坐标[batch_size,num_faces,3,3]
        triangles = torch.gather(vertices, -2, f)
        if is_batch:
            triangles = triangles.reshape(triangles.shape[0],-1,indices_v_num,vertice_num)
        else:
            triangles = triangles.reshape(-1,indices_v_num,vertice_num)
        
        # 计算法向量
        v0 = triangles[...,0,:]
        v1 = triangles[...,1,:]
        v2 = triangles[...,2,:]
        e1 = (v0 - v1)
        e2 = (v2 - v1)
        face_normals = torch.cross(e2, e1,dim=-1)
        # 单位化
        # if self.normalize:
        normals = face_normals.norm(dim=-1,keepdim=True)
        face_normals = face_normals / normals
        
        return face_normals
    
    def render_deformation(self, batch, epoch):

        # encode neural body
        sp_input = self.prepare_sp_input(batch)

        #predicting detailed model
        self.deformation_affine, self.deformation_transl = self.net.deformation_network.predicting_deformation(sp_input)
        self.deformation_affine_smpl, self.deformation_transl_smpl = self.net.deformation_network.predicting_deformation_smpl(sp_input)

        # cloth_displace = self.net.vert_displace_network(sp_input)
        self.net.deformation_network.update_embeddedgraph(self.templatecloth.reshape(-1,3))
        # NOTE: smoothloss
        self.smoothloss = self.net.deformation_network.deformationsmoothloss()
        self.smoothloss_smpl = self.net.deformation_network.deformationsmoothloss_smpl()
        
        self.deformedpersonsmpl, smplgraphdeformedverts = self.net.deformation_network.deformingsmpl_graphdeform_LBS(sp_input)
        self.deformedcloth, graphdeformedverts = self.net.deformation_network.deformingcloth_graphdeform_LBS(sp_input) 


        self.graphdeform_loss = self.l1loss(graphdeformedverts, batch["snug_cloth"])
        f_n = self.face_normal(vertices=graphdeformedverts.squeeze(), faces=self.clothfaces[0])
        f_n_snug = self.face_normal(vertices=batch["snug_cloth"].squeeze(), faces=self.clothfaces[0])
        # ss = torch.sum(torch.mul(f_n, self.f_normal),dim=-1)
        self.fn_loss = torch.mean(1 - torch.sum(torch.mul(f_n, f_n_snug),dim=-1))
        
        # NOTE: 计算碰撞损失
        self.interploss_graphdeform_disp = self.computeinterpenetrationloss_posedsmpl(graphdeformedverts)
        
        # DEBUG: 查看变形是否准确
        frame_index = sp_input['frame_index'].item()
        # NOTE: 保存可视化结果
        save_dir = "tmp"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if frame_index == 700:
            mesh_test = trimesh.Trimesh(vertices=self.deformedcloth[0].detach().cpu().numpy(), faces=self.clothfaces[0].detach().cpu().numpy())
            npfaces = np.loadtxt(os.path.join(cfg.train_dataset.data_root, 'templatedeformT/desmpl/desmpltri.txt')) - 1
            deformedpersonsmplmesh = trimesh.Trimesh(vertices=self.deformedpersonsmpl[0].detach().cpu().numpy(),
                                                faces=npfaces)
            mesh_con = trimesh.util.concatenate([mesh_test, deformedpersonsmplmesh])
            mesh_con.export(f"{save_dir}/deformedverts{frame_index:04d}.obj")

        
        

        if cfg.train_dataset.human == "0080":
            R, T = torch.split(batch['pytorch_RT'], [3, 1], dim=-1)
            R = R.transpose(-1, -2)
            T = T.transpose(-1, -2)
        else:
            R, T = torch.split(batch['RT'], [3, 1], dim=-1)
            R = R.transpose(-1, -2)
            T = T.transpose(-1, -2)
  
        cameras = PerspectiveCameras(device='cuda',
                                     K=batch['pytorch3d_K'].float(),
                                     R=R.float(),
                                     T=T[0].float())

        # NOTE: 计算变形图预测的服装与mask之间的误差
        meshvert_def = torch.cat([self.deformedcloth, self.deformedpersonsmpl], dim=1)#self.deformedcloth
        
        self.net.pcRender_def.rasterizer.cameras=cameras
        features=[torch.ones(meshvert_def.shape[1],1,device=self.device) for _ in range(1)]
        predicted_silhouette_def,frags=self.net.pcRender_def(Pointclouds(points=meshvert_def,features=features))
        predicted_silhouette_def = predicted_silhouette_def.squeeze(-1)
        self.IoUloss_def = ((predicted_silhouette_def - batch['msk']) ** 2).mean()
        
        # NOTE: 计算渲染结果分别属于人体和服装的部分
        mesh = struct.Meshes(verts=meshvert_def, faces=self.meshface) 
        
        fragments = self.net.rasterizer(mesh, cameras=cameras)
        depth = fragments.zbuf
        face_idx_map = fragments.pix_to_face[..., 0]
        self.rendermask = torch.where(depth > 0, torch.ones_like(depth),torch.zeros_like(depth))
        self.rendermask = self.rendermask.squeeze(-1).float()  
        
        self.silhouette = face_idx_map>=self.clothfaces.shape[1]# body mask

        self.silhouette = ~self.silhouette
       
        self.silhouette = self.silhouette.float()
        
        self.silhouette_smpl = (face_idx_map<self.clothfaces.shape[1]) & (face_idx_map>=0)# body mask
        self.silhouette_smpl = ~self.silhouette_smpl
        self.silhouette_smpl = self.silhouette_smpl.float()
        
        # 预测3DGS的属性
        ## 提取特征
        tri_feats_smpl = self.net.position_enc_smpl(self.net.sugar_model_smpl.points, sp_input['latent_index'].to(torch.int64)[0])
        tri_feats_cloth = self.net.position_enc_cloth(self.net.sugar_model_cloth.points, sp_input['latent_index'].to(torch.int64)[0])
        ## 3DGS颜色和几何属性
        appearance_out_smpl = self.net.appearance_dec_smpl(tri_feats_smpl)
        appearance_out_cloth = self.net.appearance_dec_cloth(tri_feats_cloth)
        geometry_out_smpl = self.net.geometry_dec_smpl(tri_feats_smpl)
        geometry_out_cloth = self.net.geometry_dec_cloth(tri_feats_cloth)
        
        # 预测透明度和球谐函数参数(SMPL和CLOTH)
        gs_opacity_smpl = appearance_out_smpl['opacity']
        # gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 16, 3)
        gs_shs_smpl = appearance_out_smpl['shs'].reshape(-1, 3)
        
        gs_opacity_cloth = appearance_out_cloth['opacity']
        # gs_shs_cloth = appearance_out_cloth['shs'].reshape(-1, 16, 3)
        gs_shs_cloth = appearance_out_cloth['shs'].reshape(-1, 3)
        
        
        # 预测旋转和尺度(SMPL和CLOTH)
        rotations_smpl = geometry_out_smpl['rotations']
        gs_scales_smpl = geometry_out_smpl['scales']
        
        rotations_cloth = geometry_out_cloth['rotations']
        gs_scales_cloth = geometry_out_cloth['scales']
        
        points_smpl = self.net.sugar_model_smpl.get_edited_points(self.deformedpersonsmpl.squeeze(0))
        quaternions_smpl, gs_scales_smpl = self.net.sugar_model_smpl.get_edited_quaternions_and_scales_with_points(points=self.deformedpersonsmpl, 
                                                                                                                    quat=rotations_smpl, 
                                                                                                                    sca=gs_scales_smpl)
        # 细化模型
        meshes = Meshes(
                verts=[self.deformedcloth.squeeze(0).to(self.device)],   
                faces=[self.clothfaces.squeeze(0)],
        )

        
        meshes = self.submesh(meshes)
        points = meshes.verts_list()[0]
        points_cloth = self.net.sugar_model_cloth.get_edited_points(points.squeeze(0))
        quaternions_cloth, gs_scales_cloth = self.net.sugar_model_cloth.get_edited_quaternions_and_scales_with_points(points=points, 
                                                                                                                    quat=rotations_cloth, 
                                                                                                                    sca=gs_scales_cloth)

        # 渲染3DGS
        ## Cloth
        gs_output_cloth = {
            "points": points_cloth,
            "quaternions": quaternions_cloth,
            "gs_scales": gs_scales_cloth,
            "gs_opacity": gs_opacity_cloth,
            "gs_shs": gs_shs_cloth
        }
        render_cloth_output = self.render_image_gaussian_rasterizer(
            sh_deg=4,
            quaternions=None,
            return_2d_radii=True,
            return_colors=True,
            return_opacities=True,
            positions=points_cloth,
            sp_input=sp_input,
            gs_output=gs_output_cloth
        )
        
        ## SMPL
        gs_output_smpl = {
            "points": points_smpl,
            "quaternions": quaternions_smpl,
            "gs_scales": gs_scales_smpl,
            "gs_opacity": gs_opacity_smpl,
            "gs_shs": gs_shs_smpl
        }
        render_smpl_output = self.render_image_gaussian_rasterizer(
            sh_deg=4,
            quaternions=None,
            return_2d_radii=True,
            return_colors=True,
            return_opacities=True,
            positions=points_smpl,
            sp_input=sp_input,
            gs_output=gs_output_smpl
        )
        
        # NOTE: 使用 1-blendmask组合渲染
        rendered_img = render_smpl_output['image'] * (self.silhouette_smpl[0].unsqueeze(-1)) + render_cloth_output['image'] * self.silhouette[0].unsqueeze(-1)
        ret = {
            'image': rendered_img,
            'cloth_scene': render_cloth_output,
            'smpl_scene': render_smpl_output,
            'gs_output_smpl': gs_output_smpl,
            'gs_output_cloth': gs_output_cloth
        }
        # 渲染结果可视化
        if frame_index == 700:
            render_img = ret['image'].cpu().detach().numpy() * 255
            render_img = cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{save_dir}/gs_render.jpg", render_img)
        
        return ret