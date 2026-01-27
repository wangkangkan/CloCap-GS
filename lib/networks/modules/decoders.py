#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.networks.modules.triplane import TriPlane
from lib.config import cfg

from .activation import SineActivation


act_fn_dict = {
    'softplus': torch.nn.Softplus(),
    'relu': torch.nn.ReLU(),
    'sine': SineActivation(omega_0=30),
    'gelu': torch.nn.GELU(),
    'tanh': torch.nn.Tanh(),
}

class PositionEncoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=64, num_train_frame = 1, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
            
        self.position_enc = TriPlane(cfg.triplane.n_features,
                                        resX=cfg.triplane.triplane_res,
                                        resY=cfg.triplane.triplane_res,
                                        resZ=cfg.triplane.triplane_res)
        
        # self.final_layer = nn.Linear(hidden_dim, n_features)
        self.latentdeform = nn.Embedding(num_train_frame, n_features)
        
    def forward(self, x, frame_index):

        num_v = x.shape[0]
        # position_code = self.final_layer(self.net(x))
        position_code = self.position_enc(x)
        latent = self.latentdeform(frame_index)
        latent = latent[None].repeat(num_v, 1)
        
        x = torch.concatenate([position_code, latent], dim=-1)
        return x
 

class AppearanceDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=64, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
            
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.opacity = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        # self.shs = nn.Linear(hidden_dim, 16*3)
        self.shs = nn.Linear(hidden_dim, 3)
        
    def forward(self, x):

        x = self.net(x)
        shs = self.shs(x)
        opacity = self.opacity(x)
        return {'shs': shs, 'opacity': opacity}
    

class DeformationDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=128, weight_norm=True, act='gelu', disable_posedirs=False):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.sine = SineActivation(omega_0=30)
        self.disable_posedirs = disable_posedirs
        
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.skinning_linear = nn.Linear(hidden_dim, hidden_dim)
        self.skinning = nn.Linear(hidden_dim, 24)
        
        if weight_norm:
            self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
            
        # initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)
        if not disable_posedirs:
            self.blendshapes = nn.Linear(hidden_dim, 3 * 207)
            torch.nn.init.constant_(self.blendshapes.bias, 0.0)
            torch.nn.init.constant_(self.blendshapes.weight, 0.0)
        
    def forward(self, x):
        x = self.net(x)
        if not self.disable_posedirs:
            posedirs = self.blendshapes(x)
            posedirs = posedirs.reshape(207, -1)
            
        lbs_weights = self.skinning(F.gelu(self.skinning_linear(x)))
        lbs_weights = F.gelu(lbs_weights)
        
        return {
            'lbs_weights': lbs_weights,
            'posedirs': posedirs if not self.disable_posedirs else None,
        }
    

class GeometryDecoder(torch.nn.Module):
    def __init__(self, n_features, use_surface=False, hidden_dim=128, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        # self.xyz = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 3))
        self.rotations = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 2))
        # self.rotations = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 6))
        self.scales = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 2 if use_surface else 3)) # 此处预测平面的尺度信息
        
    def forward(self, x):
        # xyz = self.xyz(x)
        rotations = self.rotations(x)
        scales = F.gelu(self.scales(x))
        # scales = F.relu(self.scales(x))
        # scales = self.scales(x)
        return {
            # 'xyz': xyz,
            'rotations': rotations,
            'scales': scales,
        }