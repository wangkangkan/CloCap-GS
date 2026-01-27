import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import if_clight_renderer_occupancy
from lib.utils.pytorch_ssim import ssim_m
from lib.config import cfg
from lib.utils.loss import ssim


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = if_clight_renderer_occupancy.Renderer(self.net)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def l1_loss(self, network_output, gt, mask=None):
        if mask is not None:
            return torch.abs((network_output - gt)).sum() / mask.sum()
        else:
            return torch.abs((network_output - gt)).mean()
    
    def img2mse(self, x, y, M=None):
        if M == None:
            return torch.mean((x - y) ** 2)
        else:
            return torch.sum((x - y) ** 2 * M) / (torch.sum(M) + 1e-8) / x.shape[-1]     
    def forward(self, batch, epoch):
            
        ret = self.renderer.render_deformation(batch, epoch)
        
        scalar_stats = {}
        loss = 0

        msk = batch['msk'].float()
        img_loss = self.l1_loss(ret['image'].unsqueeze(0), batch['img'] * msk.unsqueeze(-1))
        scalar_stats.update({'img_loss': img_loss})
        loss += cfg.criterions.img_loss.weight*img_loss
        
        loss_ssim = 1.0 - ssim_m(ret['image'].unsqueeze(0), batch['img'] * msk.unsqueeze(-1))
        loss_ssim = loss_ssim * (msk.sum() / (ret['image'].shape[-1] * ret['image'].shape[-2]))
        scalar_stats.update({'loss_ssim': loss_ssim})
        loss += cfg.criterions.loss_ssim.weight *loss_ssim

        
        # DEBUG: mask损失
        scalar_stats.update({'IoUloss': self.renderer.IoUloss_def})
        loss += cfg.criterions.IoUloss.weight *self.renderer.IoUloss_def#10
        
        loss += cfg.criterions.smoothloss.weight * self.renderer.smoothloss#5.0
        scalar_stats.update({'smoothloss': self.renderer.smoothloss})
        loss += cfg.criterions.smoothloss.weight * self.renderer.smoothloss_smpl#5.0
        scalar_stats.update({'smoothloss_smpl': self.renderer.smoothloss_smpl})
        # DEBUG: END
        
        # NOTE: disp碰撞损失
        loss += cfg.criterions.interploss.weight * self.renderer.interploss_graphdeform_disp
        scalar_stats.update({'interploss_graphdeform_disp': self.renderer.interploss_graphdeform_disp})
        
        scalar_stats.update({'fn_loss': self.renderer.fn_loss})
        loss += cfg.criterions.fn_loss.weight *self.renderer.fn_loss
        
        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
