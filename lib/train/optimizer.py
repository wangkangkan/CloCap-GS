import torch
from lib.utils.optimizer.radam import RAdam


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam,
    'sgd': torch.optim.SGD
}


def make_optimizer(cfg, net, lr=None, weight_decay=None):
    params = []
    lr = cfg.train.lr if lr is None else lr
    weight_decay = cfg.train.weight_decay if weight_decay is None else weight_decay

    # for key, value in net.named_parameters():
    #     if not value.requires_grad:
    #         continue
    #     params += [{"params": [value], "lr": lr, "weight_decay": weight_decay, "name": key}]
    params = [
            {'params': [net.sugar_model_smpl._xyz_weight], 'lr': cfg.lr.xyz, "name": "xyz_smpl"},
            {'params': [net.sugar_model_cloth._xyz_weight], 'lr': cfg.lr.xyz, "name": "xyz_cloth"},
            # {'params': net.sugar_model_smpl.parameters(), 'lr': cfg.lr.xyz, "name": "xyz_smpl"},
            # {'params': net.sugar_model_cloth.parameters(), 'lr': cfg.lr.xyz, "name": "xyz_cloth"},
            {'params': net.position_enc_smpl.parameters(), 'lr': cfg.lr.vembed_smpl, 'name': 'v_embed_smpl'},
            {'params': net.position_enc_cloth.parameters(), 'lr': cfg.lr.vembed_cloth, 'name': 'v_embed_cloth'},
            {'params': net.appearance_dec_smpl.parameters(), 'lr': cfg.lr.geometry_smpl, 'name': 'geometry_dec_smpl'},
            {'params': net.appearance_dec_cloth.parameters(), 'lr': cfg.lr.geometry_cloth, 'name': 'geometry_dec_cloth'},
            {'params': net.geometry_dec_smpl.parameters(), 'lr': cfg.lr.appearance_smpl, 'name': 'appearance_dec_smpl'},
            {'params': net.geometry_dec_cloth.parameters(), 'lr': cfg.lr.appearance_cloth, 'name': 'appearance_dec_cloth'},
            {'params': net.deformation_network.parameters(), 'lr': cfg.lr.deform, 'name': 'deform'},
            # {'params': net.vert_displace_network.parameters(), 'lr': cfg.lr.deform, 'name': 'deform'},
            # {'params': net.cloth_simulation.parameters(), 'lr': cfg.lr.sim, 'name': 'sim'},
            # {'params': net.tempclothpara.parameters(), 'lr': cfg.lr.param, 'name': 'param'},
        ]
    if 'adam' in cfg.train.optim:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay)
    else:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)

    return optimizer
