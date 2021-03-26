from copy import deepcopy
import numpy as np
import pickle
import random

import torch
from torch import nn

from mvn.utils import op, multiview, img, misc, volumetric

from mvn.models import pose_resnet
from mvn.models.v2v import V2VModel

class VolumetricTriangulationNet2(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints
        self.volume_aggregation_method = config.model.volume_aggregation_method

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size

        self.cuboid_side = config.model.cuboid_side

        self.kind = config.model.kind

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # modules
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False

        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)

        for p in self.backbone.final_layer.parameters():
            p.requires_grad = False

        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )

        self.volume_net = V2VModel(32, self.num_joints)

        self.inputImageShape = [384, 384]
        self.outputHeatmapShape = [96, 96]

    #def calc2dPoint(self, images, proj_matricies):

    def forward(self, images, proj_matricies):
    
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        heatmaps, features = self.backbone(images) 

        keypoints_2d = op.integrate_tensor_2d(heatmaps * self.heatmap_multiplier, self.heatmap_softmax)

        # reshape back
        keypoints_2d = keypoints_2d.view(batch_size, n_views, *keypoints_2d.shape[1:])

        # triangulate
        keypoints_3d_Alg = multiview.triangulate_batch_of_points(
            proj_matricies, keypoints_2d,
            #confidences_batch=alg_confidences
            )
        
        ## triangulate
        #try:
        #    keypoints_3d_Alg = multiview.triangulate_batch_of_points(
        #        proj_matricies, keypoints_2d,
        #        #confidences_batch=alg_confidences
        #    )
        #except RuntimeError as e:
        #    print("Error: ", e)
        #    print("confidences =", confidences_batch_pred)
        #    print("proj_matricies = ", proj_matricies)
        #    print("keypoints_2d_batch_pred =", keypoints_2d_batch_pred)
        #    exit()
        # ALG ################################

        # build coord volumes

        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device) # Bx64x64x64x3
        for batch_i in range(batch_size):

            keypoints_3d = keypoints_3d_Alg[0].to('cpu').detach().numpy().copy()
            base_point = keypoints_3d[6, :3]

            # build cuboid
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point - sides / 2

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volumes[batch_i] = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

        # process features before unprojecting
        #features = features.view(batch_size, n_views, *features.shape[1:])
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.view(batch_size, n_views, *features.shape[1:])

        # lift to volume
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method)

        # integral 3d
        volumes = self.volume_net(volumes)
        vol_keypoints_3d = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return vol_keypoints_3d, features, volumes, coord_volumes
