from copy import deepcopy
import numpy as np
import pickle
import random

import torch
from torch import nn

from mvn.utils import op, multiview, img, misc, volumetric

from mvn.models import pose_resnet
from mvn.models.v2v import V2VModel

class VolPose2dSpine2dAndFeatures(nn.Module):
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

    def forward(self, images):
        heatmaps, features = self.backbone(images) 
        heatmapSpine = heatmaps[:, 6, :, :].unsqueeze(1);
        jointSpine2d = op.integrate_tensor_2d(heatmapSpine * self.heatmap_multiplier)
        features = self.process_features(features)
        return jointSpine2d, features
