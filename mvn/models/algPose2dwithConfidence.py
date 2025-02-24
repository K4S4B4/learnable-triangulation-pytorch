from copy import deepcopy
import numpy as np
import pickle
import random

from scipy.optimize import least_squares

import torch
from torch import nn

from mvn.utils import op, multiview, img, misc, volumetric

from mvn.models import pose_resnet
from mvn.models.v2v import V2VModel


class AlgebraicTriangulationNet2(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.use_confidences = config.model.use_confidences

        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False

        if self.use_confidences:
            config.model.backbone.alg_confidences = True

        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)

        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier


    def forward(self, images):
        device = images.device
        heatmaps, _, alg_confidences = self.backbone(images)
        keypoints_2d = op.integrate_tensor_2d(heatmaps * self.heatmap_multiplier, self.heatmap_softmax)
        return keypoints_2d, alg_confidences
