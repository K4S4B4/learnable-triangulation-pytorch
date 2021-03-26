from copy import deepcopy
import numpy as np
import pickle
import random

import torch
from torch import nn

from mvn.utils import op, multiview, img, misc, volumetric

from mvn.models import pose_resnet
from mvn.models.v2v import V2VModel

class AlgPose3dPreTriangulation(nn.Module):
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


    #def calc2dPoint(self, images, proj_matricies):

    # images  [n*3*H*W]
    # prjMats [n*3*4]
    def forward(self, images, prjMats):

        images = images.permute(0,3,1,2)

        ######################################
        heatmaps, features, confidence = self.backbone(images) # n*17
        joints2d = op.integrate_tensor_2d(heatmaps * self.heatmap_multiplier) # n*17*2

        n_views = images.shape[0]

        confidenceEx = confidence.unsqueeze(2).unsqueeze(3).expand(n_views, 17, 2, 4) # [n*17] -> [n*17*1*1] -> [n*17*2*4]
        joints2d = joints2d.unsqueeze_(3).expand(n_views, 17, 2, 4) # [n*17*2] -> [n*17*2*1] -> [n*17*2*4]

        prjMats1 = prjMats[:, 2:3].expand(n_views, 2, 4) # [n*1*4] -> [n*2*4]
        prjMats2 = prjMats[:, :2] # [n*2*4]

        prjMats1 = prjMats1.unsqueeze_(1).expand(n_views, 17, 2, 4) # [n*2*4] -> [n*1*2*4] -> [n*17*2*4]
        prjMats2 = prjMats2.unsqueeze_(1).expand(n_views, 17, 2, 4) # [n*2*4] -> [n*1*2*4] -> [n*17*2*4]

        A = confidenceEx * (joints2d * prjMats1 - prjMats2) # [n*17*2*4]
        ######################################

        return A, confidence;



        #confidence = confidence.permute(1,0).unsqueeze_(2).unsqueeze_(3) # [n*17] -> [17*n*1*1]
        #joints2d = joints2d.permute(1,0,2).unsqueeze_(3) # [n*17*2] -> [17*n*2*1]

        #n_views = images.shape[0]
        #prjMats1 = prjMats[:, 2:3].expand(n_views, 2, 4) # [n*2*4]
        #prjMats2 = prjMats[:, :2] # [n*2*4]

        #prjMats1 = prjMats1.unsqueeze_(0).expand(17, n_views, 2, 4)
        #prjMats2 = prjMats2.unsqueeze_(0).expand(17, n_views, 2, 4)

        #A = confidence * (joints2d * prjMats1 - prjMats2) # [17*n*2*4]
        #return A, confidence;
