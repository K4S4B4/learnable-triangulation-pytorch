import os
import shutil
import argparse
import time
import json
from datetime import datetime
from collections import defaultdict
from itertools import islice
import pickle
import copy

import numpy as np
import cv2

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.datasets import human36m
from mvn.utils.multiview import Camera

from mvn.utils.img import resize_image, crop_image, normalize_image, image_batch_to_torch

#from detectron2_util import Detectron2util
from CameraCalibration import ArucoCalibrator

import torch.onnx
import onnx
import onnxruntime

#from torch2trt import torch2trt

from mvn.models.triangulation2 import VolumetricTriangulationNet2
from mvn.models.algPose2d import BaselinePose2d
from mvn.models.algPose2dwithConfidence import AlgebraicTriangulationNet2
from mvn.models.volPose2dFeatureOnly import VolPose2dFeatureOnly
#from mvn.models.volPose2dSpine2dAndFeatures import VolPose2dSpine2dAndFeatures
from mvn.models.algPose3dTriangulation import AlgPose3dTriangulation
from mvn.models.volPose3d import VolPose3d
from mvn.models.algPose3dPreTriangulation import AlgPose3dPreTriangulation
from mvn.models.algPose2dwithConfidence_permuteInput import AlgebraicTriangulationNetPermute
from mvn.models.algHeatmap import AlgebraicHeatmap
from mvn.models.algPose2dwithConfidence_preprocessInput import AlgebraicTriangulationNetPreprocess

def createCoordinateVolume(batch_size = 2, volume_size = 64, device='cpu', base_point=[0,0,0], cuboid_side=2500):

    coord_volumes = torch.zeros(batch_size, volume_size, volume_size, volume_size, 3, device=device) # Bx64x64x64x3
    for batch_i in range(batch_size):

        #keypoints_3d = keypoints_3d_Alg[0].to('cpu').detach().numpy().copy()
        #base_point = keypoints_3d[6, :3]

        # build cuboid
        sides = np.array([cuboid_side, cuboid_side, cuboid_side])
        position = base_point - sides / 2

        # build coord volume
        xxx, yyy, zzz = torch.meshgrid(torch.arange(volume_size, device=device), torch.arange(volume_size, device=device), torch.arange(volume_size, device=device))
        grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
        grid = grid.reshape((-1, 3))

        grid_coord = torch.zeros_like(grid)
        grid_coord[:, 0] = position[0] + (sides[0] / (volume_size - 1)) * grid[:, 0]
        grid_coord[:, 1] = position[1] + (sides[1] / (volume_size - 1)) * grid[:, 1]
        grid_coord[:, 2] = position[2] + (sides[2] / (volume_size - 1)) * grid[:, 2]

        coord_volumes[batch_i] = grid_coord.reshape(volume_size, volume_size, volume_size, 3)

    return coord_volumes;

def main():
    ###########################################
    ######### Onnx 測定 ###########
    ###########################################    

    #onnx_file_name = 'resource/LTHP/baseline_pose2d_withConf_2x384x384xBGRxByte.onnx'
    onnx_file_name = 'resource/HRNet/hrnet_2x256x256xBGRxByte_pose2dWithConf.onnx'

    ##############################################################################
    batch_size = 2
    height = 256
    width = 256
    #height = 384
    #width = 384
    ##############################################################################

    ###########################################################
    image = cv2.imread('resource/testdata/IMG_20210208_135527.jpg')
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    img_in = np.expand_dims(resized, axis=0).astype(np.uint8)
    ###########################################################

    ###########################################################
    sess_options = onnxruntime.SessionOptions()
    # Set graph optimization level
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # To enable model serialization after graph optimization set this
    #sess_options.optimized_model_filepath = "baseline_pose2d_withConfidence2_fixedBatchSize1.optimized.onnx"
    sess_options.enable_profiling = True
    ###########################################################

    ###########################################################
    providers = ['TensorrtExecutionProvider']
    #providers = ['CUDAExecutionProvider']
    #providers = ['CPUExecutionProvider']
    ###########################################################

    ort_session = onnxruntime.InferenceSession(onnx_file_name, sess_options, providers)
    input_name = ort_session.get_inputs()[0].name

    ###########################################################
    #inputTensor = tensorImage.expand(batch_size, height, width, 3)
    #ort_inputs = {input_name: inputTensor}
    ###########################################################
    inputTensor = torch.rand(batch_size, height, width, 3).byte()
    ort_inputs = {ort_session.get_inputs()[0].name: inputTensor.to('cpu').detach().numpy().copy()}
    ###########################################################

    ort_outs = ort_session.run(None, ort_inputs)

    ############################################################
    #for i in range(17):
    #    sample = ort_outs[0][i]
    #    cv2.imshow('output', sample)
    #    cv2.waitKey(0)
    ############################################################

    start = time.time()
    for i in range(10):
        ort_outs = ort_session.run(None, ort_inputs)
    delta_time = (time.time() - start)/10*1000
    print("onnx model inference took")
    print(delta_time)
    print("ms")
    ####///////////////////////////////////////

    #print("Done.")


if __name__ == '__main__':
    print(onnxruntime.get_device())
    print(onnxruntime.get_available_providers())
    start = time.time()
    main()
    delta_time = time.time() - start
    print(delta_time)
