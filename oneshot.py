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

from mvn.models.triangulation2 import VolumetricTriangulationNet2
from mvn.models.algPose2d import BaselinePose2d

from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.datasets import human36m
from mvn.utils.multiview import Camera

from mvn.utils.img import resize_image, crop_image, normalize_image, image_batch_to_torch

#from detectron2_util import Detectron2util
from CameraCalibration import ArucoCalibrator

import torch.onnx
import onnx
import onnxruntime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
    parser.add_argument("--logdir", type=str, default="/Vol1/dbstore/datasets/k.iskakov/logs/multi-view-net-repr", help="Path, where logs will be stored")
    args = parser.parse_args()
    return args

def visualizeResults(config, writer, images_batch, keypoints_3d_pred, heatmaps_pred, proj_matricies_batch, cuboids_pred, confidences_pred, volumes_pred):        
    batch_size, n_joints = keypoints_3d_pred.shape[0:2]

    # plot visualization
    vis_kind = config.kind
    for batch_i in range(min(batch_size, config.vis_n_elements)):
        keypoints_vis = vis.visualize_batch(
            images_batch, heatmaps_pred, None, proj_matricies_batch,
            None, keypoints_3d_pred,
            kind=vis_kind,
            cuboids_batch=cuboids_pred,
            confidences_batch=confidences_pred,
            batch_index=batch_i, size=5,
            max_n_cols=10
        )
        writer.add_image(f"val/keypoints_vis/{batch_i}", keypoints_vis.transpose(2, 0, 1), global_step=0)

        heatmaps_vis = vis.visualize_heatmaps(
            images_batch, heatmaps_pred,
            kind=vis_kind,
            batch_index=batch_i, size=5,
            max_n_rows=10, max_n_cols=10
        )
        writer.add_image(f"val/heatmaps/{batch_i}", heatmaps_vis.transpose(2, 0, 1), global_step=0)

def setup_experiment(config, model_name, is_train=True):
    prefix = "" if is_train else "eval_"
    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name
    experiment_title = prefix + experiment_title
    experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%d.%m.%Y-%H%M%S")) ###################################
    print("Experiment name: {}".format(experiment_name))
    experiment_dir = os.path.join(args.logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))
    # tensorboard
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))
    # dump config to tensorboard
    writer.add_text(misc.config_to_str(config), "config", 0)
    return experiment_dir, writer

class CaribratedCamera():
    def __init__(self, calibrator, humanDetector, cameraMatrix = None, distCoeffs = None):
        self.calibrator = calibrator
        self.humanDetector = humanDetector
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.isIntrinsicCalibrated = False
        self.isExtrinsicCalibrated = False

    def setCameraIntrinsic(self, cameraMatrix, distCoeffs):
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.isIntrinsicCalibrated = True

    def setCameraIntrinsicByImages(self, calibrationImagesPaths):
        self.isIntrinsicCalibrated, self.cameraMatrix, self.distCoeffs, nView_rvecs, nView_tvecs = self.calibrator.getCameraIntrinsicFromImages(calibrationImagesPaths)
        return self.isIntrinsicCalibrated

    def setCameraTransformByImage(self, imagePath):
        if self.isIntrinsicCalibrated:
            image = cv2.imread(imagePath)
            self.isExtrinsicCalibrated, self.rmat, self.tvecs = self.calibrator.getRmatAndTvecFromImgWithCharuco(image, self.cameraMatrix, self.distCoeffs)
            #########################################
            #if self.isExtrinsicCalibrated:
            #    rvecs, jacob = cv2.Rodrigues(cv2.UMat(np.array(self.rmat)))
            #    tempimg = self.calibrator.drawWorldBox(image, rvecs, self.tvecs, cv2.UMat(np.array(self.cameraMatrix)), cv2.UMat(np.array(self.distCoeffs)))
            #    print(self.rmat)
            #    print(rvecs.get())
            #    print(self.tvecs)
            #    cv2.imshow('img',tempimg)
            #    cv2.waitKey(0) & 0xff
            #########################################
            return self.isExtrinsicCalibrated
        else:
            return False;

    def prepareImageAndProjectionMatrix(self, imagePath, inputShape, outputShape):
        if self.isIntrinsicCalibrated and self.isExtrinsicCalibrated:
            image = cv2.imread(imagePath)
            camera = Camera(self.rmat, self.tvecs, self.cameraMatrix, self.distCoeffs, "NoCameraName") # R, t, K, dist

            # crop
            bbox = self.humanDetector.getBboxOfFirstHuman(image)
            image = crop_image(image, bbox) # 2nd arg is a tuple of (left, upper, right, lower)
            camera.update_after_crop(bbox)

            # resize
            image_shape_before_resize = image.shape[:2]
            image = resize_image(image, inputShape)
            camera.update_after_resize(image_shape_before_resize, outputShape) # Heatmapに対してしか使わないので、先にHeatmapのサイズに合わせておく

            # Normalize
            image = normalize_image(image)
            #cv2.imshow('img',image)
            #cv2.waitKey(0) & 0xff

            # HxWxC -> CxHxW
            tensorImage = torch.tensor(image, dtype=torch.float)
            tensorImage = tensorImage.view(-1, 3)
            tensorImage = torch.transpose(tensorImage, 0, 1)#.contiguous() 
            tensorImage = tensorImage.view(3, 384, 384)

            return True, tensorImage, torch.tensor(camera.projection, dtype=torch.float)
        else:
            return False, None, None

class PoseEstimator():
    def __init__(self, model, calibratedCameras):
        self.model = model
        self.calibratedCameras = calibratedCameras

    def estimatePose(self, nViewImagePaths):
        print("start data prep")
        start = time.time()    

        preparedImages = []
        projMatricies = []
        for i in [0,1,2]:
            success, preparedImage, projMatrix = self.calibratedCameras[i].prepareImageAndProjectionMatrix(nViewImagePaths[i], self.model.inputImageShape, self.model.outputHeatmapShape)
            if success:
                preparedImages.append(preparedImage)
                projMatricies.append(projMatrix)

        images_batch = torch.stack([torch.stack(preparedImages)])
        proj_matricies_batch = torch.stack([torch.stack(projMatricies)])

        print(images_batch)
        print(proj_matricies_batch)

        delta_time = time.time() - start
        print(delta_time)
        
        print("start pose estimation")
        start = time.time()
        keypoints_3d_pred, heatmaps_pred, volumes_pred, coord_volumes_pred = self.model.forward(images_batch, proj_matricies_batch)
        delta_time = time.time() - start
        print(delta_time)
        return keypoints_3d_pred, heatmaps_pred, volumes_pred, coord_volumes_pred, images_batch, proj_matricies_batch

def main(args):
    device = torch.device('cpu') 
    #device = torch.device('cuda') 

    config = cfg.load_config(args.config)
    experiment_dir, writer = setup_experiment(config, "VolumetricTriangulationNet", is_train=False)

    #model = VolumetricTriangulationNet2(config, device=device).to(device)
    model = BaselinePose2d(config, device=device).to(device)
    if config.model.init_weights:
        if device == 'cpu':
            state_dict = torch.load(config.model.checkpoint, map_location='cpu') 
        else:
            state_dict = torch.load(config.model.checkpoint) 
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded pretrained weights for whole model")
    model.eval()

    #input_image = torch.rand(2, 2, 3, 384, 384).cpu() #.cpu() .cuda()
    #input_projMat = torch.rand(2, 2, 3, 4).cpu() #.cpu() .cuda()

    input = torch.rand(1, 3, 384, 384).cpu()

    # Export the model
    torch.onnx.export(model,               # model being run
                      input,                         # model input (or a tuple for multiple inputs)
                      "baseline_pose2d_2.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['joints2d'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes これで入出力するTensorのdim=0が可変になる。それ以外の次元は固定
                                    'joints2d' : {0 : 'batch_size'},                                    
                                    })
    onnx_model = onnx.load("baseline_pose2d_2.onnx")
    onnx.checker.check_model(onnx_model)

    ########################################
    ###### Onnx 作成  ###########
    ####### シンプルにbackboneを実行するやつ
    ########################################    
    #input = torch.rand(1, 3, 384, 384).cuda() #.cpu()
    ## Export the model
    #torch.onnx.export(model.backbone,               # model being run
    #                  input,                         # model input (or a tuple for multiple inputs)
    #                  "onnx_pose2d_gpu.onnx",   # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=10,          # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names = ['input'],   # the model's input names
    #                  output_names = ['heatmaps', 'features'], # the model's output names
    #                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes これで入出力するTensorのdim=0が可変になる。それ以外の次元は固定
    #                                'heatmaps' : {0 : 'batch_size'},
    #                                'features' : {0 : 'batch_size'},                                    
    #                                })
    #onnx_model = onnx.load("onnx_pose2d_gpu.onnx")
    #onnx.checker.check_model(onnx_model)

    ########################################
    ###### Onnx 測定 ###########
    ########################################    
    #ort_session = onnxruntime.InferenceSession("onnx_pose2d_gpu.onnx")
    # compute ONNX Runtime output prediction
    #ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
        #print('Normal model inferece')
    #start = time.time()
    #ort_outs = ort_session.run(None, ort_inputs)
    #delta_time = time.time() - start
    #print(delta_time)
    ##///////////////////////////////////////

    ########################################
    ###### TorchScript 作成の墓場 ###########
    ########################################
    #script_model = torch.jit.script(model.backbone)
    #torch.jit.save(script_model, "script_pose2d_gpu.pt")

    #traced_cpu = torch.jit.trace(model.backbone, example_inputs=input)
    #torch.jit.save(traced_cpu, "traced_pose2d_gpu.pt")

    #print('Normal model inferece')
    #start = time.time()
    #model.backbone(input)
    #delta_time = time.time() - start
    #print(delta_time)
    
    #print('TorchScript traced model inferece')
    #start = time.time()
    #traced_cpu(input)
    #delta_time = time.time() - start
    #print(delta_time)

    #print('TorchScript model inferece')
    #start = time.time()
    #script_model(input)
    #delta_time = time.time() - start
    #print(delta_time)


    calibrationImagesPaths =[
        "testdata/IMG_20210208_135527.jpg",
        "testdata/IMG_20210208_135532.jpg",
        "testdata/IMG_20210209_114242.jpg",
        "testdata/IMG_20210209_114246.jpg",
        "testdata/IMG_20210209_114249.jpg",
        "testdata/IMG_20210209_114255.jpg",
        "testdata/IMG_20210209_114300.jpg",
        "testdata/IMG_20210209_114305.jpg",
        "testdata/IMG_20210209_114311.jpg",
        "testdata/IMG_20210209_114318.jpg",
        "testdata/IMG_20210209_114323.jpg"
        ]
    K_test =[
        [3.03742663e+03, 0.00000000e+00, 1.44546191e+03],
        [0.00000000e+00, 3.00724975e+03, 2.06361113e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    dist_test = [[ 0.32807458, -0.92854823,  0.01082845, -0.0131077,   1.22518482]]
    nView_image_paths =[
        "testdata/IMG_20210210_202957.jpg",
        "testdata/IMG_20210210_203002.jpg",
        "testdata/IMG_20210210_203017.jpg"
        ]

    caribrator = ArucoCalibrator()
    #humanDetector = Detectron2util()
    calibratedCameras = []
    for i in [0,1,2]:
        #calibratedCamera = CaribratedCamera(caribrator, humanDetector)
        calibratedCamera.setCameraIntrinsicByImages(calibrationImagesPaths)
        #print(calibratedCamera.cameraMatrix)
        #print(calibratedCamera.distCoeffs)
        #calibratedCamera.setCameraIntrinsic(K_test, dist_test) # for quick test
        calibratedCamera.setCameraTransformByImage(nView_image_paths[i])
        calibratedCameras.append(calibratedCamera)

    poseEstimator = PoseEstimator(model, calibratedCameras)
    
    grad_context = torch.no_grad # used to turn on/off gradients
    with grad_context():
        keypoints_3d_pred, heatmaps_pred, volumes_pred, coord_volumes_pred, images_batch, proj_matricies_batch = poseEstimator.estimatePose(nView_image_paths)
        print(keypoints_3d_pred)

    visualizeResults(config, writer, images_batch, keypoints_3d_pred, heatmaps_pred, proj_matricies_batch, None, None, volumes_pred)

    print("Done.")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)
