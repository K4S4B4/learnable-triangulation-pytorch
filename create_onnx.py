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
    #device = torch.device('cpu') 
    device = torch.device('cuda') 

    #config = cfg.load_config("experiments/human36m/eval/human36m_vol_softmax.yaml")
    config = cfg.load_config("experiments/human36m/eval/human36m_alg.yaml")

    #model = VolumetricTriangulationNet2(config, device=device).to(device)
    #model = BaselinePose2d(config, device=device).to(device)
    #model = AlgebraicTriangulationNet2(config, device=device).to(device)
    #model = VolPose2dFeatureOnly(config, device=device).to(device)
    #model = VolPose2dSpine2dAndFeatures(config, device=device).to(device)
    #model = AlgPose3dTriangulation(config, device=device).to(device)
    #model = VolPose3d(config, device=device).to(device)
    #model = AlgPose3dPreTriangulation(config, device=device).to(device)
    #model = AlgebraicTriangulationNetPermute(config, device=device).to(device)
    #model = AlgebraicHeatmap(config, device=device).to(device)
    model = AlgebraicTriangulationNetPreprocess(config, device=device).to(device)
    
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

    #input = torch.rand(2, 3, 384, 384).to(device)
    #input = torch.rand(1, 384, 384, 3).to(device)
    #input_projMat = torch.rand(5, 3, 4).to(device)
    
    #input_2dpoints = torch.rand(2, 2).to(device)
    #input_confidence = torch.rand(2).to(device)

    #input_volumes = torch.rand(1, 32,64,64,64).to(device)
    #input_coords = createCoordinateVolume(batch_size=1, device=device)

    ##########################################
    ######## Algebraric trianglation ######### 
    ### baseline_pose2d_withConf_2x384x384xBGRxByte (BGR2RGB and Byte2Float included in onnx)
    ##########################################   
    input = torch.rand(1, 384, 384, 3).byte().to(device)
    ### Export the model
    torch.onnx.export(model,               # model being run
                      (input),                         # model input (or a tuple for multiple inputs)
                      "baseline_pose2d_withConf_1x384x384xBGRxByte.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=12,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['images'],   # the model's input names # [n*384*384*3], [n*3*4]
                      output_names = ['joints2d', 'confidence'], # the model's output names # [n*17*2], [n*17]
                      )
    onnx_model = onnx.load("baseline_pose2d_withConf_1x384x384xBGRxByte.onnx")
    onnx.checker.check_model(onnx_model)

    ##########################################
    ######## Algebraric trianglation ########### 
    ### AlgebraicHeatmap   NonZeroを消すために。でもTensorRTを直接やるのはあきらめたので要らないの
    ##########################################   
    ## Export the model
    #torch.onnx.export(model,               # model being run
    #                  (input),                         # model input (or a tuple for multiple inputs)
    #                  "BaselineAlgHeatmapAndConf_fixedBatchsize1.onnx",   # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=12,          # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names = ['images'],   # the model's input names # [n*384*384*3], [n*3*4]
    #                  output_names = ['heatmap', 'confidence'], # the model's output names # [n*17*2], [n*17]
    #                  verbose=True)
    #onnx_model = onnx.load("BaselineAlgHeatmapAndConf_fixedBatchsize1.onnx")
    #onnx.checker.check_model(onnx_model)
    
    
    ##########################################
    ######## Algebraric trianglation ########### 
    ### algPose2dwithConfidence_permuteInput_fixedBatchSize2
    ##########################################   
    #input = torch.rand(2, 384, 384, 3).to(device)
    #### Export the model
    #torch.onnx.export(model,               # model being run
    #                  (input),                         # model input (or a tuple for multiple inputs)
    #                  "baseline_pose2d_withConf_permute_fixedBatchSize2.onnx",   # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=12,          # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names = ['images'],   # the model's input names # [n*384*384*3], [n*3*4]
    #                  output_names = ['joints2d', 'confidence'], # the model's output names # [n*17*2], [n*17]
    #                  )
    #onnx_model = onnx.load("baseline_pose2d_withConf_permute_fixedBatchSize2.onnx")
    #onnx.checker.check_model(onnx_model)

    ##########################################
    ######## Algebraric trianglation ########### 
    ### algPose2dwithConfidence_permuteInput
    ##########################################   
    ### Export the model
    #torch.onnx.export(model,               # model being run
    #                  (input),                         # model input (or a tuple for multiple inputs)
    #                  "BaselineAlgWithConfInputPermute.onnx",   # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=12,          # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names = ['images'],   # the model's input names # [n*384*384*3], [n*3*4]
    #                  output_names = ['joints2d', 'confidence'], # the model's output names # [n*17*2], [n*17]
    #                  dynamic_axes={'images' : {0 : 'batch_size'},    # variable lenght axes これで入出力するTensorのdim=0が可変になる。それ以外の次元は固定
    #                                'joints2d' : {0 : 'batch_size'},
    #                                'confidence' : {0 : 'batch_size'},
    #                                }
    #                  )
    #onnx_model = onnx.load("BaselineAlgWithConfInputPermute.onnx")
    #onnx.checker.check_model(onnx_model)

    ##########################################
    ######## Algebraric trianglation ########### 計算結果がなんか合わない
    ### algPose3dPreTriangulation
    ##########################################   
    ## Export the model
    #torch.onnx.export(model,               # model being run
    #                  (input, input_projMat),                         # model input (or a tuple for multiple inputs)
    #                  "BaselineAlgPreTriangulation5.onnx",   # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=12,          # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names = ['images', 'projMats'],   # the model's input names # [n*384*384*3], [n*3*4]
    #                  output_names = ['nRowsForA', 'confidence'], # the model's output names # [17*n*2*4], [17*n*1*1]
    #                  dynamic_axes={'images' : {0 : 'batch_size'},    # variable lenght axes これで入出力するTensorのdim=0が可変になる。それ以外の次元は固定
    #                                'projMats' : {0 : 'batch_size'},
    #                                'nRowsForA' : {0 : 'batch_size'},
    #                                'confidence' : {0 : 'batch_size'},
    #                                })
    #onnx_model = onnx.load("BaselineAlgPreTriangulation5.onnx")
    #onnx.checker.check_model(onnx_model)

    ##########################################
    ######## V2V ########### 
    ### VolPose3d
    ## →　TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
    ## 警告出たのでうまくできているのかわからん
    ##########################################   
    ### Export the model
    #torch.onnx.export(model,               # model being run
    #                  (input_volumes, input_coords),                         # model input (or a tuple for multiple inputs)
    #                  "v2v_b1.onnx",   # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=12,          # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names = ['volumes', 'coords'],   # the model's input names
    #                  output_names = ['3dPoints'], # the model's output names
    #                  dynamic_axes={'volumes' : {0 : 'batch_size'},  #[1, 32, 64, 64, 64] unproject_heatmapをした後のもの
    #                                'coords' : {0 : 'batch_size'},   #[n, 64, 64, 64, 3] いわゆるmeshgrid
    #                                '3dPoints' : {0 : 'batch_size'}, #[n, 17, 3]
    #                                })
    #onnx_model = onnx.load("v2v_b1.onnx")
    #onnx.checker.check_model(onnx_model)

    ##########################################
    ######## Algebraric trianglation ########### Exporting the operator svd to ONNX opset version 12 is not supported.ってことでダメでした
    ### AlgPose3dTriangulation
    ##########################################   
    ### Export the model
    #torch.onnx.export(model,               # model being run
    #                  (input_projMat, input_2dpoints,input_confidence),                         # model input (or a tuple for multiple inputs)
    #                  "alg_triangulation.onnx",   # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=12,          # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names = ['projMats', '2dPoints', 'confidences'],   # the model's input names
    #                  output_names = ['3dPoints'], # the model's output names
    #                  dynamic_axes={'projMats' : {0 : 'batch_size'},    # variable lenght axes これで入出力するTensorのdim=0が可変になる。それ以外の次元は固定
    #                                '2dPoints' : {0 : 'batch_size'},
    #                                'confidences' : {0 : 'batch_size'},
    #                                '3dPoints' : {0 : 'batch_size'},
    #                                })
    #onnx_model = onnx.load("alg_triangulation.onnx")
    #onnx.checker.check_model(onnx_model)

    #########################################
    ####### BackboneからFeaturesだけ出力するやつ. featuresへの後処理process_featuresも行って、[n, 32, 96, 96]を出力 + SpineのAlg2Dも計算する ##########
    ## →　TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
    ## 警告出たのでうまくできているのかわからん
    ## VolPose2dSpine2dAndFeatures
    #########################################   
    ## Export the model
    #torch.onnx.export(model,               # model being run
    #                  input,                         # model input (or a tuple for multiple inputs)
    #                  "baseline_pose2d_vol_spineAndfeatures.onnx",   # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=10,          # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names = ['input'],   # the model's input names
    #                  output_names = ['jointSpine2d', 'features'], # the model's output names
    #                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes これで入出力するTensorのdim=0が可変になる。それ以外の次元は固定
    #                                'jointSpine2d' : {0 : 'batch_size'},          #[n, 1, 2] Backboneの出力の6番目のHeatmap（Spine）を2d keypointまで計算したもの
    #                                'features' : {0 : 'batch_size'},          #[n, 32, 96, 96]
    #                                })
    #onnx_model = onnx.load("baseline_pose2d_vol_spineAndfeatures.onnx")
    #onnx.checker.check_model(onnx_model)

    ########################################
    ###### BackboneからFeaturesだけ出力するやつ. featuresへの後処理process_featuresも行って、[n, 32, 96, 96]を出力 ###########
    # VolPose2dFeatureOnly
    ########################################   
    # Export the model
    #torch.onnx.export(model,               # model being run
    #                  input,                         # model input (or a tuple for multiple inputs)
    #                  "baseline_pose2d_featuresOnly.onnx",   # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=10,          # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names = ['input'],   # the model's input names
    #                  output_names = ['features'], # the model's output names
    #                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes これで入出力するTensorのdim=0が可変になる。それ以外の次元は固定
    #                                'features' : {0 : 'batch_size'},          #[n, 32, 96, 96]
    #                                })
    #onnx_model = onnx.load("baseline_pose2d_featuresOnly.onnx")
    #onnx.checker.check_model(onnx_model)

    ########################################
    ###### Backboneして2D keypointの取得まで実行するやつ　+ Confidenceも出力 のBatchsize固定版 ########### 
    # AlgebraicTriangulationNet2
    ########################################  
    #input = torch.rand(1, 3, 384, 384).to(device)
    #torch.onnx.export(model,               # model being run
    #                  input,                         # model input (or a tuple for multiple inputs)
    #                  "baseline_pose2d_withConfidence2_fixedBatchSize1.onnx",   # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=10,          # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names = ['input'],   # the model's input names
    #                  output_names = ['joints2d', 'confidence'], # the model's output names
    #                  verbose=True
    #                  )
    #onnx_model = onnx.load("baseline_pose2d_withConfidence2_fixedBatchSize1.onnx")
    #onnx.checker.check_model(onnx_model)

    ########################################
    ###### Backboneして2D keypointの取得まで実行するやつ　+ Confidenceも出力  ########### 
    # AlgebraicTriangulationNet2
    ########################################  
    #input = torch.rand(1, 3, 384, 384).to(device)
    #torch.onnx.export(model,               # model being run
    #                  input,                         # model input (or a tuple for multiple inputs)
    #                  "baseline_pose2d_withConfidence2.onnx",   # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=10,          # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names = ['input'],   # the model's input names
    #                  output_names = ['joints2d', 'confidence'], # the model's output names
    #                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes これで入出力するTensorのdim=0が可変になる。それ以外の次元は固定 # batch_size * 3 * 384 * 384
    #                                'joints2d' : {0 : 'batch_size'}, # batch_size * n_joints * 2
    #                                'confidence' : {0 : 'batch_size'}, # batch_size * n_joints
    #                                }
    #                  )
    #onnx_model = onnx.load("baseline_pose2d_withConfidence2.onnx")
    #onnx.checker.check_model(onnx_model)

    ########################################
    ###### Backboneして2D keypointの取得まで実行するやつ ###########
    # BaselinePose2d
    ########################################  
    #input = torch.rand(1, 3, 384, 384).cpu()
    ## Export the model
    #torch.onnx.export(model,               # model being run
    #                  input,                         # model input (or a tuple for multiple inputs)
    #                  "baseline_pose2d.onnx",   # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=10,          # the ONNX version to export the model to
    #                  do_constant_folding=True,  # whether to execute constant folding for optimization
    #                  input_names = ['input'],   # the model's input names
    #                  output_names = ['joints2d'], # the model's output names
    #                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes これで入出力するTensorのdim=0が可変になる。それ以外の次元は固定
    #                                'joints2d' : {0 : 'batch_size'},                                    
    #                                })
    #onnx_model = onnx.load("baseline_pose2d.onnx")
    #onnx.checker.check_model(onnx_model)

    ########################################
    ###### シンプルにBackboneを実行するやつ ###########
    # VolumetricTriangulationNet2
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

    #########################################
    ####### model 測定 ###########
    #########################################    
    #outputs = model(input)
    #start = time.time()

    #for i in range(10):
    #    outputs = model(input)

    #delta_time = (time.time() - start)/10
    #print("torch model inference took")
    #print(delta_time)
    ##///////////////////////////////////////

    ##########################################
    ######## Onnx 修正 ###########
    ##########################################    
    #onnx_model = onnx.load("baseline_pose2d_withConfidence2_fixedBatchSize1.onnx")
    #graph = onnx_model.graph
    #node  = graph.node

    #nodeIdentity1 = onnx.helper.make_node('Identity', inputs=['1550'], outputs=['1551'])
    #graph.node.remove(node[419]) 
    #graph.node.insert(419, nodeIdentity1) 

    #nodeIdentity2 = onnx.helper.make_node('Identity', inputs=['1559'], outputs=['1560'])
    #graph.node.remove(node[428]) 
    #graph.node.insert(428, nodeIdentity2) 

    #onnx.checker.check_model(onnx_model)
    #onnx.save(onnx_model, 'baseline_pose2d_withConfidence2_fixedBatchSize1_NoNonZero.onnx')

    ###########################################
    ######### Onnx 測定 ###########
    ###########################################    
    #sess_options = onnxruntime.SessionOptions()
    ## Set graph optimization level
    #sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ## To enable model serialization after graph optimization set this
    ##sess_options.optimized_model_filepath = "baseline_pose2d_withConfidence2_fixedBatchSize1.optimized.onnx"
    #sess_options.enable_profiling = True

    ##providers = ['TensorrtExecutionProvider']
    ##providers = ['CUDAExecutionProvider']
    #providers = ['CPUExecutionProvider']

    #ort_session = onnxruntime.InferenceSession("resource/YOLOv4/yolov4-tiny_1_416_416_3_static.onnx", sess_options, providers)

    #input = torch.rand(1, 416, 416, 3).to(device)
    #ort_inputs = {ort_session.get_inputs()[0].name: input.to('cpu').detach().numpy().copy()}
    #ort_outs = ort_session.run(None, ort_inputs)

    #start = time.time()
    #for i in range(10):
    #    ort_outs = ort_session.run(None, ort_inputs)
    #delta_time = (time.time() - start)/10
    #print("onnx model inference took")
    #print(delta_time)
    ####///////////////////////////////////////

    #print("Done.")


if __name__ == '__main__':
    print(onnxruntime.get_device())
    print(onnxruntime.get_available_providers())
    start = time.time()
    main()
    delta_time = time.time() - start
    print(delta_time)
