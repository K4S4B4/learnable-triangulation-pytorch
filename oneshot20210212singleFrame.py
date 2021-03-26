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
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

from mvn.models.triangulation2 import VolumetricTriangulationNet2
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss

from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.datasets import human36m
from mvn.datasets import utils as dataset_utils
from mvn.utils.multiview import Camera

from mvn.utils.img import get_square_bbox, resize_image, crop_image, normalize_image, scale_bbox, image_batch_to_torch

from detectron2_util import Detectron2util
from CameraCalibration import ArucoCalibrator

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
    #parser.add_argument("--configAlg", type=str, required=True, help="Path, where config file is stored")
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
    parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")

    parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--logdir", type=str, default="/Vol1/dbstore/datasets/k.iskakov/logs/multi-view-net-repr", help="Path, where logs will be stored")

    args = parser.parse_args()
    return args


def setup_human36m_dataloaders(config, is_train, distributed_train):
    train_dataloader = None
    if is_train:
        # train
        train_dataset = human36m.Human36MMultiViewDataset(
            h36m_root=config.dataset.train.h36m_root,
            pred_results_path=config.dataset.train.pred_results_path if hasattr(config.dataset.train, "pred_results_path") else None,
            train=True,
            test=False,
            image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
            labels_path=config.dataset.train.labels_path,
            with_damaged_actions=config.dataset.train.with_damaged_actions,
            scale_bbox=config.dataset.train.scale_bbox,
            kind=config.kind,
            undistort_images=config.dataset.train.undistort_images,
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [],
            crop=config.dataset.train.crop if hasattr(config.dataset.train, "crop") else True,
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.train.shuffle and (train_sampler is None), # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.train.randomize_n_views,
                                                     min_n_views=config.dataset.train.min_n_views,
                                                     max_n_views=config.dataset.train.max_n_views),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True
        )

    # val
    val_dataset = human36m.Human36MMultiViewDataset(
        h36m_root=config.dataset.val.h36m_root,
        pred_results_path=config.dataset.val.pred_results_path if hasattr(config.dataset.val, "pred_results_path") else None,
        train=False,
        test=True,
        image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
        labels_path=config.dataset.val.labels_path,
        with_damaged_actions=config.dataset.val.with_damaged_actions,
        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
        scale_bbox=config.dataset.val.scale_bbox,
        kind=config.kind,
        undistort_images=config.dataset.val.undistort_images,
        ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
        crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                 min_n_views=config.dataset.val.min_n_views,
                                                 max_n_views=config.dataset.val.max_n_views),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, train_sampler


def setup_dataloaders(config, is_train=True, distributed_train=False):
    if config.dataset.kind == 'human36m':
        train_dataloader, val_dataloader, train_sampler = setup_human36m_dataloaders(config, is_train, distributed_train)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))

    return train_dataloader, val_dataloader, train_sampler


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

def makeNViewImagesAndCameras():
    sample = defaultdict(list) # return value

    calib = ArucoCalibrator()
    image_paths_test =[
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
    calibration, cameraMatrix, distCoeffs, nView_rvecs, nView_tvecs = calib.getCameraIntrinsicFromImages(image_paths_test)

    print(cameraMatrix)
    print(distCoeffs)

    image_paths_test =[
        "testdata/IMG_20210210_202957.jpg",
        "testdata/IMG_20210210_203002.jpg",
        "testdata/IMG_20210210_203017.jpg"
        ]

    #camera_names_test = ['54138969', '55011271', '58860488', '60457274']
    #image_paths_test =[
    #    "C:\\Users\\User\\Downloads\\learnable-triangulation-pytorch-master\\data\\human36m\\processed\\S9\\Posing-1\\imageSequence\\54138969\\img_001771.jpg",
    #    "C:\\Users\\User\\Downloads\\learnable-triangulation-pytorch-master\\data\\human36m\\processed\\S9\\Posing-1\\imageSequence\\55011271\\img_001771.jpg",
    #    "C:\\Users\\User\\Downloads\\learnable-triangulation-pytorch-master\\data\\human36m\\processed\\S9\\Posing-1\\imageSequence\\58860488\\img_001771.jpg",
    #    "C:\\Users\\User\\Downloads\\learnable-triangulation-pytorch-master\\data\\human36m\\processed\\S9\\Posing-1\\imageSequence\\60457274\\img_001771.jpg"
    #    ]
    #bbox_test = [
    #    [221, 185, 616, 580],
    #    [201, 298, 649, 746],
    #    [182, 416, 554, 788],
    #    [120, 332, 690, 902]
    #    ]
  #  R_test = [
  #      [[-0.9153617,   0.40180838,  0.02574755],
  #       [ 0.05154812,  0.18037356, -0.9822465 ],
  #       [-0.39931902, -0.89778364, -0.18581952]],
		#[[ 0.92816836,  0.37215385,  0.00224838],
		# [ 0.08166409, -0.1977723,  -0.9768404 ],
		# [-0.36309022,  0.9068559,  -0.2139576 ]],
		#[[-0.91415495, -0.40277803, -0.04572295],
		# [-0.04562341,  0.2143085,  -0.97569996],
		# [ 0.4027893 , -0.8898549,  -0.21428728]],
		#[[ 0.91415626, -0.40060705,  0.06190599],
		# [-0.05641001, -0.2769532,  -0.9592262 ],
		# [ 0.40141782,  0.8733905,  -0.27577674]]
  #      ]
  #  t_test = [
		#[[-346.0509 ],
		# [ 546.98083],
		# [5474.481  ]],
		#[[ 251.4252 ],
		# [ 420.94223],
		# [5588.196  ]],
		#[[ 480.4826 ],
		# [ 253.83238],
		# [5704.2075 ]],
		#[[  51.883537],
		# [ 378.4209  ],
		# [4406.1494  ]]
  #      ]
  #  K_test = [
		#[[1.1450494e+03, 0.0000000e+00, 5.1254150e+02],
		# [0.0000000e+00, 1.1437811e+03, 5.1545148e+02],
		# [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]],
		#[[1.1496757e+03, 0.0000000e+00, 5.0884863e+02],
		# [0.0000000e+00, 1.1475917e+03, 5.0806491e+02],
		# [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]],
		#[[1.1491407e+03, 0.0000000e+00, 5.1981586e+02],
		# [0.0000000e+00, 1.1487990e+03, 5.0140265e+02],
		# [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]],
		#[[1.1455114e+03, 0.0000000e+00, 5.1496820e+02],
		# [0.0000000e+00, 1.1447739e+03, 5.0188202e+02],
		# [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]]
  #      ]
  #  dist_test = [
		#[-0.20709892,  0.24777518, -0.00142447, -0.0009757,  -0.00307515],
		#[-0.19421363,  0.24040854, -0.00274089, -0.00161903,  0.00681998],
		#[-0.20833819,  0.255488,   -0.00076,     0.00148439, -0.0024605 ],
		#[-0.19838409,  0.21832368, -0.00181336, -0.00058721, -0.00894781]
  #      ]   

    images = []
    cameras = []
    for i, image_path in enumerate(image_paths_test):

        image = cv2.imread(image_paths_test[i])
        #retval_camera = Camera(R_test[i], t_test[i], K_test[i], dist_test[i], camera_names_test[i])

        retval, rmat, tvecs = calib.getRmatAndTvecFromImgWithCharuco(image, cameraMatrix, distCoeffs)
        if not retval:
            continue
        retval_camera = Camera(rmat, tvecs, cameraMatrix, distCoeffs, "asaba_cell_photo") # R, t, K, dist
        ############################
        rvecs, jacob = cv2.Rodrigues(cv2.UMat(np.array(rmat)))
        #tempimg = calib.drawWorldBox(image, rvecs, tvecs, cameraMatrix, distCoeffs)
        print(rmat)
        print(rvecs.get())
        print(tvecs)
        print(retval_camera.projection)
        #cv2.imshow('img',tempimg)
        #cv2.waitKey(0) & 0xff
        ############################

        # Get bbox
        det2u = Detectron2util()
        bbox = det2u.getBboxOfFirstHuman(image)

        # crop
        image = crop_image(image, bbox) # 2nd arg is a tuple of (left, upper, right, lower)
        retval_camera.update_after_crop(bbox)

        # resize
        image_shape_before_resize = image.shape[:2]
        image = resize_image(image, [384, 384])
        retval_camera.update_after_resize(image_shape_before_resize, [96, 96]) #Heatmap後にしか使わないので、先にHeatmapのサイズに合わせてCameraIntrinsicを修正

        # Normalize
        image = normalize_image(image) # 384x384x3
        #cv2.imshow('img',image)
        #cv2.waitKey(0) & 0xff

        # HxWxC -> CxHxW
        tensorImage = torch.tensor(image, dtype=torch.float)
        tensorImage = tensorImage.view(-1, 3)
        tensorImage = torch.transpose(tensorImage, 0, 1)#.contiguous() 
        tensorImage = tensorImage.view(3, 384, 384)
        images.append(tensorImage)

        cameras.append(torch.tensor(retval_camera.projection, dtype=torch.float))

    return torch.stack(images), torch.stack(cameras)

#def one_epoch(model, criterion, opt, config, dataloader, device, epoch, n_iters_total=0, is_train=True, caption='', master=False, experiment_dir=None, writer=None):
#def one_epoch(model, config, modelAlg, configAlg, device, epoch, n_iters_total=0, is_train=True, caption='', experiment_dir=None, writer=None):
def one_epoch(model, config, device, epoch, n_iters_total=0, is_train=True, caption='', experiment_dir=None, writer=None):
    #name = "train" if is_train else "val"
    name =  "val"

    model_type = config.model.name

    #if is_train:
    #    model.train()
    #else:
    #    model.eval()
    model.eval()

    metric_dict = defaultdict(list)
    results = defaultdict(list)

    # used to turn on/off gradients
    #grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    grad_context = torch.no_grad
    with grad_context():
        end = time.time()

        #iterator = enumerate(dataloader)
        #if is_train and config.opt.n_iters_per_epoch is not None:
        #    iterator = islice(iterator, config.opt.n_iters_per_epoch)


        # measure data loading time
        data_time = time.time() - end

        #if batch is None:
        #    print("Found None batch")
        #    continue

        #images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = dataset_utils.prepare_batch(batch, device, config)

        #item = makeNViewImagesAndCameras()
        images, cameras = makeNViewImagesAndCameras()


        images_batch = torch.stack([images])
        proj_matricies_batch = torch.stack([cameras])

        print(images_batch)
        print(proj_matricies_batch)
        ######### VOL ##########
        keypoints_3d_pred, heatmaps_pred, volumes_pred, coord_volumes_pred = model(images_batch, proj_matricies_batch)
        print(keypoints_3d_pred)

        batch_size, n_views, image_shape = images_batch.shape[0], images_batch.shape[1], tuple(images_batch.shape[3:])
        n_joints = keypoints_3d_pred.shape[1]

        # plot visualization
        if n_iters_total % config.vis_freq == 0:# or total_l2.item() > 500.0:
            vis_kind = config.kind
            if (config.transfer_cmu_to_human36m if hasattr(config, "transfer_cmu_to_human36m") else False):
                vis_kind = "coco"

            for batch_i in range(min(batch_size, config.vis_n_elements)):
                keypoints_vis = vis.visualize_batch(
                    images_batch, heatmaps_pred, None, proj_matricies_batch,
                    None, keypoints_3d_pred,
                    kind=vis_kind,
                    cuboids_batch=None,
                    confidences_batch=None,
                    batch_index=batch_i, size=5,
                    max_n_cols=10
                )
                writer.add_image(f"{name}/keypoints_vis/{batch_i}", keypoints_vis.transpose(2, 0, 1), global_step=n_iters_total)

                heatmaps_vis = vis.visualize_heatmaps(
                    images_batch, heatmaps_pred,
                    kind=vis_kind,
                    batch_index=batch_i, size=5,
                    max_n_rows=10, max_n_cols=10
                )
                writer.add_image(f"{name}/heatmaps/{batch_i}", heatmaps_vis.transpose(2, 0, 1), global_step=n_iters_total)

                #if model_type == "vol":
                #    volumes_vis = vis.visualize_volumes(
                #        images_batch, volumes_pred, proj_matricies_batch,
                #        kind=vis_kind,
                #        cuboids_batch=None,
                #        batch_index=batch_i, size=5,
                #        max_n_rows=1, max_n_cols=16
                #    )
                #    writer.add_image(f"{name}/volumes/{batch_i}", volumes_vis.transpose(2, 0, 1), global_step=n_iters_total)

        ## dump weights to tensoboard
        #if n_iters_total % config.vis_freq == 0:
        #    for p_name, p in model.named_parameters():
        #        try:
        #            writer.add_histogram(p_name, p.clone().cpu().data.numpy(), n_iters_total)
        #        except ValueError as e:
        #            print(e)
        #            print(p_name, p)
        #            exit()

        ## dump to tensorboard per-iter loss/metric stats
        #if is_train:
        #    for title, value in metric_dict.items():
        #        writer.add_scalar(f"{name}/{title}", value[-1], n_iters_total)

        ## measure elapsed time
        #batch_time = time.time() - end
        #end = time.time()

        ## dump to tensorboard per-iter time stats
        #writer.add_scalar(f"{name}/batch_time", batch_time, n_iters_total)
        #writer.add_scalar(f"{name}/data_time", data_time, n_iters_total)

        ## dump to tensorboard per-iter stats about sizes
        #writer.add_scalar(f"{name}/batch_size", batch_size, n_iters_total)
        #writer.add_scalar(f"{name}/n_views", n_views, n_iters_total)

        #n_iters_total += 1

    ## calculate evaluation metrics
    #if master:
    #    if not is_train:
    #        results['keypoints_3d'] = np.concatenate(results['keypoints_3d'], axis=0)
    #        results['indexes'] = np.concatenate(results['indexes'])

    #        try:
    #            scalar_metric, full_metric = dataloader.dataset.evaluate(results['keypoints_3d'])
    #        except Exception as e:
    #            print("Failed to evaluate. Reason: ", e)
    #            scalar_metric, full_metric = 0.0, {}

    #        metric_dict['dataset_metric'].append(scalar_metric)

    #        checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
    #        os.makedirs(checkpoint_dir, exist_ok=True)

    #        ## dump results
    #        #with open(os.path.join(checkpoint_dir, "results.pkl"), 'wb') as fout:
    #        #    pickle.dump(results, fout)

    #        # dump full metric
    #        with open(os.path.join(checkpoint_dir, "metric.json".format(epoch)), 'w') as fout:
    #            json.dump(full_metric, fout, indent=4, sort_keys=True)

    #    # dump to tensorboard per-epoch stats
    #    for title, value in metric_dict.items():
    #        writer.add_scalar(f"{name}/{title}_epoch", np.mean(value), epoch)

    return n_iters_total


def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        return False

    torch.cuda.set_device(args.local_rank)

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"

    torch.manual_seed(args.seed)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return True


def main(args):
    #print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    #is_distributed = init_distributed(args)
    #master = True
    #if is_distributed and os.environ["RANK"]:
    #    master = int(os.environ["RANK"]) == 0

    #if is_distributed:
    #    device = torch.device(args.local_rank)
    #else:
    #    device = torch.device(0)
    device = torch.device('cpu') ######################################################

    # config
    config = cfg.load_config(args.config)
    #configAlg = cfg.load_config(args.configAlg)######### ALG ##########
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size

    #model = {
    #    "ransac": RANSACTriangulationNet,
    #    "alg": AlgebraicTriangulationNet,
    #    "vol": VolumetricTriangulationNet
    #}[config.model.name](config, device=device).to(device)
    #modelAlg = AlgebraicTriangulationNet(configAlg, device=device).to(device)######### ALG ##########
    model = VolumetricTriangulationNet2(config, device=device).to(device)

    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint, map_location='cpu') ######################################################
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded pretrained weights for whole model")

    ######### ALG ##########
    #if configAlg.model.init_weights:
    #    state_dict = torch.load(configAlg.model.checkpoint, map_location='cpu') ######################################################
    #    for key in list(state_dict.keys()):
    #        new_key = key.replace("module.", "")
    #        state_dict[new_key] = state_dict.pop(key)

    #    modelAlg.load_state_dict(state_dict, strict=True)
    #    print("Successfully loaded pretrained weights for whole model")
    ######### ALG ##########

    # criterion
    #criterion_class = {
    #    "MSE": KeypointsMSELoss,
    #    "MSESmooth": KeypointsMSESmoothLoss,
    #    "MAE": KeypointsMAELoss
    #}[config.opt.criterion]

    #if config.opt.criterion == "MSESmooth":
    #    criterion = criterion_class(config.opt.mse_smooth_threshold)
    #else:
    #    criterion = criterion_class()

    # optimizer
    #opt = None
    #if not args.eval:
    #    if config.model.name == "vol":
    #        opt = torch.optim.Adam(
    #            [{'params': model.backbone.parameters()},
    #             {'params': model.process_features.parameters(), 'lr': config.opt.process_features_lr if hasattr(config.opt, "process_features_lr") else config.opt.lr},
    #             {'params': model.volume_net.parameters(), 'lr': config.opt.volume_net_lr if hasattr(config.opt, "volume_net_lr") else config.opt.lr}
    #            ],
    #            lr=config.opt.lr
    #        )
    #    else:
    #        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.opt.lr)

    ## datasets
    #print("Loading data...")
    #train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=is_distributed)

    # experiment
    #experiment_dir, writer = None, None
    #if master:
    #    experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)
    experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)

    # multi-gpu
    #if is_distributed:
    #    model = DistributedDataParallel(model, device_ids=[device])

    #if not args.eval:
    #    # train loop
    #    n_iters_total_train, n_iters_total_val = 0, 0
    #    for epoch in range(config.opt.n_epochs):
    #        if train_sampler is not None:
    #            train_sampler.set_epoch(epoch)

    #        n_iters_total_train = one_epoch(model, criterion, opt, config, train_dataloader, device, epoch, n_iters_total=n_iters_total_train, is_train=True, master=master, experiment_dir=experiment_dir, writer=writer)
    #        n_iters_total_val = one_epoch(model, criterion, opt, config, val_dataloader, device, epoch, n_iters_total=n_iters_total_val, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)

    #        if master:
    #            checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
    #            os.makedirs(checkpoint_dir, exist_ok=True)

    #            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))

    #        print(f"{n_iters_total_train} iters done.")
    #else:
    #    if args.eval_dataset == 'train':
    #        one_epoch(model, criterion, opt, config, train_dataloader, device, 0, n_iters_total=0, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)
    #    else:
    #        one_epoch(model, criterion, opt, config, val_dataloader, device, 0, n_iters_total=0, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)

    one_epoch(model, config, device, 0, n_iters_total=0, is_train=False, experiment_dir=experiment_dir, writer=writer)
    #one_epoch(model, config, modelAlg, configAlg, device, 0, n_iters_total=0, is_train=False, experiment_dir=experiment_dir, writer=writer)

    print("Done.")

if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)
