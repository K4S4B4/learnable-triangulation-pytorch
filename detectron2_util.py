# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

class Detectron2util():
    def __init__(self):
        self.cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
        self.predictor = DefaultPredictor(self.cfg)

    def getBboxOfFirstHuman(self, frame):
        outputs = self.predictor(frame)

        for clazz, bbox in zip(outputs["instances"].pred_classes, outputs["instances"].pred_boxes):
            if clazz.item() == 0:
                return self.square_the_bbox(bbox)

        return None

        #v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2.imshow('frame',out.get_image()[:, :, ::-1])
        #while(True):
        #    if cv2.waitKey(1) & 0xFF == ord('q'):
        #        break
        #return outputs["instances"]

    def square_the_bbox(self, bbox):
        left, top, right, bottom = (bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()) # Bbox to a tuple of (left, top, right, bottom)
        width = right - left
        height = bottom - top

        if height < width:
            center = (top + bottom) * 0.5
            top = int(round(center - width * 0.5))
            bottom = top + width
        else:
            center = (left + right) * 0.5
            left = int(round(center - height * 0.5))
            right = left + height

        return left, top, right, bottom

    def visualizeResult(self, frame):
        outputs = self.predictor(frame)
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow('frame',out.get_image()[:, :, ::-1])
        img = out.get_image()[:, :, ::-1]
        #cv2.imshow('frame',img)
        cv2.waitKey(1) & 0xFF == ord('q')
        return img


