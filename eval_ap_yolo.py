###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012                   #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012                   #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: May 24th 2018                                                 #
###########################################################################################

import _init_paths
import json, pickle
import numpy as np
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import *


def getBoundingBoxes():
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()
    import glob
    import os
    # Read ground truths
    pth = "../Wildtrack_dataset/annotations_positions/*.json"
    files = glob.glob(pth)
    files.sort()
    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    # Read GT detections from txt files
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for idx, f in enumerate(files):
        with open(f) as j:
            data = json.load(j)
        # print(f)
        for d in data:
            for v in d['views']:
                if v['viewNum'] != 0 or v['xmax'] == -1:
                    continue

                idClass = 'person'  # class
                x = float(v['xmin'])  # confidence
                y = float(v['ymin'])
                w = float(v['xmax'])
                h = float(v['ymax'])
                bb = BoundingBox(
                    str(idx),
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    CoordinatesType.Absolute, (1920, 1080),
                    BBType.GroundTruth,
                    format=BBFormat.XYX2Y2)
                allBoundingBoxes.addBoundingBox(bb)
        


    # Read detections
    with open("wildtrack_yolo_tiny.out", "rb") as fin:
        pred = pickle.load(fin)
    for idx, (k, value) in enumerate(pred.items()):
        for d in value:
            if d['tag'] != "person":
                continue
            
            box = d['box']
            idClass = 'person'  # class
            x = float(box[0])  # confidence
            y = float(box[1])
            w = float(box[2])
            h = float(box[3])
            bb = BoundingBox(
                str(idx),
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (1920, 1080),
                BBType.Detected,
                d['score'],
                format=BBFormat.XYX2Y2)
            allBoundingBoxes.addBoundingBox(bb)
    return allBoundingBoxes


# # def createImages(dictGroundTruth, dictDetected):
# #     """Create representative images with bounding boxes."""
# #     import numpy as np
# #     import cv2
# #     # Define image size
# #     width = 200
# #     height = 200
# #     # Loop through the dictionary with ground truth detections
# #     for key in dictGroundTruth:
# #         image = np.zeros((height, width, 3), np.uint8)
# #         gt_boundingboxes = dictGroundTruth[key]
# #         image = gt_boundingboxes.drawAllBoundingBoxes(image)
# #         detection_boundingboxes = dictDetected[key]
# #         image = detection_boundingboxes.drawAllBoundingBoxes(image)
# #         # Show detection and its GT
# #         cv2.imshow(key, image)
# #         cv2.waitKey()


# Read txt files containing bounding boxes (ground truth and detections)
boundingboxes = getBoundingBoxes()
# Uncomment the line below to generate images based on the bounding boxes
# createImages(dictGroundTruth, dictDetected)
# Create an evaluator object in order to obtain the metrics
evaluator = Evaluator()
##############################################################
# VOC PASCAL Metrics
##############################################################
# Plot Precision x Recall curve
# evaluator.PlotPrecisionRecallCurve(
#     boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
#     IOUThreshold=0.3,  # IOU threshold
#     method=MethodAveragePrecision.EveryPointInterpolation,  # As the official matlab code
#     showAP=True,  # Show Average Precision in the title of the plot
# #     showInterpolatedPrecision=True)  # Plot the interpolated precision curve
# Get metrics with PASCAL VOC metrics
metricsPerClass = evaluator.GetPascalVOCMetrics(
    boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
    IOUThreshold=0.3,  # IOU threshold
    method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
print("Average precision values per class:\n")
# Loop through classes to obtain their metrics
for mc in metricsPerClass:
    # Get metric values per each class
    c = mc['class']
    precision = mc['precision']
    recall = mc['recall']
    average_precision = mc['AP']
    ipre = mc['interpolated precision']
    irec = mc['interpolated recall']
    # Print AP per class
    print('%s: %f' % (c, average_precision))