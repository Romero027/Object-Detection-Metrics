import _init_paths
from utils import *
import json
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes

dict = [{'box': [1614, 184, 1694, 366], 'label': 'person', 'score': 0.67870224}, {'box': [243, 191, 309, 292], 'label': 'person', 'score': 0.6672403}, {'box': [1246, 138, 1301, 288], 'label': 'person', 'score': 0.64157844}, {'box': [943, 134, 1000, 275], 'label': 'person', 'score': 0.6350653}, {'box': [51, 52, 1877, 477], 'label': 'person', 'score': 0.6077829}, {'box': [112, 223, 200, 410], 'label': 'person', 'score': 0.59162873}, {'box': [954, 387, 1188, 795], 'label': 'person', 'score': 0.55491996}, {'box': [738, 117, 794, 264], 'label': 'person', 'score': 0.5538351}, {'box': [1519, 763, 1902, 1080], 'label': 'person', 'score': 0.55083704}, {'box': [1455, 168, 1513, 317], 'label': 'person', 'score': 0.5340955}, {'box': [1185, 133, 1243, 305], 'label': 'person', 'score': 0.50797313}, {'box': [1551, 172, 1597, 319], 'label': 'person', 'score': 0.49383837}, {'box': [892, 393, 1220, 879], 'label': 'person', 'score': 0.474105}, {'box': [29, 162, 71, 228], 'label': 'person', 'score': 0.46673012}, {'box': [51, 40, 1877, 1038], 'label': 'person', 'score': 0.46512935}, {'box': [451, 260, 506, 348], 'label': 'person', 'score': 0.4571249}, {'box': [517, 253, 584, 340], 'label': 'person', 'score': 0.4557678}, {'box': [1047, 134, 1104, 283], 'label': 'person', 'score': 0.4549216}, {'box': [1352, 171, 1412, 312], 'label': 'person', 'score': 0.441226}, {'box': [340, 207, 390, 278], 'label': 'person', 'score': 0.43963793}, {'box': [1352, 171, 1412, 312], 'label': 'bike', 'score': 0.3757741}, {'box': [958, 191, 1012, 317], 'label': 'person', 'score': 0.3656317}, {'box': [15, 195, 86, 323], 'label': 'person', 'score': 0.31937876}, {'box': [1634, 221, 1708, 379], 'label': 'person', 'score': 0.31857038}]

with open('./00000000.json') as f:
  data = json.load(f)

gt_boundingBoxs = []
for d in data:
    for v in d['views']:
        if v['viewNum'] != 0 or v['xmax'] == -1:
            continue
        print(v)
        gt_boundingBoxs.append(BoundingBox(imageName='cam1_0.png', classId='person', x=v['xmin'], y=v['ymin'],
                               w=v['xmax'], h=v['ymax'], typeCoordinates=CoordinatesType.Absolute,
                               bbType=BBType.GroundTruth, format=BBFormat.XYX2Y2, imgSize=(1920,1080)))
       
    

detected_boundingBoxs = []
for d in dict:
    box = d['box']
    if d['label'] != 'person' or d['score'] < 0.5:
        continue
    detected_boundingBoxs.append(BoundingBox(imageName='cam1_0.png', classId='person', classConfidence= d['score'],
                                     x=box[0], y=box[1], w=box[2], h=box[3], typeCoordinates=CoordinatesType.Absolute,
                                     bbType=BBType.Detected, format=BBFormat.XYX2Y2, imgSize=(1920,1080)))
                


# Creating the object of the class BoundingBoxes
myBoundingBoxes = BoundingBoxes()

for box in gt_boundingBoxs:
    myBoundingBoxes.addBoundingBox(box)

for box in detected_boundingBoxs:
    myBoundingBoxes.addBoundingBox(box)


import cv2
import numpy as np
import os
currentPath = os.path.dirname(os.path.realpath(__file__))
gtImages = ['cam1_0.png']
for imageName in gtImages:
    im = cv2.imread("./cam1_0.png")
    # Add bounding boxes
    im = myBoundingBoxes.drawAllBoundingBoxes(im, imageName)
    # Uncomment the lines below if you want to show the images
    # cv2.imshow(imageName+'.jpg', im)
    # cv2.waitKey(0)
    cv2.imwrite("./dt.png",im)
    print('Image %s created successfully!' % imageName)