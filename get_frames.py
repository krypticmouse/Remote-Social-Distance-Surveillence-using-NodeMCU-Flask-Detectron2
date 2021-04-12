from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from firebase_utils import post_data

import numpy as np
from scipy.spatial import distance as dist
import cv2

cfg = get_cfg()
print('done')
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
print('done')
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
print('done')
cfg.MODEL.DEVICE = 'cuda'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
print('done')
predictor = DefaultPredictor(cfg)
print('done')

def gen_frames(path,idx):  
    cap = cv2.VideoCapture(path)

    while cap.isOpened:
        ret,frame = cap.read()
        if not ret:
            break
            
        outputs = predictor(frame)
        points,boxes = [],[]
        
        for box,label in zip(outputs['instances'].pred_boxes, outputs['instances'].pred_classes):
            if label.cpu().numpy() == 0:
                boxes.append(box.cpu().numpy())
                x,y,w,h = boxes[-1]
                center = (int((x+w)/2), int((y+h)/2))
                points.append(center)
        
        is_violated = False
                
        if len(points) >= 2:
            d = dist.cdist(points, points, metric="euclidean")
            sec_min =  np.sort(d,axis = 1)[:,1]      
            for box,sec in zip(boxes,sec_min):
                color = (0,0,255) if sec < 70 else (0,255,0)
                is_violated = (is_violated or color == (0,0,255))
                x,y,w,h = box
                cv2.rectangle(frame,(x,y),(w,h),color,2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        post_data(idx,is_violated)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')