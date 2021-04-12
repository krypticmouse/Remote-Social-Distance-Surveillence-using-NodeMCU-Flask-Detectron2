from firebase_utils import post_data
import numpy as np
from scipy.spatial import distance as dist
import cv2

yolo = cv2.dnn.readNet('yolov3.weights','yolov3-tiny.cfg')
output_layers = yolo.getUnconnectedOutLayersNames()

def gen_frames(path,idx):
    cap = cv2.VideoCapture(path)

    while(cap.isOpened()):
        ret, img = cap.read()

        if not ret:
            break

        blob = cv2.dnn.blobFromImage(img,0.004,(416,416),(0,0,0),True,False)
        yolo.setInput(blob)
        output = yolo.forward(output_layers)


        height,width,channels = img.shape

        boxes = []
        scores = []

        for out in output:
            for detect in out:
                confidence_scores = detect[5:]
                class_id = np.argmax(confidence_scores)
                class_score = confidence_scores[class_id]
                if class_score > 0.5 and class_id == 0:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x,y,w,h])
                    scores.append(float(class_score))

        is_violated = False

        if len(boxes) != 0:
            indexes = cv2.dnn.NMSBoxes(boxes,scores,0.5,0.4)
            box,points = [],[]

            for i in indexes:
                box.append(boxes[i[0]])
                x,y,w,h = boxes[i[0]]
                center = (int((x+w)/2), int((y+h)/2))
                points.append(center)

            points = np.array(points)

            if len(points) >= 2:
                d = dist.cdist(points, points, metric="euclidean")
                sec_min =  np.sort(d,axis = 1)[:,1]      
                for b,sec in zip(box,sec_min):
                    color = (0,0,255) if sec < 40 else (0,255,0)
                    is_violated = (is_violated or color == (0,0,255))
                    x,y,w,h = b
                    
                    cv2.rectangle(img,(x,y),(x+w,y+h),color,2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        post_data(idx,is_violated)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')