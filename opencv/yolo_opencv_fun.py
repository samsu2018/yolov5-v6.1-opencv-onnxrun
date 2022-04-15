# -*- coding: utf-8 -*-
###################################################################
# Object detection - YOLOv5 - OpenCV
# Author : https://github.com/hpc203/yolov5-v6.1-opencv-onnxrun
# Modify : Sam Su
##################################################################
import cv2
import argparse
import numpy as np
import time

def get_obj(img, modelpath, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default=img, help="image path")
    parser.add_argument('--modelpath', type=str, default=modelpath)
    parser.add_argument('--confThreshold', default=confThreshold, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=nmsThreshold, type=float, help='nms iou thresh') # low are best
    parser.add_argument('--objThreshold', default=objThreshold, type=float, help='object confidence')
    args = parser.parse_args()

    yolonet = yolov5(args.modelpath, confThreshold=args.confThreshold,
                     nmsThreshold=args.nmsThreshold, objThreshold=args.objThreshold)

    srcimg = cv2.imread(args.imgpath)
    nms_dets, frame = yolonet.detect(srcimg)
    return nms_dets, frame

def object_ratio(frame, x1, y1, x2, y2):
    '''Input images and coordimates, reture object ratio'''
    main_area_th = 0.8  # 0.7 is for main content
    h, w, d = frame.shape
    if h>w:
        min_long = int(w // 2 * main_area_th) # 0.7 is for main content
    else:
        min_long = int(h // 2 * main_area_th)
    
    # square size
    cneter_x, center_y = (w // 2, h // 2) # 找到圖片中心
    new_x1 = cneter_x - min_long
    new_y1 = center_y - min_long
    new_x2 = cneter_x + min_long
    new_y2 = center_y + min_long
    # frame = cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), thickness=4)

    # corrected obj's bbox
    x1 = new_x1 if x1 < new_x1 else x1
    y1 = new_y1 if y1 < new_y1 else y1
    x2 = new_x2 if x2 > new_x2 else x2
    y2 = new_y2 if y2 > new_y2 else y2
    # frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=4)
    
    # calculate the object ration
    center_area = (new_x2 - new_x1) * (new_y2 - new_y1)
    object_area = (x2 - x1) * (y2 - y1)
    obj_ratio = round(float(object_area)/float(center_area),2)
    return obj_ratio

class yolov5():
    def __init__(self, modelpath, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        with open('class.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.num_classes = len(self.classes)
        if modelpath.endswith('6.onnx'):
            self.inpHeight, self.inpWidth = 1280, 1280
            anchors = [[19, 27, 44, 40, 38, 94], [96, 68, 86, 152, 180, 137], [140, 301, 303, 264, 238, 542],
                       [436, 615, 739, 380, 925, 792]]
            self.stride = np.array([8., 16., 32., 64.])
        else:
            self.inpHeight, self.inpWidth = 640, 640
            anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
            self.stride = np.array([8., 16., 32.])
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [np.zeros(1)] * self.nl
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.net = cv2.dnn.readNet(modelpath)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
        self._inputNames = ''

    def resize_image(self, srcimg, keep_ratio=True, dynamic=False):
        top, left, newh, neww = 0, 0, self.inpWidth, self.inpHeight
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                if not dynamic:
                    left = int((self.inpWidth - neww) * 0.5)
                    img = cv2.copyMakeBorder(img, 0, 0, left, self.inpWidth - neww - left, cv2.BORDER_CONSTANT,
                                             value=(114, 114, 114))  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                if not dynamic:
                    top = int((self.inpHeight - newh) * 0.5)
                    img = cv2.copyMakeBorder(img, top, self.inpHeight - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                             value=(114, 114, 114))
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    def postprocess(self, frame, outs, padsize=None):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        newh, neww, padh, padw = padsize
        ratioh, ratiow = frameHeight / newh, frameWidth / neww
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.

        confidences = []
        boxes = []
        classIds = []
        nms_dets = []
        for detection in outs:
            if detection[4] > self.objThreshold:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId] * detection[4]
                if confidence > self.confThreshold:
                    center_x = int((detection[0] - padw) * ratiow)
                    center_y = int((detection[1] - padh) * ratioh)
                    width = int(detection[2] * ratiow)
                    height = int(detection[3] * ratioh)
                    left = int(center_x - width * 0.5)
                    top = int(center_y - height * 0.5)

                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
                    classIds.append(classId)
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold).flatten()
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = left + box[2] # Here  coordinates calculates differently with v60
            height = top + box[3] # Here  coordinates calculates differently with v60
            # draw bbox
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, width, height)
            # store the detection results
            conf = round(confidences[i],4)
            nms_classId = classIds[i]
            label = str(self.classes[nms_classId])
            obj_ratio = object_ratio(frame, left, top, width, height)
            # hpc203 calculates coordinates differently with v60, 'left + width, top + height'
            nms_dets.append([label, conf, left, top, width, height, classId, obj_ratio])
        return nms_dets, frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame

    def detect(self, srcimg):
        img, newh, neww, padh, padw = self.resize_image(srcimg)
        blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255.0, swapRB=True)
        # blob = cv2.dnn.blobFromImage(self.preprocess(img))
        # Sets the input to the network
        self.net.setInput(blob, self._inputNames)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0].squeeze(axis=0)

        # inference output
        row_ind = 0
        for i in range(self.nl):
            h, w = int(self.inpHeight / self.stride[i]), int(self.inpWidth / self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h, w):
                self.grid[i] = self._make_grid(w, h)

            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
                self.grid[i], (self.na, 1))) * int(self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
                self.anchor_grid[i], h * w, axis=0)
            row_ind += length
        nms_dets, frame = self.postprocess(srcimg, outs, padsize=(newh, neww, padh, padw))
        return nms_dets, frame

# =============================================================================
# The following main functions are used for standalong testing
# =============================================================================
if __name__ == "__main__":
    imgpath = 'images/bus.jpg'
    modelpath = '../../../yolov5_6.1_onnx/yolov5n6.onnx'
    tStart = time.time()
    dets, frame = get_obj(imgpath, modelpath)
    print(dets)
    cv2.imwrite('output.jpg', frame)
    print('Spend time:{}'.format(time.time()-tStart))