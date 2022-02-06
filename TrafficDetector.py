'''
Model is an abstract class for creating new types of Model classes.
By default, we've used a YoloV4 model with 256x256 resolution.

TrafficDetector is a wrapper object for video detection Model objects.
Models can be swapped here without the need to modify Camera objects.
'''
from abc import ABC, abstractmethod
from numpy import ndarray
from typing import List, Tuple
import random
import colorsys
import cv2

from tools.utils import CLASSES


class Model(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def traffic_detections(self,
                           frame: ndarray) -> Tuple[List[ndarray], ndarray]:
        '''
        Takes a camera frame and returns detected objects and a new frame
        with drawn bounding box overlays.

        Detected objects must be in the follow format:
        [boxes, scores, classes, num] where
        boxes: ndarray of shape(1, n, 4) is an array of bounding box coords
        scores: ndarray of shape(1, n) is an array of confidence scores
        classes: ndarray of shape(1, n) is an array of detected classes
        num: ndarray of shape(1, ) has value n
        '''
        raise NotImplementedError("Method not implemented")


class TrafficDetector():
    def __init__(self):
        ### Change this to your desired Model object ###
        from tools.video.models.yolov4.YoloV4 import YoloV4
        self.model: Model = YoloV4()
        ###

    def traffic_detections(self,
                           frame: ndarray) -> Tuple[List[object], ndarray]:
        objects, frame = self.model.traffic_detections(frame)
        frame = self.draw_bbox(frame, objects)
        return objects, frame

    def draw_bbox(self, image, bboxes, show_label=True):
        num_classes = len(CLASSES)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.)
                      for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)
        fontScale = 0.5

        out_boxes, out_scores, out_classes, num_boxes = bboxes
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(
                    out_classes[0][i]) not in CLASSES:
                continue
            coor = out_boxes[0][i]

            score = out_scores[0][i]
            class_ind = int(out_classes[0][i])
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
            if c1 == c2 or c1[0] == 0 or c1[1] == 0 or c2[0] == 0 or c2[1] == 0:
                continue
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            if show_label:
                bbox_mess = '%s: %.2f' % (CLASSES[class_ind], score)
                t_size = cv2.getTextSize(bbox_mess,
                                         0,
                                         fontScale,
                                         thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, c3, bbox_color, -1)  #filled

                cv2.putText(image,
                            bbox_mess, (c1[0], c1[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0),
                            bbox_thick // 2,
                            lineType=cv2.LINE_AA)
        return image
