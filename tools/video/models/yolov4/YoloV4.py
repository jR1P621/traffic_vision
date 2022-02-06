import cv2
import tensorflow as tf
import numpy as np

from tools.utils import CONFIG
from TrafficDetector import Model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

SIZE = 256
TFLITE_PATH = 'tools/video/models/yolov4/yolov4-256.tflite'


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def filter_boxes(box_xywh,
                 scores,
                 score_threshold=0.40,
                 input_shape=tf.constant([416, 416])):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(
        class_boxes, [tf.shape(scores)[0], -1,
                      tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf,
                           [tf.shape(scores)[0], -1,
                            tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat(
        [
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ],
        axis=-1)
    return (boxes, pred_conf)


class YoloV4(Model):
    def __init__(self, iou_threshold=0.45, score_threshold=0.40) -> None:
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def traffic_detections(
        self,
        frame,
    ):
        image_h, image_w, _ = frame.shape
        input_frame = cv2.resize(frame, (SIZE, SIZE))
        input_frame = input_frame.astype(np.float32)
        input_tensor = tf.expand_dims(input_frame, axis=0) / 255.0

        self.interpreter.set_tensor(self.input_details[0]['index'],
                                    input_tensor)
        self.interpreter.invoke()
        pred = [
            self.interpreter.get_tensor(self.output_details[i]['index'])
            for i in range(len(self.output_details))
        ]
        boxes, pred_conf = filter_boxes(pred[0],
                                        pred[1],
                                        score_threshold=self.score_threshold,
                                        input_shape=tf.constant([SIZE, SIZE]))
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf,
                (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold)
        pred_bbox = [
            boxes.numpy(),
            scores.numpy(),
            classes.numpy(),
            valid_detections.numpy()
        ]
        out_boxes, _, out_classes, num_boxes = pred_bbox
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(
                    out_classes[0][i]) not in CONFIG['classes']:
                continue
            coor = out_boxes[0][i]
            coor[0] = int(coor[0] * image_h)
            coor[2] = int(coor[2] * image_h)
            coor[1] = int(coor[1] * image_w)
            coor[3] = int(coor[3] * image_w)
        return pred_bbox, frame