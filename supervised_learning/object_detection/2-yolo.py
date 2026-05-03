#!/usr/bin/env python3
"""
This module implements `Yolo` class
that uses the Yolo v3 algorithm to perform object detection.
"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    A class that uses the Yolo v3 algorithm to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class.
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs from the Darknet model.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            box_conf = output[..., 4:5]
            box_class = output[..., 5:]

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            box_xy = sigmoid(t_xy)
            box_conf = sigmoid(box_conf)
            box_class = sigmoid(box_class)

            cx = np.arange(grid_w)
            cy = np.arange(grid_h)
            grid_x, grid_y = np.meshgrid(cx, cy)
            grid = np.stack((grid_x, grid_y), axis=-1)
            grid = grid.reshape((grid_h, grid_w, 1, 2))

            box_xy = (box_xy + grid) / [grid_w, grid_h]

            input_w = self.model.input.shape[1]
            input_h = self.model.input.shape[2]

            anchors = self.anchors[i]
            box_wh = (np.exp(t_wh) * anchors) / [input_w, input_h]

            x1y1 = box_xy - (box_wh / 2)
            x2y2 = box_xy + (box_wh / 2)

            box_coords = np.concatenate([x1y1, x2y2], axis=-1)

            box_coords[..., [0, 2]] *= image_w
            box_coords[..., [1, 3]] *= image_h

            boxes.append(box_coords)
            box_confidences.append(box_conf)
            box_class_probs.append(box_class)

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on their objectness score and class probability.

        Args:
            boxes:
                list of ndarrays of shape (grid_h, grid_w, anchors, 4)
            box_confidences:
                list of ndarrays of shape (grid_h, grid_w, anchors, 1)
            box_class_probs:
                list of ndarrays of shape (grid_h, grid_w, anchors, cls)

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            # Calculate box scores: confidence * class probabilities
            # (grid_h, grid_w, anchors, classes)
            scores = box_confidences[i] * box_class_probs[i]

            # Find the best class index and the score for that class
            box_class = np.argmax(scores, axis=-1)
            box_score = np.max(scores, axis=-1)

            # Create a mask for boxes that exceed the threshold
            mask = box_score >= self.class_t

            # Filter and add to lists
            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_class[mask])
            box_scores.append(box_score[mask])

        # Concatenate all outputs into single numpy arrays
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)
