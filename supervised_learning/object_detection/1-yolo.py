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

        Args:
            model_path (str):
                Path to where a Darknet Keras model is stored.
            classes_path (str):
                Path to where the list of class names is found.
            class_t (float):
                Box score threshold for the initial filtering step.
            nms_t (float):
                IOU threshold for non-max suppression.
            anchors (numpy.ndarray):
                Array of shape (outputs, anchor_boxes, 2)
                containing all of the anchor boxes.
        """
        # Load the Darknet Keras model
        self.model = K.models.load_model(model_path)

        # Load the class names from the provided file path
        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file.readlines()]

        # Store the thresholds and anchors as public instance attributes
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

            # 1. Extract raw predictions
            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            box_conf = output[..., 4:5]
            box_class = output[..., 5:]

            # 2. Apply sigmoid activation
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            box_xy = sigmoid(t_xy)
            box_conf = sigmoid(box_conf)
            box_class = sigmoid(box_class)

            # 3. Create Grid
            cx = np.arange(grid_w)
            cy = np.arange(grid_h)
            # Use default meshgrid, but ensure it matches the output shape
            grid_x, grid_y = np.meshgrid(cx, cy)
            grid = np.stack((grid_x, grid_y), axis=-1)
            grid = grid.reshape((grid_h, grid_w, 1, 2))

            # 4. Transform to relative (0 to 1) coordinates
            # Box center: (sigmoid(t_xy) + grid_offset) / grid_size
            box_xy = (box_xy + grid) / [grid_w, grid_h]

            # Box dimensions: (anchor * exp(t_wh)) / model_input_size
            # Note: The model input size is usually implicitly defined by
            # (grid_size * stride). Stride is usually 32, 16, 8.
            # Using the self.model.input.shape is the most reliable way:
            input_w = self.model.input.shape[1]
            input_h = self.model.input.shape[2]

            anchors = self.anchors[i]
            box_wh = (np.exp(t_wh) * anchors) / [input_w, input_h]

            # 5. Convert to [x1, y1, x2, y2]
            x1y1 = box_xy - (box_wh / 2)
            x2y2 = box_xy + (box_wh / 2)

            # Concatenate to (grid_h, grid_w, anchor_boxes, 4)
            # Order: x1, y1, x2, y2
            box_coords = np.concatenate([x1y1, x2y2], axis=-1)

            # 6. Final Scale to Image Size
            # Ensure we scale [x1, y1, x2, y2] by [w, h, w, h]
            box_coords[..., [0, 2]] *= image_w
            box_coords[..., [1, 3]] *= image_h

            boxes.append(box_coords)
            box_confidences.append(box_conf)
            box_class_probs.append(box_class)

        return (boxes, box_confidences, box_class_probs)
