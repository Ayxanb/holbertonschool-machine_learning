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

        Args:
            outputs (list):
                A list of numpy.ndarrays containing the predictions.
            image_size (numpy.ndarray):
                The image's original size [image_height, image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size
        model_h = int(self.model.input.shape[1])
        model_w = int(self.model.input.shape[2])

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # Slicing the output
            # to extract bounding box coordinates, confidences
            # and class probabilities
            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            box_conf = output[..., 4:5]
            box_class = output[..., 5:]

            # Apply sigmoid function to t_xy, box_conf, and box_class
            box_xy = 1 / (1 + np.exp(-t_xy))
            box_conf = 1 / (1 + np.exp(-box_conf))
            box_class = 1 / (1 + np.exp(-box_class))

            # Create a grid of offsets for each cell
            cx = np.arange(grid_w)
            cy = np.arange(grid_h)
            cx, cy = np.meshgrid(cx, cy)

            # Stack and expand dimensions to match (grid_h, grid_w, 1, 2)
            grid = np.expand_dims(np.stack((cx, cy), axis=-1), axis=2)

            # Add the grid offset to box_xy, normalize by the grid dimensions
            box_xy = (box_xy + grid) / [grid_w, grid_h]

            # Calculate the box dimensions (width and height)
            # normalize by model dimensions
            anchors = self.anchors[i]
            box_wh = (np.exp(t_wh) * anchors) / [model_w, model_h]

            # Convert (center_x, center_y, width, height) to (x1, y1, x2, y2)
            x1 = box_xy[..., 0:1] - (box_wh[..., 0:1] / 2)
            y1 = box_xy[..., 1:2] - (box_wh[..., 1:2] / 2)
            x2 = box_xy[..., 0:1] + (box_wh[..., 0:1] / 2)
            y2 = box_xy[..., 1:2] + (box_wh[..., 1:2] / 2)

            # Scale the bounding box coordinates
            # back to the original image size
            x1 *= image_w
            y1 *= image_h
            x2 *= image_w
            y2 *= image_h

            # Concatenate coordinates to
            # shape (grid_h, grid_w, anchor_boxes, 4)
            box_coords = np.concatenate([x1, y1, x2, y2], axis=-1)

            boxes.append(box_coords)
            box_confidences.append(box_conf)
            box_class_probs.append(box_class)

        return (boxes, box_confidences, box_class_probs)
