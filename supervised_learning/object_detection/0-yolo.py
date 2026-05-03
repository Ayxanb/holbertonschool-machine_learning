#!/usr/bin/env python3
"""
This module implements `Yolo` class
that uses the Yolo v3 algorithm to perform object detection.
"""

import tensorflow.keras as K


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
