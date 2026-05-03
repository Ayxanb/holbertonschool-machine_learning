#!/usr/bin/env python3
"""
This module implements `Yolo` class
that uses the Yolo v3 algorithm to perform object detection.
"""

import tensorflow.keras as K
import numpy as np
import cv2
import os


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
            grid = np.stack(
                (grid_x, grid_y),
                axis=-1).reshape((grid_h, grid_w, 1, 2))

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
        Filters boxes based on score threshold.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            box_class = np.argmax(scores, axis=-1)
            box_score = np.max(scores, axis=-1)

            mask = box_score >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_class[mask])
            box_scores.append(box_score[mask])

        return (np.concatenate(filtered_boxes, axis=0),
                np.concatenate(box_classes, axis=0),
                np.concatenate(box_scores, axis=0))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Performs Non-Max Suppression on the filtered boxes.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in np.unique(box_classes):
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            idxs = np.argsort(cls_scores)[::-1]

            keep = []
            while len(idxs) > 0:
                current = idxs[0]
                keep.append(current)

                if len(idxs) == 1:
                    break

                rect1 = cls_boxes[current]
                rest = cls_boxes[idxs[1:]]

                x1 = np.maximum(rect1[0], rest[:, 0])
                y1 = np.maximum(rect1[1], rest[:, 1])
                x2 = np.minimum(rect1[2], rest[:, 2])
                y2 = np.minimum(rect1[3], rest[:, 3])

                intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
                area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
                area_rest = (rest[:, 2] - rest[:, 0]) * \
                            (rest[:, 3] - rest[:, 1])

                iou = intersection / (area1 + area_rest - intersection)
                idxs = idxs[1:][iou < self.nms_t]

            box_predictions.append(cls_boxes[keep])
            predicted_box_classes.append(np.full(len(keep), cls))
            predicted_box_scores.append(cls_scores[keep])

        return (np.concatenate(box_predictions, axis=0),
                np.concatenate(predicted_box_classes, axis=0),
                np.concatenate(predicted_box_scores, axis=0))

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a specified folder alphabetically.
        """
        images = []
        image_paths = []
        filenames = sorted(os.listdir(folder_path))

        for filename in filenames:
            path = os.path.join(folder_path, filename)
            image = cv2.imread(path)
            if image is not None:
                images.append(image)
                image_paths.append(path)

        return (images, image_paths)

    def preprocess_images(self, images):
        """
        Preprocesses images for model input.
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append(img.shape[:2])
            resized = cv2.resize(img, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)
            rescaled = resized / 255.0
            pimages.append(rescaled)

        return (np.array(pimages), np.array(image_shapes))

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names, and scores.
        """
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]
            label = "{} {:.2f}".format(class_name, score)

            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            save_path = os.path.join('detections', file_name)
            cv2.imwrite(save_path, image)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Performs full object detection pipeline on folder of images.

        Args:
            folder_path (str): path to the folder holding images

        Returns:
            tuple: (predictions, image_paths)
        """
        # 1. Load images and paths
        images, image_paths = self.load_images(folder_path)

        # 2. Preprocess images
        pimages, image_shapes = self.preprocess_images(images)

        # 3. Model Inference
        # outputs is a list of arrays from different detection scales
        outputs = self.model.predict(pimages)

        predictions = []

        for i in range(len(images)):
            # Extract the raw output for the current image i across all scales
            image_outputs = [output[i] for output in outputs]

            # 4. Process raw outputs into coordinates/scores
            boxes, conf, probs = self.process_outputs(image_outputs,
                                                      image_shapes[i])

            # 5. Filter by threshold
            f_boxes, f_classes, f_scores = self.filter_boxes(
                                            boxes, conf, probs)

            # 6. Apply Non-Max Suppression
            n_boxes, n_classes, n_scores = self.non_max_suppression(
                f_boxes, f_classes, f_scores)

            # 7. Display/Save using show_boxes
            # Name window after the filename (basename)
            filename = os.path.basename(image_paths[i])
            self.show_boxes(images[i], n_boxes, n_classes, n_scores, filename)

            # Collect results
            predictions.append((n_boxes, n_classes, n_scores))

        return (predictions, image_paths)
