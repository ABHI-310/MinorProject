"""
Data Augmentation for PCB Defect Detection
Applies geometric and photometric augmentations with precise YOLO bounding box transforms.
"""

import random
import cv2
import numpy as np


class PCBAugmentor:
    def __init__(
        self,
        rotation_range=(-15, 15),
        hflip_prob=0.5,
        brightness_range=(-0.2, 0.2),
        gaussian_noise_sigma=5.0,
        zoom_range=(0.8, 1.2),
    ):
        self.rotation_range = rotation_range
        self.hflip_prob = hflip_prob
        self.brightness_range = brightness_range
        self.gaussian_noise_sigma = gaussian_noise_sigma
        self.zoom_range = zoom_range

    def random_horizontal_flip(self, image: np.ndarray, bboxes: list):
        """
        Flip image horizontally and adjust YOLO bboxes [cls, x_c, y_c, w, h]
        """
        if random.random() < self.hflip_prob:
            image = cv2.flip(image, 1)
            new_bboxes = []
            for box in bboxes:
                cls_id, x_c, y_c, w, h = box
                new_x_c = 1.0 - x_c
                new_bboxes.append([cls_id, new_x_c, y_c, w, h])
            return image, new_bboxes
        return image, bboxes

    def adjust_brightness(self, image: np.ndarray):
        """
        Adjust brightness by a random factor within brightness_range
        """
        factor = 1.0 + random.uniform(*self.brightness_range)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def add_gaussian_noise(self, image: np.ndarray):
        """
        Add zero-mean Gaussian noise with specified standard deviation
        """
        if self.gaussian_noise_sigma <= 0:
            return image
        noise = np.random.normal(0, self.gaussian_noise_sigma, image.shape)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    def random_zoom(self, image: np.ndarray, bboxes: list):
        """
        Zoom in or out while clipping to original canvas and adapting bboxes
        """
        scale = random.uniform(*self.zoom_range)
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        canvas = np.zeros_like(image)
        new_bboxes = []

        if scale >= 1.0:
            # Crop center
            crop_x = (new_w - w) // 2
            crop_y = (new_h - h) // 2
            canvas = resized[crop_y : crop_y + h, crop_x : crop_x + w]
            for box in bboxes:
                cls_id, x_c, y_c, bw, bh = box
                # Convert normalized to resized image pixels
                abs_xc = x_c * new_w - crop_x
                abs_yc = y_c * new_h - crop_y
                abs_w = bw * new_w
                abs_h = bh * new_h

                norm_xc = abs_xc / w
                norm_yc = abs_yc / h
                norm_w = abs_w / w
                norm_h = abs_h / h

                if 0 < norm_xc < 1 and 0 < norm_yc < 1:
                    new_bboxes.append([
                        cls_id,
                        float(np.clip(norm_xc, 0.01, 0.99)),
                        float(np.clip(norm_yc, 0.01, 0.99)),
                        float(np.clip(norm_w, 0.01, 0.99)),
                        float(np.clip(norm_h, 0.01, 0.99)),
                    ])
        else:
            # Pad on canvas
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
            for box in bboxes:
                cls_id, x_c, y_c, bw, bh = box
                abs_xc = x_c * new_w + pad_x
                abs_yc = y_c * new_h + pad_y
                abs_w = bw * new_w
                abs_h = bh * new_h
                new_bboxes.append([
                    cls_id,
                    float(abs_xc / w),
                    float(abs_yc / h),
                    float(abs_w / w),
                    float(abs_h / h),
                ])

        return canvas, new_bboxes

    def random_rotation(self, image: np.ndarray, bboxes: list):
        """
        Rotate image by ±15 degrees and transform bounding boxes
        """
        angle = random.uniform(*self.rotation_range)
        if abs(angle) < 0.5:
            return image, bboxes

        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114)
        )

        new_bboxes = []
        for box in bboxes:
            cls_id, x_c, y_c, bw, bh = box
            # 4 corner points in normalized -> pixels
            x1 = (x_c - bw / 2.0) * w
            y1 = (y_c - bh / 2.0) * h
            x2 = (x_c + bw / 2.0) * w
            y2 = (y_c + bh / 2.0) * h

            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            # Add ones for affine transform
            ones = np.ones(shape=(len(pts), 1))
            pts_ones = np.hstack([pts, ones])
            transformed_pts = matrix.dot(pts_ones.T).T

            rx1 = np.min(transformed_pts[:, 0])
            ry1 = np.min(transformed_pts[:, 1])
            rx2 = np.max(transformed_pts[:, 0])
            ry2 = np.max(transformed_pts[:, 1])

            # Clip within boundaries
            rx1 = max(0, min(w, rx1))
            ry1 = max(0, min(h, ry1))
            rx2 = max(0, min(w, rx2))
            ry2 = max(0, min(h, ry2))

            new_w_box = (rx2 - rx1) / w
            new_h_box = (ry2 - ry1) / h
            new_xc = (rx1 + rx2) / (2.0 * w)
            new_yc = (ry1 + ry2) / (2.0 * h)

            if new_w_box > 0.005 and new_h_box > 0.005:
                new_bboxes.append([cls_id, float(new_xc), float(new_yc), float(new_w_box), float(new_h_box)])

        return rotated, new_bboxes

    def augment(self, image: np.ndarray, bboxes: list):
        """
        Run full augmentation pipeline in sequence
        """
        img, boxes = self.random_horizontal_flip(image, bboxes)
        img, boxes = self.random_rotation(img, boxes)
        img, boxes = self.random_zoom(img, boxes)
        img = self.adjust_brightness(img)
        img = self.add_gaussian_noise(img)
        return img, boxes

