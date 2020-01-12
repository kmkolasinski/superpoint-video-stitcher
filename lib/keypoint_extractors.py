from pathlib import Path
from typing import List, Tuple, Optional

import cv2 as cv
import numpy as np
import tensorflow as tf
from dataclasses import dataclass


@dataclass(frozen=True)
class Keypoints:

    keypoints: np.ndarray  # [num_points, 2] in (x, y) format
    probs: np.ndarray  # [num_points]
    features: np.ndarray  # [num_points, 256]

    def get_cv_keypoints(self, radius: float = 5) -> List[cv.KeyPoint]:
        return [cv.KeyPoint(pt[0], pt[1], radius) for pt in self.keypoints]


class SuperPointExtractor:
    def __init__(
        self,
        export_dir: Path,
        width: Optional[int],
        height: Optional[int],
        k_top_keypoints: int = 1000,
    ):
        """
        Extracts SuperPoint keypoints with descriptions.
        Usage:

            ex = SuperPointExtractor(path, 200, 200)
            keypoints = ex.extract(image)

        Args:
            export_dir: exported SavedModel path
            width: image resize width. Can be None if height is defined.
            height: image resize height. Can be None if width is defined.
            k_top_keypoints:
        """

        assert (
            width is not None or height is not None
        ), "Only one of [width, height] can be None"

        self.predict_fn = tf.contrib.predictor.from_saved_model(
            str(export_dir), signature_def_key="serving_default"
        )
        self.k_top_keypoints = k_top_keypoints
        self.width = width
        self.height = height

    def extract(self, image: np.ndarray) -> Keypoints:
        """

        Args:
            image: uint8 image of shape [height, width, 3]

        Returns:
            keypoints
        """
        src_height, src_width = image.shape[:2]
        if self.width is not None and self.height is not None:
            sx, sy = src_width / self.width, src_height / self.height
            resize = (self.width, self.height)
        elif self.width is None:
            sy = src_height / self.height
            sx = sy
            resize = (int(src_width / sx), self.height)
        else:
            sx = src_width / self.width
            sy = sx
            resize = (self.width, int(src_height / sy))

        img_preprocessed = preprocess_image(image, resize)

        predictions = self.predict_fn(dict(image=img_preprocessed))
        keypoints_map = np.squeeze(predictions["prob_nms"])
        descriptor_map = np.squeeze(predictions["descriptors"])
        keypoints, probs, features = extract_superpoint_keypoints_and_descriptors(
            keypoints_map, descriptor_map, self.k_top_keypoints
        )
        # swap (y, x) => (x, y) and rescale back points to source image
        keypoints = keypoints[:, (1, 0)] * np.array([[sx, sy]])

        return Keypoints(keypoints, probs, features)


def preprocess_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Prepare input image to form accepted SuperPoint predictor

    Args:
        img: an uint8 image of shape [height', width', 3]
        size: target image resize size (width, height)
    Returns:
        a processed gray scale image of shape [1, height, width, 1] and dtype=float32

    """

    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.resize(img, size, interpolation=cv.INTER_AREA)
    img = np.expand_dims(img, -1)
    img = img.astype(np.float32)
    img_preprocessed = np.expand_dims(img, 0)
    return img_preprocessed


def extract_superpoint_keypoints_and_descriptors(
    keypoint_map: np.ndarray, descriptor_map: np.ndarray, keep_k_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :3]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :][::-1, :]

    # Extract keypoints
    keypoints = np.where(keypoint_map >= 1e-5)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)
    keypoints = select_k_best(keypoints, keep_k_points)
    keypoints_probs = keypoints[:, 2]
    int_keypoints = keypoints[:, :2].astype(int)

    # Get descriptors for keypoints
    desc = descriptor_map[int_keypoints[:, 0], int_keypoints[:, 1]]

    # Convert from just pts to cv2.KeyPoints
    keypoints = keypoints[:, :2].astype(np.float32)
    keypoints_probs = keypoints_probs.astype(np.float32)
    return keypoints, keypoints_probs, desc.astype(np.float32)
