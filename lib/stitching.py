from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from tqdm.auto import tqdm
from lib.io import ImagesStreamer
from lib.keypoint_extractors import SuperPointExtractor

import os

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
CV_CONFIDENCE = 0.999


class VideoStitcher:
    def __init__(
        self, extractor_crop_size: Tuple[Optional[int], Optional[int]], nn_thresh=0.7
    ):
        """
        A stitcher which uses SuperPointExtractor to stitch sequence of
        images.

        Args:
            extractor_crop_size:
            nn_thresh: nearest neighbour distance threshold used for matching.
        """
        self.extractor = SuperPointExtractor(
            dir_path.parent / "models/sp_v5/",
            width=extractor_crop_size[0],
            height=extractor_crop_size[1],
        )
        self.nn_thresh = nn_thresh

    def stitch_naive_left_right_sequence(self, streamer: ImagesStreamer) -> np.ndarray:
        """
        Stitching which assumes left-to-right planar motion. Naively cut slices
        from image and then glue them together.
        Args:
            streamer:

        Returns:
            stitched image
        """
        scan_images, scan_offsets = naive_panorama_slices_from_left_right_scan(
            streamer, self.extractor, self.nn_thresh
        )
        return stitch_left_right_panorama_slices_list(scan_images, scan_offsets)

    def stitch_left_right_sequence(
        self, streamer: ImagesStreamer, use_homography: bool = False
    ) -> np.ndarray:
        """
        Stitching which assumes left-to-right planar motion. This function
        will apply additional image correction
        Args:
            streamer:
            use_homography: match images by homography relationship. If
                False Affine transform is used.
        Returns:
            stitched image
        """
        return stitch_left_right_scan(
            streamer, self.extractor, self.nn_thresh, use_homography=use_homography
        )

    def stitch_sequence(
        self,
        streamer: ImagesStreamer,
        size: Tuple[Optional[int], Optional[int]],
        use_homography: bool = False,
    ):
        """
        Stitch sequence of images, with any direction of camera motion.

        Usage:
            a) undefined motion direction:
                image = st.stitch_scan(streamer, (250, 250))
            b) horizontal direction (left <=> right camera movement):
                image = st.stitch_scan(streamer, (None, 250))
            c) vertical direction (top <=> bottom camera movement):
                image = st.stitch_scan(streamer, (250, None))
        Args:
            streamer:
            size: area of size (height, width) from the center of the image
                which will be used in stitching process.
            use_homography: match images by homography relationship. If
                False Affine transform is used.

        Returns:
            stitched image
        """
        return stitch_scan(
            streamer, self.extractor, center_size=size, use_homography=use_homography
        )


def match(
    features1: np.ndarray, features2: np.ndarray, nn_thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match features between images.
    Args:
        features1: array of shape [num_points, feature_size]
        features2: array of shape [num_points, feature_size]
        nn_thresh: nearest neighbour distance threshold
    """

    assert features1.shape[1] == features2.shape[1]
    if features1.shape[0] == 0 or features2.shape[0] == 0:
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,))

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(queryDescriptors=features1, trainDescriptors=features2)
    matches = [m for m in matches if m.distance < nn_thresh]

    features1_idx = np.array([m.queryIdx for m in matches])
    features2_idx = np.array([m.trainIdx for m in matches])
    distances = np.array([m.distance for m in matches])

    return features1_idx, features2_idx, distances


def naive_panorama_slices_from_left_right_scan(
    streamer: ImagesStreamer, extractor: SuperPointExtractor, nn_thresh: float = 0.7
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:

    image_slices = []
    image_slices_xy_offsets = []
    last_keypoints = None
    last_image = None

    progress_bar = tqdm(total=streamer.num_images)
    while True:
        progress_bar.update(1)
        img, status = streamer.next_frame()
        if img is not None:
            last_image = img
        if status is False:
            break

        height, width = img.shape[:2]
        keypoints = extractor.extract(img)

        if last_keypoints is not None:
            last_idx, idx, _ = match(
                last_keypoints.features, keypoints.features, nn_thresh
            )
            prev_points = last_keypoints.keypoints[last_idx, :2]
            curr_points = keypoints.keypoints[idx, :2]
            ox, oy = np.mean(curr_points - prev_points, 0).astype(np.int64)
            image_slices_xy_offsets.append((ox, oy))
            image_slice = img[:, width // 2 + min(ox, 0) : width // 2, :]
            image_slices.append(image_slice)
        else:
            # first frame
            image_slice = img[:, : width // 2, :]
            image_slices.append(image_slice)
            image_slices_xy_offsets.append((width // 2, 0))

        last_keypoints = keypoints

    progress_bar.close()

    # last frame
    height, width = last_image.shape[:2]
    image_slice = last_image[:, width // 2 :, :]

    image_slices.append(image_slice)
    image_slices_xy_offsets.append((width // 2, 0))
    return image_slices, image_slices_xy_offsets


def stitch_left_right_panorama_slices_list(
    image_slices: List[np.ndarray], image_slices_offsets: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Stitch slices returned by naive_panorama_slices_from_left_right_scan
    function

    Args:
        image_slices:
        image_slices_offsets:

    Returns:
        stitched image
    """
    new_tiles = []
    total_y_offset = 0

    IMAGE_OFFSET = 1000
    SEARCH_SIZE = 50

    h, w = image_slices[0].shape[:2]
    min_y, max_y = IMAGE_OFFSET, 0

    for scan, (ox, oy) in zip(image_slices, image_slices_offsets):
        total_y_offset += -oy / 4  # initial offset
        hi, wi = scan.shape[:2]
        if wi == 0:
            continue

        empty_scan = np.zeros([h + IMAGE_OFFSET, wi, 3], dtype=scan.dtype)
        INC_BASE = IMAGE_OFFSET // 2 + int(total_y_offset)
        y = 0
        if len(new_tiles) <= 2:
            empty_scan[INC_BASE : INC_BASE + hi, :, :] = scan
        else:
            last_y_scan_0 = new_tiles[-1][:, -1, :].mean(-1)
            last_y_scan_1 = new_tiles[-2][:, -1, :].mean(-1)
            # match slices by measuring difference between pixels
            # in two consecutive slices
            align_scores = []
            for y in range(-SEARCH_SIZE, SEARCH_SIZE):
                opt_y_scan = np.zeros_like(last_y_scan_0)
                opt_y_scan[INC_BASE + y : INC_BASE + y + hi] = scan[:, 0, :].mean(-1)
                score = np.sum(abs(opt_y_scan - last_y_scan_0))
                score = score + np.sum(abs(opt_y_scan - last_y_scan_1))
                align_scores.append((score, y))

            y = sorted(align_scores)[0][1]
            empty_scan[INC_BASE + y : INC_BASE + y + hi, :, :] = scan

        min_y = min(INC_BASE + y, min_y)
        max_y = max(INC_BASE + y + hi, max_y)
        new_tiles.append(empty_scan)

    final_image = np.concatenate(new_tiles, 1)
    return final_image[min_y:max_y]


def stitch_left_right_scan(
    streamer: ImagesStreamer,
    fe: SuperPointExtractor,
    nn_thresh: float = 0.7,
    use_homography: bool = False,
) -> np.ndarray:

    last_keypoints = None
    transform_matrix = np.eye(3)
    progress_bar = tqdm(total=streamer.num_images)
    reconstructed_img = None
    while True:
        img, status = streamer.next_frame()
        progress_bar.update(1)
        if not status:
            break

        H, W = img.shape[:2]
        keypoints = fe.extract(img)

        if last_keypoints is not None:

            matrix, (dx, dy) = find_transformation_matrix(last_keypoints,
                                                          keypoints, nn_thresh,
                                                          use_homography)

            transform_matrix = transform_matrix @ matrix

            curr_img_mask = np.zeros_like(img)
            dx = abs(int(dx))
            if dx == 0:
                continue

            curr_img_mask[1:-1, W // 2 - dx + 1:] = 255
            mask, _, _ = warp_image(
                curr_img_mask, transform_matrix, reconstructed_img,
                cv2.INTER_NEAREST
            )

            curr_img_mask[:, W // 2 - dx :] = img[:, W // 2 - dx :]

            new_img, transform_matrix, reconstructed_img = warp_image(
                curr_img_mask, transform_matrix, reconstructed_img,
                cv2.INTER_LINEAR
            )

            mask = 1 * (mask > 0).astype(np.uint8)
            reconstructed_img = reconstructed_img * (1 - mask) + mask * new_img
        else:
            reconstructed_img = img.copy()

        last_keypoints = keypoints

    reconstructed_img, transform_matrix = correct_left_to_right_panorama(
        reconstructed_img, transform_matrix, width=W, height=H
    )

    progress_bar.close()
    return reconstructed_img


def stitch_scan(
    streamer: ImagesStreamer,
    fe: SuperPointExtractor,
    center_size=(200, 200),
    nn_thresh=0.7,
    use_homography: bool = False,
):

    last_keypoints = None
    transform_matrix = np.eye(3)
    progress_bar = tqdm(total=streamer.num_images)
    reconstructed_img = None
    center_size = list(center_size)

    while True:
        img, status = streamer.next_frame()
        progress_bar.update(1)
        if not status:
            break

        H, W = img.shape[:2]
        if center_size[0] is None:
            center_size[0] = W
        if center_size[1] is None:
            center_size[1] = H

        xs, ys = center_size[0] // 2, center_size[1] // 2
        ymin, ymax, xmin, xmax = H // 2 - ys, H // 2 + ys, W // 2 - xs, W // 2 + xs
        keypoints = fe.extract(img)

        if last_keypoints is not None:
            curr_img_mask = np.zeros_like(img)
            curr_img_mask[ymin + 2 : ymax - 2, xmin + 2 : xmax - 2] = 255
            curr_img = np.zeros_like(img)
            curr_img[ymin:ymax, xmin:xmax] = img[ymin:ymax, xmin:xmax]
            reconstructed_img, transform_matrix = match_and_warp(
                last_keypoints,
                keypoints,
                nn_thresh,
                use_homography,
                transform_matrix,
                curr_img,
                curr_img_mask,
                reconstructed_img,
            )
        else:
            reconstructed_img = img.copy()

        last_keypoints = keypoints
    progress_bar.close()
    return crop_largest_region(reconstructed_img)


def crop_largest_region(image: np.ndarray):
    ret, threshed_img = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                      0, 255, cv2.THRESH_BINARY)
    mask, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(c) for c in contours]
    largest_bbox = list(sorted(bboxes, key=lambda bbox: bbox[-1] * bbox[-2]))[
        -1]
    x1, y1, w, h = largest_bbox
    x2, y2 = x1 + w, y1 + h
    return image[y1:y2, x1:x2]


def find_transformation_matrix(last_keypoints, keypoints, nn_thresh, use_homography):
    last_idx, idx, _ = match(
        last_keypoints.features, keypoints.features, nn_thresh
    )
    prev_points = last_keypoints.keypoints[last_idx, :2]
    curr_points = keypoints.keypoints[idx, :2]

    if use_homography:
        matrix, _ = cv2.findHomography(
            curr_points, prev_points, confidence=CV_CONFIDENCE
        )
    else:
        matrix, _ = cv2.estimateAffine2D(
            curr_points, prev_points, confidence=CV_CONFIDENCE
        )
        matrix = np.vstack([matrix, np.array([0, 0, 1])])

    dx, dy = np.median(curr_points - prev_points, 0)
    return matrix, (dx, dy)


def match_and_warp(
    last_keypoints,
    keypoints,
    nn_thresh,
    use_homography,
    transform_matrix,
    curr_img,
    curr_img_mask,
    reconstructed_img,
):

    matrix, (dx, dy) = find_transformation_matrix(last_keypoints, keypoints, nn_thresh, use_homography)
    transform_matrix = transform_matrix @ matrix
    dx = abs(int(dx))

    if abs(dx) + abs(dy) == 0:
        return reconstructed_img, transform_matrix

    warped_mask, *_ = warp_image(curr_img_mask, transform_matrix, reconstructed_img)

    new_img, transform_matrix, reconstructed_img = warp_image(
        curr_img, transform_matrix, reconstructed_img
    )

    warped_mask = 1 * (warped_mask > 0).astype(np.uint8)
    reconstructed_img = reconstructed_img * (1 - warped_mask) + warped_mask * new_img
    return reconstructed_img, transform_matrix


def correct_left_to_right_panorama(
    reconstructed_img: np.ndarray, transform_matrix: np.ndarray, width: int, height: int
):

    # align image
    recon_height, recon_width = reconstructed_img.shape[:2]
    matrix = transform_matrix.copy()

    crop_curr_points = np.array([[width, 0.0], [width, height]]).astype(np.float32)

    crop_prev_points = cv2.perspectiveTransform(
        crop_curr_points.reshape([1, -1, 2]), matrix
    )
    crop_prev_points = crop_prev_points.squeeze()

    source_corners = np.array(
        [[0.0, 0.0], [0.0, recon_height], crop_prev_points[0], crop_prev_points[1]]
    ).astype(np.float32)

    target_corners = np.array(
        [
            [0.0, 0.0],
            [0.0, recon_height],
            [recon_width, crop_prev_points[0][1]],
            [recon_width, crop_prev_points[1][1]],
        ]
    ).astype(np.float32)

    align_matrix, _ = cv2.findHomography(
        source_corners, target_corners, confidence=CV_CONFIDENCE
    )

    reconstructed_img, align_matrix, _ = warp_image(
        reconstructed_img, align_matrix, None
    )
    reconstructed_img = reconstructed_img[:, :recon_width]

    transform_matrix = transform_matrix @ align_matrix
    return reconstructed_img, transform_matrix


def get_image_corner_points(images: np.ndarray) -> np.ndarray:
    max_height, max_width = np.max([img.shape[:2] for img in [images]], axis=0)
    return np.array(
        [[0, 0], [0, max_height], [max_width, 0], [max_width, max_height]]
    ).astype(np.float32)


def get_transformation_bounding_box(
    homography: np.ndarray, image: np.ndarray
) -> np.ndarray:
    image_corners = get_image_corner_points(image)
    points = np.expand_dims(image_corners, axis=0)
    transformed_points = np.concatenate(
        [cv2.perspectiveTransform(points, m=H).reshape(-1, 2) for H in [homography]]
    )
    x_min, y_min = np.min(transformed_points, axis=0)
    x_max, y_max = np.max(transformed_points, axis=0)
    bbox = np.ceil([x_min, y_min, x_max, y_max]).astype(np.int32)
    return bbox


def translate_homography(homography: np.ndarray, tx: float, ty: float) -> np.ndarray:
    return homography + np.array(
        [tx * homography[2], ty * homography[2], [0.0, 0.0, 0.0]]
    )


def warp_image(
    image: np.ndarray,
    homography: np.ndarray,
    previous_image: np.ndarray = None,
    flags=cv2.INTER_LINEAR,
):

    height, width, img_c = image.shape[:3]
    x_min, y_min, x_max, y_max = get_transformation_bounding_box(homography, image)
    # pixels to move the original image right and down
    x_t = abs(min(x_min, 0))
    y_t = abs(min(y_min, 0))

    translated_homography = translate_homography(homography, x_t, y_t)

    # resulting image size
    img_w = max(x_max, width) + x_t
    img_h = max(y_max, height) + y_t

    if previous_image is not None:
        height, width = previous_image.shape[:2]
        img_w = max(img_w, width + x_t)
        img_h = max(img_h, height + y_t)

        corrected_image = np.zeros(shape=[img_h, img_w, img_c], dtype=image.dtype)
        corrected_image[y_t : y_t + height, x_t : x_t + width, :] = previous_image
    else:
        corrected_image = np.zeros(shape=[img_h, img_w, img_c], dtype=image.dtype)

    prev_image_resized = corrected_image.copy()
    cv2.warpPerspective(
        src=image,
        M=translated_homography,
        dsize=(img_w, img_h),
        dst=corrected_image,
        flags=flags,
    )
    return corrected_image, translated_homography, prev_image_resized
