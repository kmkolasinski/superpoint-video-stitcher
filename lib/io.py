from glob import glob
from typing import Tuple, Optional, Type
import cv2
import numpy as np
from natsort import natsorted


class ImagesStreamer:
    def __init__(
        self, images_glob: str, skip_frames: int = 1, max_frames: Optional[int] = None
    ):
        """
        Image sequence loader class.

        Usage:

            streamer = ImagesStreamer("images/*.png", 10)
            while True:
                image, status = streamer.next_frame()
                if not status:
                    break

        Args:
            images_glob: for example "path/to/images/*.png"
            skip_frames: frames stride. If 1 all frames will be used.
            max_frames: maximum number of frames to read
        """

        self.images_glob = images_glob
        self.skip_frames = skip_frames
        self.max_frames = max_frames
        self.current_frame_index = 0
        self.listing = []
        self.init()

    def copy(self, base_type: Type["ImagesStreamer"] = None) -> 'ImagesStreamer':
        if base_type is None:
            base_type = ImagesStreamer
        return base_type(
            images_glob=self.images_glob, skip_frames=self.skip_frames, max_frames=self.max_frames
        )

    def init(self):
        self.listing = natsorted(glob(self.images_glob))
        self.listing = self.listing[::self.skip_frames]
        if self.max_frames is not None:
            self.listing = self.listing[:self.max_frames]

        if self.num_images == 0:
            raise IOError(f"No images were found in path: {self.images_glob}")

    @property
    def num_images(self) -> int:
        return len(self.listing)

    def reset(self) -> "ImagesStreamer":
        return self.copy()

    def reverse(self) -> "ImagesStreamer":
        streamer = self.copy()
        streamer.listing = streamer.listing[::-1]
        return streamer

    def read_image(self, image_path: str) -> np.ndarray:
        frame_image = cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)]
        return frame_image

    def next_frame(self) -> Tuple[Optional[np.ndarray], bool]:
        """
        Process next frame for the directory
        Returns:
            image: a numpy uint8 image of shape [height, width, 3]
            status: if False "streaming" is done
        """
        if self.current_frame_index == self.num_images:
            return None, False

        input_image = self.read_image(self.listing[self.current_frame_index])
        self.current_frame_index = self.current_frame_index + 1
        return input_image, True
