# superpoint-video-stitcher

Fast video sequence stitching using SuperPoint [1] keypoint extractor.

# Requirements

See `requirements.txt`, `python>=3.6.6` is required.

# Usage:

Convert video to images:
```bash
ffmpeg -i video.mp4 %06d.jpg -hide_banner
```
Check stitching in notebook
```python
from lib.keypoint_extractors import SuperPointExtractor, Keypoints
from lib.io import ImagesStreamer
import lib.stitching as st

# assuming horizontal movement, set te resolution of the extractor
# the higher the better (more keypoints) but a cost of processing time
stitcher = st.VideoStitcher((None, 300))

input_folder = "images/*.jpg"
skip = 10
streamer = ImagesStreamer(input_folder, skip, None)

# any direction movement (less stable)
image = stitcher.stitch_sequence(streamer, (200, None))

# horizontal movement (from left to right ==>)
image = stitcher.stitch_left_right_sequence(streamer.reset())

# horizontal movement (from right to left <== )
image = stitcher.stitch_left_right_sequence(streamer.reverse())

# top - bottom
image = stitcher.stitch_top_bottom_sequence(streamer.reset())
```

# References 

* Model weights comes from: https://github.com/mmmfarrell/SuperPoint/blob/master/pretrained_models/sp_v5.tgz
* Tensorflow SuperPoint implementation: https://github.com/rpautrat/SuperPoint
* Original SuperPoint implementation: https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork

# Jupyter notebook tqdm widget

In order to see nice progress bar in the notebook one need to install 
following packages and extenstions

```bash
conda install nodejs
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```