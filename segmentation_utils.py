# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cv2
import os
import numpy as np
import sys

# append a new directory to sys.path
sys.path.append('/home/agostinh/Desktop/BundleSDF/XMem2')

from inference.run_on_video import run_on_video

class Segmenter():
    def __init__(self):
        self.imgs_path = '/home/agostinh/Desktop/BundleSDF/data/rgb'
        self.masks_path = '/home/agostinh/Desktop/BundleSDF/data/masks'   # Should contain annotation masks for frames in `frames_with_masks`
        self.output_path = '/home/agostinh/Desktop/BundleSDF/data'
        return

    def run(self, mask_file=None):
        frames_with_masks = [0]  # indices of frames for which there is an annotation mask
        run_on_video(self.imgs_path, self.masks_path, self.output_path, frames_with_masks)
