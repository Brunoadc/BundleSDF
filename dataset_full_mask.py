import sys

# append a new directory to sys.path
sys.path.append('/home/agostinh/Desktop/BundleSDF/XMem2')

from inference.run_on_video import run_on_video

print("Hello")

imgs_path = '/home/agostinh/Desktop/BundleSDF/data/rgb'
masks_path = '/home/agostinh/Desktop/BundleSDF/data/masks'   # Should contain annotation masks for frames in `frames_with_masks`
output_path = '/home/agostinh/Desktop/BundleSDF/data'
frames_with_masks = [0]  # indices of frames for which there is an annotation mask
run_on_video(imgs_path, masks_path, output_path, frames_with_masks)