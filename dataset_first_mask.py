import cv2
import torch
from segment_anything import sam_model_registry
from segment_anything import SamPredictor

import numpy as np
import matplotlib.pyplot as plt


def apply_red_mask(image, mask):
    # Create an all-red image
    red_img = np.zeros_like(image)
    red_img[:, :, 2] = 255  # Only the red channel

    # Combine the red image with the original one according to the mask
    red_masked_img = np.where(mask[:, :, np.newaxis] == True, red_img, image)
    return red_masked_img

def draw_red_rectangle(image_rgb, box):

    # Unpack the box coordinates
    x_min, y_min, x_max, y_max = box

    # Draw a red rectangle on the image
    return cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

def plot_images(image1, image2, title1='Image 1', title2='Image 2'):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axs[0].set_title(title1)
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axs[1].set_title(title2)
    axs[1].axis('off')

    plt.show()


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
MODEL_TYPE = "vit_h"
checkpoint_path = "/home/bruno/Desktop/Darko/BundleSDF/segmentation/sam_vit_h_4b8939.pth"
sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint_path)
# We force the use of cpu due to NN size > 6gb
DEVICE = 'cpu'
sam.to(device=DEVICE)


mask_predictor = SamPredictor(sam, )

image_path = "/home/bruno/Desktop/Darko/BundleSDF/video/rgb/0000.png"
mask_path = "/home/bruno/Desktop/Darko/BundleSDF/video/masks/0000.png"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

print('Original Dimensions : ',image_rgb.shape)
scale_percent = 100 # percent of original size
width = int(image_rgb.shape[1] * scale_percent / 100)
height = int(image_rgb.shape[0] * scale_percent / 100)
dim = (width, height)
image_rgb_resized = cv2.resize(image_rgb, dim, interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ',image_rgb_resized.shape)


mask_predictor.set_image(image_rgb_resized)

box = np.array([300, 200, 425, 400]) * scale_percent // 100
masks, scores, logits = mask_predictor.predict(
    box=box,
    multimask_output=True
)

masked = apply_red_mask(image_rgb_resized, masks[0])
rectangle = draw_red_rectangle(image_rgb_resized, box)
cv2.imwrite(mask_path, masks[0]*255)
plot_images(rectangle, masked)



print("END")