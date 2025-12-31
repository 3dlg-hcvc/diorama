import argparse
import os
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation


def extract_object(image, birefnet):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Pad image first, rather than resizing
    h, w = image.size
    pad_h = (image_size[0] - h) // 2
    pad_w = (image_size[1] - w) // 2
    image = ImageOps.expand(image, border=(pad_h, pad_w, pad_h, pad_w), fill=(255, 255, 255))
    input_images = transform_image(image).unsqueeze(0).to('cuda')

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    mask_np = np.array(mask)
    mask_np = mask_np[pad_w:pad_w+w, pad_h:pad_h+h]
    mask_np[mask_np > 0] = 255
    image.putalpha(mask)
    return image, mask_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # Load BiRefNet
    birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    birefnet.to('cuda')
    birefnet.eval()

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Process each scene
    for scene_path in tqdm(glob(f"{args.data_path}/scene*")):
        scene_id = scene_path.split("/")[-1]
        img_path = f"{scene_path}/scene.png"
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert('RGB')
        im, mask = extract_object(image, birefnet)
        # Unpad mask
        os.makedirs(f"{args.output_path}/{scene_id}", exist_ok=True)
        cv2.imwrite(f"{args.output_path}/{scene_id}/dichotomous.png", mask)
        im.save(f"{args.output_path}/{scene_id}/dichotomous_color.png")
        # Now mask-out already masked areas from the image
        image_np = np.array(image)
        image_np[mask == 255] = [255, 255, 255]
        image_masked = Image.fromarray(image_np)
        # Apply second time
        im2, mask2 = extract_object(image_masked, birefnet)
        # Combine masks
        mask_combined = np.logical_or(mask, mask2)
        mask_combined = mask_combined.astype(np.uint8) * 255

        # Combine images
        im_np = np.array(im)
        im2_np = np.array(im2)
        im_combined = im_np + im2_np
        im_combined = Image.fromarray(im_combined)

        im_combined.save(f"{args.output_path}/{scene_id}/dichotomous_color_2_step.png")
        cv2.imwrite(f"{args.output_path}/{scene_id}/dichotomous_2_step.png", mask_combined)
