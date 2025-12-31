import argparse
import os

import cv2
import numpy as np
import torch
from diorama.model.metric3d import Metric3D
from diorama.utils.depth_util import (
    DEPTH_SCALE,
    INTRINSICS,
    back_project_depth_to_points,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./output')
    parser.add_argument('--encoder', type=str, default='vit_giant2', choices=['vit_small', 'vit_large', 'vit_giant2'])
    parser.add_argument('--intrinsics', type=str, choices=['wss', 'nyu', 'multiview', '<path/to/custom/intrinsics>'], default='wss', help='camera intrinsics')
    parser.add_argument('--normals', action="store_true", help='return normals in addition to depth')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Metric3D(encoder=args.encoder)
    model.to(DEVICE).eval()
    
    scene_name = args.img_path.split('/')[-1].split('.')[0]
    out_dir = f"{args.out_dir}/{scene_name}/arch"
    os.makedirs(out_dir, exist_ok=True)

    img_path = f"{out_dir}/inpainted_merged.png"

    raw_image_with_alpha = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    raw_image = raw_image_with_alpha[:, :, :3]
    alpha_channel = raw_image_with_alpha[:, :, 3]
    alpha_mask = (alpha_channel > 0).astype(np.uint8)
    h, w = raw_image.shape[:2]

    if os.path.exists(args.intrinsics):
        intrinsics = np.loadtxt(args.intrinsics)
    else:
        intrinsics = INTRINSICS[args.intrinsics]
    out = model.infer_image(raw_image, intrinsics=intrinsics)

    depth_metric = out["depth"]
    # cv2.imwrite(f"{out_dir}/inpainted-depth-metric3d-metric.png", (depth_metric * 1000).astype(np.uint16))

    depth_norm = 1 - (depth_metric - depth_metric.min()) / (depth_metric.max() - depth_metric.min())
    depth_vis = cv2.applyColorMap((depth_norm * 255.).astype(np.uint8), cv2.COLORMAP_INFERNO)
    # cv2.imwrite(f"{out_dir}/inpainted-depth-metric3d.png", (depth_norm * 1000).astype(np.uint16))
    cv2.imwrite(f"{out_dir}/inpainted-depth-vis.png", depth_vis)

    np.save(f"{out_dir}/inpainted-depth-metric.npy", depth_metric)
    # np.save(f"{out_dir}/inpainted-depth-metric3d.npy", depth_norm)

    if args.normals:
        pred_normal = out["normal"].transpose(1, 2, 0)
        normal_vis = ((pred_normal + 1) / 2 * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, 'inpainted-normal-vis.png'), normal_vis[:, :, ::-1])
        np.save(f"{out_dir}/inpainted-normal.npy", pred_normal)
        if os.path.exists(args.intrinsics):
            intrinsics = np.loadtxt(args.intrinsics)
        else:
            intrinsics = args.intrinsics
        _, point_to_pixel = back_project_depth_to_points(depth_metric, intrinsics=intrinsics, image=raw_image, save_path=os.path.join(out_dir, "inpainted_pcd.ply"), normal_map=pred_normal, alpha_mask=alpha_mask, return_pixel_map=True)
        np.savez(os.path.join(out_dir, "inpainted_pcd_mapping.npz"), map=point_to_pixel)
    else:
        if os.path.exists(args.intrinsics):
            intrinsics = np.loadtxt(args.intrinsics)
        else:
            intrinsics = args.intrinsics
        _, point_to_pixel = back_project_depth_to_points(depth_metric, intrinsics=intrinsics, image=raw_image, save_path=os.path.join(out_dir, "inpainted_pcd.ply"), alpha_mask=alpha_mask, return_pixel_map=True)
        np.savez(os.path.join(out_dir, "inpainted_pcd_mapping.npz"), map=point_to_pixel)