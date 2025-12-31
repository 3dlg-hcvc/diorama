import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Metric3D(nn.Module):
    def __init__(
        self,
        encoder='vit_large'
    ):
        super(Metric3D, self).__init__()
        
        self.model = torch.hub.load('yvanyin/metric3d', f'metric3d_{encoder}', pretrain=True)
        
        self.input_size = (616, 1064) if 'vit' in encoder else (544, 1216)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def preprocess_image(self, rgb_origin, intrinsics):
        h, w = rgb_origin.shape[:2]
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        
        intrinsic_scaled = [i * scale for i in intrinsics[:4]] + list(intrinsics[4:])
        
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = self.input_size[0] - h
        pad_w = self.input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :]
        
        return rgb, intrinsic_scaled, pad_info, scale
    
    def postprocess_depth(self, pred_depth, pad_info, original_size, intrinsics):
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
        pred_depth = F.interpolate(pred_depth[None, None, :, :], original_size, mode='bilinear').squeeze()
        
        canonical_to_real_scale = intrinsics[0] / 1000.0
        pred_depth = pred_depth * canonical_to_real_scale
        pred_depth = torch.clamp(pred_depth, 0, 300)
        
        return pred_depth
        
    @torch.no_grad()
    def infer_image(self, raw_image, **kwargs):
        intrinsics = kwargs.get('intrinsics', None)
        assert intrinsics is not None, 'Intrinsics is required for inference'
        raw_image = raw_image[:, :, ::-1]  # Convert BGR to RGB
        h, w = raw_image.shape[:2]
        
        preprocessed_image, scaled_intrinsic, pad_info, scale = self.preprocess_image(raw_image, intrinsics)
        preprocessed_image = preprocessed_image.to(self.device)
        
        pred_depth, confidence, output_dict = self.model.inference({'input': preprocessed_image})
        
        depth_metric = self.postprocess_depth(pred_depth, pad_info, (h, w), scaled_intrinsic)
        
        pred_normal = output_dict['prediction_normal'][:, :3, :, :]
        pred_normal = pred_normal.squeeze()
        pred_normal = pred_normal[:, pad_info[0] : pred_normal.shape[1] - pad_info[1], pad_info[2] : pred_normal.shape[2] - pad_info[3]]
        pred_normal = F.interpolate(pred_normal[None, :, :, :], (h, w), mode='bilinear').squeeze()
        
        out = {"depth": depth_metric.cpu().numpy(),
               "normal": pred_normal.cpu().numpy()}
        
        return out