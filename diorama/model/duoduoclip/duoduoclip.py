import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from . import custom_clip


class DuoduoCLIP(nn.Module):
    def __init__(self, model_name="ViT-B-32-MV", layers_threshold=6, lambda_text=1.0, lambda_image=1.0):
        super().__init__()

        self.layers_threshold = layers_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = custom_clip.get_tokenizer(model_name)
        self.duoduoclip, _, _ = custom_clip.create_model_and_transforms(model_name)

        self.unlock_mha()

        self.norm_transform = transforms.Compose([
            transforms.RandomCrop((224 - 16, 224 - 16)),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def unlock_mha(self):
        for param in self.duoduoclip.parameters():
            param.requires_grad = False
        self.duoduoclip.visual.unlock_mha(layers_threshold=self.layers_threshold)
        
    @classmethod
    def load_pretrained(cls, pretrained_path):
        model = cls()
        pretrained_weights = torch.load(pretrained_path)["state_dict"]
        model.load_state_dict(pretrained_weights, strict=False)
        return model

    def encode_text(self, texts):
        text_tokens = self.tokenizer(texts).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.duoduoclip.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=1)
        return text_features
    
    def encode_image(self, mv_images, is_multiview=False):
        if isinstance(mv_images, list):
            mv_images = np.concatenate([np.array(im)[None, ...] for im in mv_images], axis=0)
        
        # Single-view image
        if len(mv_images.shape) == 3:
            mv_images = torch.from_numpy(mv_images)[None, None, ...]
        # Multi-view image
        elif len(mv_images.shape) == 4:
            if is_multiview:
                mv_images = torch.from_numpy(mv_images).unsqueeze(0)
            else:
                mv_images = torch.from_numpy(mv_images).unsqueeze(1)
        else:
            raise NotImplementedError
        
        mv_images = mv_images.to(torch.float16).permute(0, 1, 4, 2, 3) / 255
        if mv_images.shape[3] != 224 or mv_images.shape[4] != 224:
            mv_images = F.interpolate(mv_images, size=224, mode='bilinear', align_corners=False)

        data_dict = {}
        data_dict['mv_images'] = mv_images.to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            mv_image_features = self(data_dict)["mv_image_features"]
            mv_image_features = F.normalize(mv_image_features, dim=1)
        return mv_image_features

    def forward(self, data_dict):
        bs, f, c, h, w = data_dict['mv_images'].shape
        mv_images = data_dict['mv_images'].reshape(bs * f, c, h, w)
        mv_images = self.norm_transform(mv_images)

        num_frames_list = [1] * self.layers_threshold + [f] * (self.duoduoclip.visual.transformer.layers - self.layers_threshold) + [f]
        
        mv_image_features = self.duoduoclip.encode_image(mv_images, num_frames=num_frames_list)

        output_dict = {"mv_image_features":  mv_image_features}

        return output_dict
        
