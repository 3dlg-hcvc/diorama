import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange


class BatchedData:
    """
    A structure for storing data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, batch_size, data=None, **kwargs) -> None:
        self.batch_size = batch_size
        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self):
        assert self.batch_size is not None, "batch_size is not defined"
        if isinstance(self.data, np.ndarray):
            return np.ceil(len(self.data) / self.batch_size).astype(int)
        elif isinstance(self.data, torch.Tensor):
            length = self.data.shape[0]
            return np.ceil(length / self.batch_size).astype(int)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size : (idx + 1) * self.batch_size]

    def cat(self, data, dim=0):
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data):
        self.data.append(data)

    def stack(self, dim=0):
        self.data = torch.stack(self.data, dim=dim)


def gather(features, index_patches):
    """
    Args:
    - features: (B, C, H, W)
    - index_patches: (B, N, 2) where N is the number of patches, and 2 is the (x, y) index of the patch
    Output:
    - selected_features: (BxN, C) where index_patches!= -1
    """
    B, C, H, W = features.shape
    features = rearrange(features, "b c h w -> b (h w) c")

    index_patches = index_patches.clone()
    x, y = index_patches[:, :, 0], index_patches[:, :, 1]
    mask = torch.logical_and(x != -1, y != -1)
    index_patches[index_patches == -1] = H - 1  # dirty fix so that gather does not fail

    # Combine index_x and index_y into a single index tensor
    index = y * W + x

    # Gather features based on index tensor
    flatten_features = torch.gather(
        features, dim=1, index=index.unsqueeze(-1).repeat(1, 1, C)
    )

    # reshape to (BxN, C)
    flatten_features = rearrange(flatten_features, "b n c -> (b n) c")
    mask = rearrange(mask, "b n -> (b n)")
    return flatten_features[mask]



class ISTNet(pl.LightningModule):
    def __init__(
        self,
        model_name,
        backbone,
        regressor,
        max_batch_size,
        patch_size=14,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.patch_size = patch_size
        self.backbone = backbone
        self.regressor = regressor
        self.max_batch_size = max_batch_size
        self._init_weights()

    def get_toUpdate_parameters(self):
        return list(self.backbone.parameters()) + list(self.regressor.parameters())

    def _init_weights(self):
        """Init weights for the MLP"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs):
        batch_rgbs = BatchedData(batch_size=self.max_batch_size, data=processed_rgbs)
        patch_features = BatchedData(batch_size=self.max_batch_size)
        for idx_batch in range(len(batch_rgbs)):
            feats = self.backbone(batch_rgbs[idx_batch])
            patch_features.cat(feats)
        return patch_features.data

    def forward(self, src_img, tar_img, src_pts, tar_pts):
        src_feat = self.forward_by_chunk(src_img)
        tar_feat = self.forward_by_chunk(tar_img)

        src_feat_ = gather(src_feat, src_pts.clone())
        tar_feat_ = gather(tar_feat, tar_pts.clone())
        feats = torch.cat([tar_feat_, src_feat_], dim=1)

        scale = self.regressor.scale_predictor(feats)
        cos_sin_inplane = self.regressor.inplane_predictor(feats)
        if self.regressor.normalize_output:
            cos_sin_inplane = F.normalize(cos_sin_inplane, dim=1)
        return {
            "scale": scale.squeeze(1),
            "inplane": cos_sin_inplane,
        }

    def inference_by_chunk(
        self,
        src_feat,
        tar_feat,
        src_pts,
        tar_pts,
        max_batch_size,
    ):
        batch_src_feat = BatchedData(batch_size=max_batch_size, data=src_feat)
        batch_tar_feat = BatchedData(batch_size=max_batch_size, data=tar_feat)
        batch_src_pts = BatchedData(batch_size=max_batch_size, data=src_pts)
        batch_tar_pts = BatchedData(batch_size=max_batch_size, data=tar_pts)

        pred_scales = BatchedData(batch_size=max_batch_size)
        pred_cosSin_inplanes = BatchedData(batch_size=max_batch_size)

        for idx_sample in range(len(batch_src_feat)):
            pred_scales_, pred_cosSin_inplanes_ = self.inference(
                batch_src_feat[idx_sample],
                batch_tar_feat[idx_sample],
                batch_src_pts[idx_sample],
                batch_tar_pts[idx_sample],
            )
            pred_scales.cat(pred_scales_)
            pred_cosSin_inplanes.cat(pred_cosSin_inplanes_)
        return pred_scales.data, pred_cosSin_inplanes.data

    def inference(self, src_feat, tar_feat, src_pts, tar_pts):
        src_feat_ = gather(src_feat, src_pts.clone())
        tar_feat_ = gather(tar_feat, tar_pts.clone())
        feats = torch.cat([tar_feat_, src_feat_], dim=1)

        scale = self.regressor.scale_predictor(feats)
        cos_sin_inplane = self.regressor.inplane_predictor(feats)

        # same as forward but keep its structure and output angle
        B, N = src_pts.shape[:2]
        device = src_pts.device

        pred_scales = torch.full((B, N), -1000, dtype=src_feat.dtype, device=device)
        pred_cosSin_inplanes = torch.full(
            (B, N, 2), -1000, dtype=src_feat.dtype, device=device
        )

        src_mask = torch.logical_and(src_pts[:, :, 0] != -1, src_pts[:, :, 1] != -1)
        tar_mask = torch.logical_and(tar_pts[:, :, 0] != -1, tar_pts[:, :, 1] != -1)
        assert torch.sum(src_mask) == torch.sum(tar_mask)

        pred_scales[src_mask] = scale.squeeze(1)
        pred_cosSin_inplanes[src_mask] = cos_sin_inplane
        return pred_scales, pred_cosSin_inplanes


class Regressor(nn.Module):
    """
    A simple MLP to regress scale and rotation from DINOv2 features
    """

    def __init__(
        self,
        descriptor_size,
        hidden_dim,
        use_tanh_act,
        normalize_output,
    ):
        super(Regressor, self).__init__()
        self.descriptor_size = descriptor_size
        self.normalize_output = normalize_output
        self.use_tanh_act = use_tanh_act

        self.scale_predictor = nn.Sequential(
            nn.Linear(descriptor_size * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.inplane_predictor = nn.Sequential(
            nn.Linear(descriptor_size * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Tanh() if self.use_tanh_act else nn.Identity(),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        input_dim = config["input_dim"]
        self.input_size = config["input_size"]
        initial_dim = config["initial_dim"]
        block_dims = config["block_dims"]
        descriptor_size = config["descriptor_size"]
        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(
            input_dim, initial_dim, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8
        self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/16
        # 3. FPN upsample
        self.layer4_outconv = conv1x1(block_dims[3], descriptor_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # dirty fix for (224, 224) input
        x = F.interpolate(
            x, (self.input_size, self.input_size), mode="bilinear", align_corners=True
        )
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16
        x4_out = self.layer4_outconv(x4)

        return x4_out
