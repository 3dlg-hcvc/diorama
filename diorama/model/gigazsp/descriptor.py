from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from diorama.model.dinov2 import DINOv2

# Utils
from diorama.model.gigazsp.correspondence import find_correspondences_batch
from diorama.model.gigazsp.utils import scale_points_from_patch


class PatchDescriptor:
    PRETRAINED = {
        'dinov2_vitl14': str(Path(__file__).parent.parent.parent.parent / 'weights/dinov2_vitl14_pretrain.pth'),
        'dinov2_vitl14_giga': str(Path(__file__).parent.parent.parent.parent / 'weights/dinov2_vitl14_gigapose.pth'),
    }
    
    def __init__(
        self,
        model_name,
        patch_size=8,
        feat_layer=9,
        image_size=224,
        n_ref=5,
        num_correspondences=50,
        sim_threshold=0.5,
        best_frame_mode="corresponding_feats_similarity",
    ):
        self.model_name = model_name
        self.patch_size = patch_size
        self.feat_layer = feat_layer
        self.image_size = image_size
        self.n_ref = n_ref
        self.num_correspondences = num_correspondences
        self.sim_threshold = sim_threshold
        self.best_frame_mode = best_frame_mode

        self.batched_correspond = True
        # Image processing
        self.image_norm_mean = (0.485, 0.456, 0.406)
        self.image_norm_std = (0.229, 0.224, 0.225)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_name.startswith('dinov2_vitl14'):
            model = DINOv2("vitl")
            model.load_state_dict(torch.load(self.PRETRAINED[model_name]))
            model.to(self.device).eval()
        self.model = model

    @torch.no_grad()
    def extract_features(self, all_images):
        """
        A definition of relevant dimensions {all_b, nh, t, d}:
            image_size: Side length of input images (assumed square)
            all_b: The first dimension size of the input tensor - not necessarily
                the same as "batch size" in high-level script, as we assume that
                images are all flattened-then-concatenated along the batch dimension. 
                e.g. a batch size of 2, and 5 ref images, 1 query image; all_b = 2 * (5+1) = 12
            h: number of heads in ViT, e.g. 6
            t: number of items in ViT keys/values/tokens, e.g. 785 (= 28*28 + 1)
            d: feature dim in ViT, e.g. 64

        Args:
            all_images (torch.Tensor): shape (all_b, 3, image_size, image_size)
        Returns:
            features (torch.Tensor): shape (all_b, t-1, nh*d) e.g. (12, 784, 384)
            cls_tokens (torch.Tensor): shape (all_b, nh*d) e.g. (12, 384)
        """
        c, img_h, img_w = all_images.shape[-3:]
        all_images = all_images.view(-1, c, img_h, img_w)
        all_images_batch_size = all_images.size(0)
        
        torch.cuda.empty_cache()

        if not self.model_name.startswith('dinov2_vitl14'):
            MAX_BATCH_SIZE = 50
            if all_images_batch_size <= MAX_BATCH_SIZE:
                data = self.model.get_specific_tokens(all_images, layers_to_return=(9, 11))
                features = data[self.feat_layer]['k']
                cls_tokens = data[11]['t'][:, 0, :]
            # Process in chunks to avoid CUDA out-of-memory
            else:
                num_chunks = np.ceil(all_images_batch_size / MAX_BATCH_SIZE).astype('int')
                data_chunks = []
                for i, ims_ in enumerate(all_images.chunk(num_chunks)):
                    data_chunk = self.model.get_specific_tokens(ims_, layers_to_return=(9, 11))
                    data_chunks.append(data_chunk)
                features = torch.cat([d[self.feat_layer]['k'] for d in data_chunks], dim=0)
                cls_tokens = torch.cat([d[11]['t'][:, 0, :] for d in data_chunks], dim=0)
            
            all_b, nh, nt, d = features.shape
            # Remove cls output (first 'patch') from features
            features = features[:, :, 1:, :]  # (all_b) x h x (t-1) x d  e.g. (12, 6, 784, 64)
            features = features.permute(0, 2, 1, 3).reshape(all_b, nt-1, nh * d)  # all_b x (t-1) x (d*h)
        else:
            MAX_BATCH_SIZE = 20
            if all_images_batch_size <= MAX_BATCH_SIZE:
                features, cls_tokens = self.model.get_intermediate_layers(all_images, n=[self.feat_layer], return_class_token=True)[0]
            else:
                num_chunks = np.ceil(all_images_batch_size / MAX_BATCH_SIZE).astype('int')
                output_chunks = []
                for _, ims_ in enumerate(all_images.chunk(num_chunks)):
                    output_chunks.extend(self.model.get_intermediate_layers(ims_, n=[self.feat_layer], return_class_token=True))
                features, cls_tokens = zip(*output_chunks)
                features = torch.cat(features, dim=0)
                cls_tokens = torch.cat(cls_tokens, dim=0)

        return features, cls_tokens

    def create_reshape_descriptors(self, features, batch_size):
        """
        Relevant dimensions are defined as for extract_features model above
        
        3 new dimension params here are:
            B: This is the batch size used in the dataloader/calling script

        Args:
            features (torch.Tensor): shape (all_b, np, d) e.g. (12, 784, 384)

        Returns:
            features (torch.Tensor): shape Bx(n_ref+1)x1x(t-1)xfeat_dim.
        """
        # Reshape back to batched view
        features = features[:, None, :, :]
        _, _, np, feat_dim = features.size()
        features = features.view(batch_size, -1, 1, np, feat_dim) # Bx(n_tgt+1)x1x(t-1)xfeat_dim
        return features

    def split_query_ref(self, features):
        """
        Reshapes, repeats and splits features into query/ref

        Dimensions as for extract_features, create_reshape_descriptors
        Args:
            features (torch.Tensor): shape Bx(n_ref+1)x1x(t-1)xfeat_dim, this is
                a descriptor tensor, rather than raw features from the ViT.

        Returns:
            query_feats (torch.Tensor): shape (B*n_ref)x1x(t-1)xfeat_dim, repeated n_ref times to match ref_feats. 
            ref_feats (torch.Tensor): shape (B*n_ref)x1x(t-1)xfeat_dim.
        """
        batch_size, _, _, t, feat_dim_after_binning = features.shape
        
        # Split descriptors back to query image & ref images
        query_feats, ref_feats = features.split((1, self.n_ref), dim=1)
        query_feats = query_feats.repeat(1, self.n_ref, 1, 1, 1)

        # Flatten first 2 dims again:
        query_feats = query_feats.view(batch_size * self.n_ref, 1, t, feat_dim_after_binning)
        ref_feats = ref_feats.reshape(batch_size * self.n_ref, 1, t, feat_dim_after_binning)
        
        return query_feats, ref_feats


    def get_correspondences(self, query_feats, ref_feats, query_masks, ref_masks, device):
        """
        Args:
            query_feats (torch.Tensor): shape (B*n_ref)x1x(t-1)xfeat_dim, repeated n_ref times to match ref_feats. 
            ref_feats (torch.Tensor): shape (B*n_ref)x1x(t-1)xfeat_dim.
            query_masks (torch.Tensor): shape (B*n_ref)x(t-1), the foreground mask of query features.
            ref_masks (torch.Tensor): shape (B*n_ref)xhxtxt, the foreground mask of ref features.

        Returns:
            selected_points_image_2 (torch.Tensor): Shape (B*n_ref)xself.num_correspondencesx2, 
                this is a tensor giving the 
            selected_points_image_1 (torch.Tensor):
            cyclical_dists (torch.Tensor):
            sim_selected_12 (torch.Tensor):
            
        """
        corresp_points_query, corresp_points_ref, cooresp_points_mask, cyclical_dists, sim_selected_12 = find_correspondences_batch(
            descriptors1=query_feats,
            descriptors2=ref_feats,
            mask1=query_masks, 
            mask2=ref_masks, 
            num_pairs=self.num_correspondences,
            sim_threshold=self.sim_threshold,
            device=device)

        return (corresp_points_query, corresp_points_ref, cooresp_points_mask, cyclical_dists, sim_selected_12)

    def find_closest_match(self, cls_tokens, query_patchs, renders_patchs, sim_selected_12, batch_size, valid_ref_masks):
        if self.best_frame_mode == 'cls_similarity':
            cls_tokens = torch.cat(cls_tokens, dim=0)
            cls_tokens = cls_tokens.view(batch_size, self.n_ref + 1, -1)
            query_cls_tokens, ref_cls_tokens = cls_tokens.split((1, self.n_ref), dim=1)
            similarities = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)(query_cls_tokens, ref_cls_tokens)
            similarities[~valid_ref_masks] = -1e6
            best_idxs = similarities.argmax(dim=-1)
        elif self.best_frame_mode == 'corresponding_feats_similarity':
            similarities = sim_selected_12.view(batch_size, self.n_ref, self.num_correspondences).sum(dim=-1)
            similarities[~valid_ref_masks] = -1e6
            best_idxs = similarities.argmax(dim=-1)
        elif self.best_frame_mode == 'patch_feats_similarity':
            query_patchs = torch.cat(query_patchs, dim=0)
            renders_patchs = torch.cat(renders_patchs, dim=0)
            similarities = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)(query_patchs, renders_patchs).mean(-1).view(-1, self.n_ref) # (bs, n_ref)
            similarities[~valid_ref_masks] = -1e6
            best_idxs = similarities.argmax(dim=-1)
        else:
            raise ValueError(f'model of picking best frame not implemented: {self.best_frame_mode}')
        return similarities, best_idxs


    def scale_patch_to_pix(self, points1, points2, N):
        """
        Args:
            points1 (torch.Tensor): shape num_correspondencesx2, the *patch* coordinates
                of correspondence points in image 1 (the query image)
            points2 (torch.Tensor): shape num_correspondencesx2, the *patch* coordinates
                of correspondence points in image 2 (the best ref image)
            N (int): N is the height or width of the feature map
        """
        if self.batched_correspond:
            points1_rescaled, points2_rescaled = (scale_points_from_patch(
                p, vit_image_size=self.image_size, num_patches=N) for p in (points1, points2))
        else: # earlier descriptor extractor functions for ViT features scaled before return
            points1_rescaled, points2_rescaled = (points1, points2)
        return points1_rescaled, points2_rescaled


    def get_transform(self):
        image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.image_norm_mean, std=self.image_norm_std)
        ])
        return image_transform


    def denorm_torch_to_pil(self, image):
        image = image * torch.Tensor(self.image_norm_std)[:, None, None]
        image = image + torch.Tensor(self.image_norm_mean)[:, None, None]
        return Image.fromarray((image.permute(1, 2, 0) * 255).numpy().astype(np.uint8))