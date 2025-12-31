import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.measure import ransac
from torchvision.transforms import ToTensor, Normalize
from tqdm import tqdm

from diorama.model.gigazsp.descriptor import PatchDescriptor
from diorama.model.gigazsp.utils import (
    CropResizePad,
    RigidBodyUmeyama,
    ransac_sample_validation,
    inverse_affine,
)
from diorama.model.gigazsp.ransac import BatchAffine2DRANSAC

from diorama.utils.cad_util import RENDER_DIR
from diorama.utils.depth_util import back_project_depth_to_points, load_K
from diorama.utils.obb_util import OBB
# from diorama.utils.render_util import render_with_view_T
from diorama.utils.viz_util import draw_correspondences_lines, tile_ims_horizontal_highlight_best



class GigaZSP:
    
    def __init__(
        self,
        model_name=None,
        patch_size=8,
        feat_layer=9,
        n_ref=24,
        num_correspondences=50,
        sim_threshold=0.5,
        take_best_view=False,    # if True, simply use the best view as the pose estimate
        ransac_threshold=0.2,
        ransac_min_samples=None,
        ransac_max_trials=10000,
        best_frame_mode='corresponding_feats_similarity',
        cad_pick=0,
        max_chunk_size=2,
        ist_net=None,
        ist_net_ckpt_path=None,
        max_objs_per_forward=2
    ):
        self.model_name = model_name
        self.patch_size = patch_size
        self.feat_layer = feat_layer
        self.ransac_threshold = ransac_threshold
        self.num_correspondences = num_correspondences
        self.n_ref = n_ref
        self.take_best_view = take_best_view
        self.ransac_min_samples = ransac_min_samples
        self.ransac_max_trials = ransac_max_trials
        self.cad_pick = cad_pick
        self.max_chunk_size = max_chunk_size
        self.max_objs_per_forward = max_objs_per_forward
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.desc = PatchDescriptor(
            model_name,
            patch_size=patch_size,
            feat_layer=feat_layer,
            image_size=224,
            n_ref=n_ref,
            num_correspondences=num_correspondences,
            sim_threshold=sim_threshold,
            best_frame_mode=best_frame_mode
        )
        
        ist_net.load_state_dict(torch.load(ist_net_ckpt_path)["state_dict"])
        ist_net.to(self.device).eval()
        self.ist_net = ist_net
        
        self.crop_transform = CropResizePad(target_size=224, patch_size=14)
        self.norm_transform = Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    
    
    def prepare_inputs(self, query_dir, scenespec, **kwargs):
        self.query_dir = query_dir
        self.scenespec = scenespec
        self.cad_pick = kwargs.get("cad_pick", self.cad_pick)
        self.obj_ids = kwargs.get("obj_ids", list(scenespec.objects.keys()))
        
        query_images, query_pcds, query_masks, all_ref_images, all_ref_masks = [], [], [], [], []
        query_xyxys = []
        for idx, obj_id in enumerate(self.obj_ids):
            cad_entry = scenespec.objects[obj_id]['retrieval'][self.cad_pick]
            dataset, subdir, cad_id = cad_entry.split(",")[1:4]
            
            img = Image.open(f"{query_dir}/detection/rgb.{obj_id}.jpg")
            box = img.getbbox() # (left, upper, right, lower)
            query_xyxys.append(box)
            query_images.append(img)
            
            pcd = np.load(f"{query_dir}/pcd/pcd.{obj_id}.npy")
            query_pcds.append(pcd)
            # pcd_crop = pcd[box[1]:box[3], box[0]:box[2], :]
            # pcd_crop_resize = cv2.resize(pcd_crop, inst_size, interpolation=cv2.INTER_NEAREST)
            # query_pcd = np.zeros((224, 224, 3))
            # query_pcd[(224-inst_size[1])//2:(224+inst_size[1])//2, 
            #           (224-inst_size[0])//2:(224+inst_size[0])//2, :] = pcd_crop_resize
            # query_pcds.append(query_pcd)
            # query_masks.append(np.any(query_pcd, axis=-1))
            
            all_ref_images.extend([Image.open(os.path.join(RENDER_DIR[dataset].format(subdir=subdir, cad_id=cad_id, n_ref=self.n_ref), f"{i}.png")).convert("RGB").resize((224, 224)) for i in range(self.n_ref)])
            all_ref_masks.extend([Image.open(os.path.join(RENDER_DIR[dataset].format(subdir=subdir, cad_id=cad_id, n_ref=self.n_ref), f"{i}.mask.png")).convert("L").resize((224, 224)) for i in range(self.n_ref)])
        
        query_images = torch.stack([ToTensor()(im) for im in query_images]) # (bs, 3, h, w)
        query_xyxys = torch.stack([torch.tensor(box).long() for box in query_xyxys]) # (bs, 4)
        query_pcds = torch.stack([torch.tensor(pcd).permute(2, 0, 1) for pcd in query_pcds]) # (bs, 3, h, w)
        
        cropped_images, crop_Ms = self.crop_transform(query_images, query_xyxys)
        cropped_images = self.norm_transform(cropped_images)
        cropped_pcds, _ = self.crop_transform(query_pcds, query_xyxys)
        cropped_masks = torch.any(cropped_pcds, dim=1, keepdim=True).float()
        
        self.query_images = cropped_images # (bs, 3, h, w)
        self.query_pcds = [cropped_pcds[b].permute(1,2,0).numpy() for b in range(len(cropped_pcds))]
        self.query_Ms = crop_Ms # (bs, 3, 3)

        image_transform = self.desc.get_transform()
        # self.query_images = torch.stack([image_transform(im) for im in query_images]) * 1 # TODO
        # self.query_masks = torch.stack([ToTensor()(m.astype(float)) for m in query_masks])
        self.query_patch_masks = torch.nn.MaxPool2d(self.patch_size, stride=self.patch_size)(cropped_masks)
        # self.query_pcds = query_pcds
        
        self.all_ref_images = torch.stack([image_transform(im) for im in all_ref_images]).view(-1, self.n_ref, 3, 224, 224)
        all_ref_masks = torch.stack([ToTensor()(m) for m in all_ref_masks]) # (bs*n_ref, 1, 224, 224)
        self.all_ref_valid_flags = torch.any(all_ref_masks, dim=[1,2,3]).view(-1, self.n_ref)
        self.all_ref_patch_masks = torch.nn.MaxPool2d(self.patch_size, stride=self.patch_size)(all_ref_masks)
        
        self.query_K = load_K(kwargs["data_source"], scene_name=kwargs["scene_name"]) # (1, 3, 3)
        self.ref_K = load_K("multiview")
        self.focal_ratio = self.query_K[0, 0] / self.ref_K[0, 0]
        
        self.encode_templates()
    
    
    @torch.no_grad()
    def encode_templates(self):
        torch.cuda.empty_cache()

        ist_features = self.ist_net.forward_by_chunk(self.all_ref_images.view(-1, 3, 224, 224).to(self.device))
        self.all_ref_ist_features = ist_features.reshape(-1, self.n_ref, 256, 16, 16) # (bs, n_ref, 256, h, w)
        
    
    def get_correspond_and_best_view(self):
        # ---------------
        # GET DINO FEATURES
        # ---------------
        # Query images shape: B x 3 x S x S (S = size, assumed square images)
        # Ref images shape: B x N_Ref x 3 x S x S
        batch_size = self.query_images.shape[0]
        all_images = torch.cat([self.query_images.unsqueeze(1), self.all_ref_images], dim=1).to(self.device) # B x (n_ref + 1) x 3 x S x S
        self.dino_feat_size = all_images.shape[-1] // self.patch_size
        
        query_masks = self.query_patch_masks.repeat(1, self.n_ref, 1, 1).reshape(batch_size*self.n_ref, -1).to(bool).to(self.device) # (B*n_ref)x(28*28)
        all_ref_masks = self.all_ref_patch_masks.reshape(batch_size*self.n_ref, -1).to(bool).to(self.device) # (B*n_ref)x(28*28)
        
        all_cls_tokens, all_query_feats, all_ref_feats = [], [], []
        corresp_points_query, corresp_points_ref, cooresp_points_mask, sim_selected_12 = [], [], [], []
        for i in range(int(np.ceil(batch_size/self.max_chunk_size))):
            all_images_chunk = all_images[i*self.max_chunk_size:(i+1)*self.max_chunk_size]
            query_masks_chunk = query_masks[i*self.max_chunk_size*self.n_ref:(i+1)*self.max_chunk_size*self.n_ref]
            all_ref_masks_chunk = all_ref_masks[i*self.max_chunk_size*self.n_ref:(i+1)*self.max_chunk_size*self.n_ref]
            chunk_size = all_images_chunk.shape[0]
            
            # Extract features and cls_tokens
            features, cls_tokens = self.desc.extract_features(all_images_chunk)
            # Create descriptors from features, descriptors shape bsx(n_ref+1)x1x(t-1)xfeat_dim
            features = self.desc.create_reshape_descriptors(features, chunk_size)
            # Split query/ref, repeat query to match size of ref, and flatten into batch dimension
            query_feats, ref_feats = self.desc.split_query_ref(features)
            
            if self.desc.best_frame_mode == 'cls_similarity':
                all_cls_tokens.append(cls_tokens)
            elif self.desc.best_frame_mode == 'patch_feats_similarity':
                all_query_feats.append(query_feats.squeeze(1))
                all_ref_feats.append(ref_feats.squeeze(1))
            
            # ----------------
            # GET CORRESPONDENCES
            # ----------------
            corresp_results = self.desc.get_correspondences(query_feats, ref_feats, query_masks_chunk, all_ref_masks_chunk, self.device)
            corresp_points_query.append(corresp_results[0]) # (bs*n_ref)x50x2
            corresp_points_ref.append(corresp_results[1]) # (bs*n_ref)x50x2
            cooresp_points_mask.append(corresp_results[2]) # (bs*n_ref)x50
            sim_selected_12.append(corresp_results[4]) # (bs*n_ref)x50
            # corresp_results[3] is cyclical_dists, # (bs*n_ref)x28x28
            
        # all_cls_tokens = torch.cat(all_cls_tokens, dim=0)
        # all_query_feats = torch.cat(all_query_feats, dim=0)
        # all_ref_feats = torch.cat(all_ref_feats, dim=0)
        self.corresp_points_query = torch.cat(corresp_points_query, dim=0)
        self.corresp_points_ref = torch.cat(corresp_points_ref, dim=0)
        self.cooresp_points_mask = torch.cat(cooresp_points_mask, dim=0)
        sim_selected_12 = torch.cat(sim_selected_12, dim=0)
        
        _, best_idxs = self.desc.find_closest_match(all_cls_tokens, all_query_feats, all_ref_feats, sim_selected_12, batch_size, self.all_ref_valid_flags)
        self.best_idxs = best_idxs.view(-1).cpu().numpy()
    
    
    def predict_scale(self):
        B = self.query_images.shape[0]
        num_patches = self.corresp_points_ref.shape[1] # 50
        k = 1
        best_view_idxs = torch.tensor(self.best_idxs).unsqueeze(1)
        pred_scales = torch.zeros(B, k, num_patches, device=self.device)
        pred_cosSin_inplanes = torch.zeros(B, k, num_patches, 2, device=self.device)
        src_pts = torch.zeros(B, k, num_patches, 2, device=self.device, dtype=torch.long)
        tar_pts = torch.zeros(B, k, num_patches, 2, device=self.device, dtype=torch.long)

        idx_sample = torch.arange(0, B)
        for idx_k in range(k):
            idx_views = [idx_sample, best_view_idxs[:, idx_k]]

            src_ist_features = self.all_ref_ist_features[idx_sample].to(self.device)
            tar_ist_features = self.ist_net.forward_by_chunk(self.query_images[idx_sample].to(self.device))
            src_pts[:, idx_k] = self.corresp_points_ref.view(B, self.n_ref, num_patches, 2)[idx_views].to(self.device)
            tar_pts[:, idx_k] = self.corresp_points_query.view(B, self.n_ref, num_patches, 2)[idx_views].to(self.device)

            if self.max_objs_per_forward is not None:
                (
                    pred_scales[:, idx_k],
                    pred_cosSin_inplanes[:, idx_k],
                ) = self.ist_net.inference_by_chunk(
                    src_feat=src_ist_features[idx_views],
                    tar_feat=tar_ist_features,
                    src_pts=src_pts[:, idx_k],
                    tar_pts=tar_pts[:, idx_k],
                    max_batch_size=self.max_objs_per_forward,
                )
            else:
                (
                    pred_scales[:, idx_k],
                    pred_cosSin_inplanes[:, idx_k],
                ) = self.ist_net.inference(
                    src_feat=src_ist_features[idx_views],
                    tar_feat=tar_ist_features,
                    src_pts=src_pts[:, idx_k],
                    tar_pts=tar_pts[:, idx_k],
                )
                
        affine2d_ransac = BatchAffine2DRANSAC(pixel_threshold=self.patch_size)
        ransac_output = affine2d_ransac.forward_ransac(src_pts=src_pts, tar_pts=tar_pts, relScales=pred_scales, relInplanes=pred_cosSin_inplanes)

        # self.pred_scales = torch.norm(ransac_output["best_Ms"][:, :, :2, 0], dim=2)
        scales2d = torch.norm(ransac_output["best_Ms"][:, :, :2, 0], dim=2)
        inv_query_M = inverse_affine(self.query_Ms)
        inv_query_M = inv_query_M.unsqueeze(1).repeat(1, k, 1, 1)
        self.pred_scales = scales2d.cpu() * torch.norm(inv_query_M[:, :, :2, 0], dim=2)
        # self.pred_inplanes = pred_cosSin_inplanes
    
    
    def prepare_pcd(self):
        self.all_points1, self.all_points2, self.best_view_pcd = [], [], []
        for i, obj_id in enumerate(self.obj_ids):
            cad_entry = self.scenespec.objects[obj_id]['retrieval'][self.cad_pick]
            dataset, subdir, cad_id = cad_entry.split(",")[1:4]
            # -----------------
            # PREPARE DATA
            # -----------------
            ref_frame = self.best_idxs[i]

            # query_scaling = 1 # if the original img size is 224
            # ref_scaling = 1 # if the original img size is 224
            ref_depth = cv2.imread(os.path.join(RENDER_DIR[dataset].format(subdir=subdir, cad_id=cad_id, n_ref=self.n_ref), f"{ref_frame}.depth.png"), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR).astype(float) / 1000.

            # From here on, '1' <--> query and '2' <--> ref
            # Get points and if necessary scale them from patch to pixel space
            points1, points2 = (
                self.corresp_points_query[i*self.n_ref + ref_frame],
                self.corresp_points_ref[i*self.n_ref + ref_frame]
            ) # 50x2
            points1, points2 = (
                points1[self.cooresp_points_mask[i*self.n_ref + ref_frame]],
                points2[self.cooresp_points_mask[i*self.n_ref + ref_frame]]
            ) # valid_corresp x 2
            points1_rescaled, points2_rescaled = self.desc.scale_patch_to_pix(
                points1, points2, self.dino_feat_size
            ) # valid_corresp x 2, pixel positions
            self.all_points1.append(points1_rescaled.clone().long())
            self.all_points2.append(points2_rescaled.clone().long())
            # # Now rescale to the *original* image pixel size, i.e. prior to resizing crop to square
            # points1_rescaled, points2_rescaled = (scale_points_to_orig(p, s) for p, s in zip(
            #     (points1_rescaled, points2_rescaled), (query_scaling, ref_scaling)
            # ))
            
            struct_pcd2 = back_project_depth_to_points(ref_depth, intrinsics="multiview")
            self.best_view_pcd.append(struct_pcd2)
    
    
    def estimate_pose_from_best_view(self, struct_pcd1, struct_pcd2, pred_scale=None):
        pcd1 = struct_pcd1[np.any(struct_pcd1, -1)]
        # yy_ind, xx_ind = np.nonzero(np.any(struct_pcd1, -1).T)
        # band_width = (yy_ind.max() - yy_ind.min()) // 40
        # # left_end_ind = np.where(yy_ind == yy_ind[0])[0]
        # left_end_ind = np.where((yy_ind >= yy_ind[0]) & (yy_ind <= yy_ind[0]+band_width))[0]
        # left_end_points1 = struct_pcd1[xx_ind[left_end_ind], yy_ind[left_end_ind]]
        # # right_end_ind = np.where(yy_ind == yy_ind[-1])[0]
        # right_end_ind = np.where((yy_ind >= yy_ind[-1]-band_width) & (yy_ind <= yy_ind[-1]))[0]
        # right_end_points1 = struct_pcd1[xx_ind[right_end_ind], yy_ind[right_end_ind]]
        # scale1 = np.linalg.norm(left_end_points1.mean(0) - right_end_points1.mean(0))
        
        pcd2 = struct_pcd2[np.any(struct_pcd2, -1)]
        # try:
        #     yy_ind, xx_ind = np.nonzero(np.any(struct_pcd2, -1).T)
        #     band_width = (yy_ind.max() - yy_ind.min()) // 40
        #     # left_end_ind = np.where(yy_ind == yy_ind[0])[0]
        #     left_end_ind = np.where((yy_ind >= yy_ind[0]) & (yy_ind <= yy_ind[0]+band_width))[0]
        #     left_end_points2 = struct_pcd2[xx_ind[left_end_ind], yy_ind[left_end_ind]]
        #     # right_end_ind = np.where(yy_ind == yy_ind[-1])[0]
        #     right_end_ind = np.where((yy_ind >= yy_ind[-1]-band_width) & (yy_ind <= yy_ind[-1]))[0]
        #     right_end_points2 = struct_pcd2[xx_ind[right_end_ind], yy_ind[right_end_ind]]
        #     scale2 = np.linalg.norm(left_end_points2.mean(0) - right_end_points2.mean(0))
        # except:
        #     scale2 = 1
        
        # trans21 = get_world_to_view_transform()
        trans21 = np.eye(4)
        median_pos1 = np.concatenate([pcd1[:,:2].mean(0), np.median(pcd1[:,2], keepdims=True)])
        pcd1_z_ctr = median_pos1[-1]
        median_pos2 = np.concatenate([pcd2[:,:2].mean(0), np.median(pcd2[:,2], keepdims=True)])
        pcd2_z_ctr = median_pos2[-1]
        trans21[0,0] = trans21[1,1] = trans21[2,2] = (pcd1_z_ctr / pcd2_z_ctr) * pred_scale / self.focal_ratio
        # trans21[0,0] = trans21[1,1] = trans21[2,2] = (scale1 + 1e-6) / (scale2 + 1e-6)
        translation21 = median_pos1 - trans21[0,0] * median_pos2
        trans21[:3,3] = translation21
        
        return trans21
    
    
    def estimate_pose_from_correspond(self, struct_pcd1, pcd1_pos, struct_pcd2, pcd2_pos, pred_scale=None):
        if len(pcd1_pos) < 3 or len(pcd2_pos) < 3:
            return None, -1
        world_corr1 = struct_pcd1[pcd1_pos[:, 0], pcd1_pos[:, 1]]#.numpy()
        world_corr2 = struct_pcd2[pcd2_pos[:, 0], pcd2_pos[:, 1]]#.numpy()
        invalid = np.logical_or(world_corr1[:, -1] == 0, world_corr2[:, -1] == 0)
        world_corr1 = world_corr1[~invalid]
        world_corr2 = world_corr2[~invalid]
        if not (len(world_corr1) and len(world_corr2)):
            return None, -1
        pcd1_z_ctr = (world_corr1[..., -1].max() + world_corr1[..., -1].min()) / 2
        pcd2_z_ctr = (world_corr2[..., -1].max() + world_corr2[..., -1].min()) / 2
        # -----------------
        # COMPUTE RELATIVE OFFSET
        # -----------------
        best_transform = None
        best_err = np.inf
        min_samples = min(self.ransac_min_samples, len(world_corr2)) if self.ransac_min_samples is not None else len(world_corr2)//2+1
        if min_samples < 3:
            return None, -1
        
        world_corr2_homo = np.vstack((world_corr2.T, np.ones((1, (len(world_corr2))))))
        for ransac_threshold in [0.001, 0.005, 0.01, 0.05, 0.1]:
        # for ransac_threshold in [0.01, 0.05, 0.1]:
            # trans21 = pose.solve_umeyama_ransac(world_corr1, world_corr2)
            # trans21, R, T, scale = pose.solve_umeyama_ransac(world_corr1, world_corr2)
            rbt, inliers = self.solve_umeyama_ransac(world_corr2, 
                                                     world_corr1, 
                                                     ransac_threshold=ransac_threshold, 
                                                     min_samples=min_samples)
            if rbt is None or inliers.sum() < min_samples:
                # print("No inliers found. Model not fitted")
                continue
            trans21 = rbt.T
            world_corr1_est = (trans21 @ world_corr2_homo)[:3, :].T
            mean_err = np.linalg.norm(world_corr1_est-world_corr1, axis=1).mean()
            if mean_err < best_err:
                best_transform = trans21.copy()
                best_err = mean_err.copy()
            
            if pred_scale:
                scale = (pcd1_z_ctr / pcd2_z_ctr) * pred_scale / self.focal_ratio
                trans21[:3, :3] = trans21[:3, :3] / rbt.lam * scale
                world_corr1_est = (trans21 @ world_corr2_homo)[:3, :].T
                mean_err = np.linalg.norm(world_corr1_est-world_corr1, axis=1).mean()
                if mean_err < best_err:
                    best_transform = trans21.copy()
                    best_err = mean_err.copy()
            # else:
            #     scale = rbt.lam
            #     if scale < 1e-3 or scale > 1e2:
            #         continue

        return best_transform, best_err
    
    
    def estimate_poses(self, output_dir):
        self.get_correspond_and_best_view()
        self.predict_scale()
        self.prepare_pcd()
        
        all_trans21, all_errs = [], []
        for bi, (struct_pcd1, struct_pcd2, pcd1_pos, pcd2_pos) in tqdm(enumerate(zip(self.query_pcds, self.best_view_pcd, self.all_points1, self.all_points2))):
            pred_scale2d = self.pred_scales[bi].item()
            # --- If "take best view", simply return identity transform estimate ---
            if self.take_best_view or pcd1_pos is None or pcd2_pos is None:
                trans21 = self.estimate_pose_from_best_view(struct_pcd1, struct_pcd2, pred_scale=pred_scale2d)
                mean_err = -1
            # --- Otherwise, compute transform based on 3D point correspondences ---
            else:
                # try:
                # assert pcd1_pos is not None and pcd2_pos is not None, "points positions are None..."
                trans21, mean_err = self.estimate_pose_from_correspond(struct_pcd1, pcd1_pos, struct_pcd2, pcd2_pos, pred_scale=pred_scale2d)
                # except:
                if trans21 is None:
                    print("Correspondence-based pose estimation failed... Take best view instead...")
                    trans21 = self.estimate_pose_from_best_view(struct_pcd1, struct_pcd2, pred_scale=pred_scale2d)
                    mean_err = -1
            all_trans21.append(trans21)
            all_errs.append(mean_err)
            
        self.all_trans21 = all_trans21
        self.all_errs = all_errs
        
        self.compose_and_save_view_transform(output_dir)
        
        return self.scenespec
    
    
    def solve_umeyama_ransac(self, world_corr1, world_corr2, ransac_threshold=0.02, min_samples=10):
        rbt, inliers = ransac(data=(world_corr1, world_corr2),
                                model_class=RigidBodyUmeyama,
                                min_samples=min_samples,
                                is_data_valid=ransac_sample_validation,
                                residual_threshold=ransac_threshold,
                                max_trials=5000)

        return rbt, inliers
        # print(f"{sum(inliers)} inliers from {self.num_correspondences} points")
        # R = rbt.T[:3, :3] / rbt.lam
        # t = rbt.T[:3, 3:]
        # scale = rbt.lam
        # return rbt.T, R, t, scale, inliers

        # R_ = Rotate(torch.Tensor(R.T).unsqueeze(0))
        # T_ = Translate(torch.tensor(t.T))
        # S_ = Scale(scale)
        # # trans21 = get_world_to_view_transform(torch.Tensor(R.T).unsqueeze(0), torch.tensor(t.T))
        # trans21 = S_.compose(R_.compose(T_))
        # return trans21
    
    
    def compose_and_save_view_transform(self, output_dir):
        if self.n_ref == 24:
            cam_poses = np.load("data/renders-poses/cam_poses_24.npy")
        elif self.n_ref == 60:
            source = "shapenet" if "s2c" in output_dir else "wss"
            cam_poses = np.load(f"data/renders-poses/cam_poses_60_{source}.npy")
        elif self.n_ref == 180:
            source = "shapenet" if "s2c" in output_dir else "wss"
            cam_poses = np.load(f"data/renders-poses/cam_poses_180_{source}.npy")
        
        all_view_trans = []
        for i, obj_id in enumerate(self.obj_ids):
            view_idx = self.best_idxs[i]
            cam_pose = cam_poses[view_idx]
            world2cam = np.linalg.inv(cam_pose)
            
            trans21 = self.all_trans21[i]
            view_trans = trans21 @ world2cam
            
            obb_path = os.path.join(output_dir, f'cad/{obj_id}.{self.cad_pick}.obb.json')
            obb = OBB.create_from_json(obb_path)
            obb.apply_transform(view_trans)
            obb.export_json(obb_path)
            
            view_trans_nmz = view_trans @ obb.norm_mat
            self.scenespec.objects[obj_id].setdefault("view_idx", []).append(int(view_idx))
            self.scenespec.objects[obj_id].setdefault("view_trans", []).append(view_trans_nmz.tolist())
            self.scenespec.objects[obj_id].setdefault("trans_err", []).append(self.all_errs[i])
            all_view_trans.append(view_trans_nmz)
        
        self.all_view_trans = all_view_trans
    
    
    def viz_correspondence_and_pose(self, output_dir, **kwargs):
        corresp_dir = f'{output_dir}/correspondence'
        os.makedirs(corresp_dir, exist_ok=True)
        scene_dir = f'{output_dir}/scene'
        os.makedirs(scene_dir, exist_ok=True)
        
        # meshes = []
        for i, obj_id in enumerate(self.obj_ids):
            cad_entry = self.scenespec.objects[obj_id]['retrieval'][self.cad_pick]
            ref_frame = self.best_idxs[i]
            fig, axs = plt.subplot_mosaic([['A', 'B', 'B'],
                                           ['C', 'C', 'D']],
                                           figsize=(10,5))
            for ax in axs.values():
                ax.axis('off')
            axs['A'].set_title('Query image')
            axs['B'].set_title('Reference images')
            axs['C'].set_title('Correspondences')
            axs['D'].set_title('Reference object in query pose')
            # fig.suptitle(f'Error: {all_errs[i]:.2f}', fontsize=6)
            axs['A'].imshow(self.desc.denorm_torch_to_pil(self.query_images[i]))
            tgt_pils = [self.desc.denorm_torch_to_pil(self.all_ref_images[i][j%self.n_ref]) for j in range(ref_frame-2, ref_frame+3)]
            tgt_pils = tile_ims_horizontal_highlight_best(tgt_pils, highlight_idx=2)
            axs['B'].imshow(tgt_pils)

            draw_correspondences_lines(self.all_points1[i], self.all_points2[i],
                                        self.desc.denorm_torch_to_pil(self.query_images[i]),
                                        self.desc.denorm_torch_to_pil(self.all_ref_images[i][ref_frame]),
                                        axs['C'])
            
            # view_trans = self.all_view_trans[i]
            # mesh = load_mesh(cad_entry, normalize=False, load_textures=False)
            # im = render_with_view_T(mesh, view_trans, obj_id=obj_id, img_size=kwargs['img_size'], data_source=kwargs['data_source'], scene_name=kwargs['scene_name'])
            # axs['D'].imshow(im)
            # meshes.append(mesh)
            
            plt.tight_layout()
            plt.savefig(corresp_dir+f'/corresp.{obj_id}.{self.cad_pick}.png', dpi=150)
            plt.close('all')

        # scene_mesh = sum(meshes)
        # scene_mesh.export(scene_dir+'/scene_init.ply')
