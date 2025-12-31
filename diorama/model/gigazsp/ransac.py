import torch
from diorama.model.gigazsp.utils import affine_torch, apply_affine


class RANSAC(torch.nn.Module):
    def __init__(
        self,
        pixel_threshold,
        patch_size=14,
    ):
        super(RANSAC, self).__init__()
        self.patch_size = patch_size
        self.pixel_threshold = pixel_threshold

    @staticmethod
    def _sample(N, device):
        """
        Taking a random sample of combinations k of N
        Output:
            selected_idx: range(N, 1)
            remaining_idx: (N, N-1) with selected_idx removed
        """
        # Generate all combinations and pick a random sample
        selected_idx = torch.arange(N)
        remaing_idx = torch.zeros(N, dtype=torch.long).unsqueeze(1).repeat(1, N - 1)
        for idx in range(N):
            mask = torch.ones(N, dtype=torch.bool)
            mask[idx] = False
            remaing_idx[idx] = selected_idx.clone()[mask]
        selected_idx = selected_idx
        return selected_idx.to(device), remaing_idx.to(device)

    def forward_(
        self,
        src_pts,
        tar_pts,
        score,
        relScale,
        relInplane,
    ):
        """
        Finding the mapping from src_pts to tar_pts using RANSAC
        src_pts: (N, 2)
        tar_pts: (N, 2)
        score: (N, )
        relScale: (N, )
        relInplane: (N, 2) # cos, sin
        """
        N_total = src_pts.shape[0]
        device = src_pts.device

        # mutiply by patch_size to get pixel locations
        src_pts = src_pts * self.patch_size # + self.patch_size / 2
        tar_pts = tar_pts * self.patch_size # + self.patch_size / 2

        # sample max_iterations times num_corres correspondences
        train_idx, val_idx = self._sample(N_total, device)

        # (N, 1, 2) (N, N-1, 2), (N, N-1)
        train_src_pts, val_src_pts = (
            src_pts[train_idx],
            src_pts[val_idx],
        )

        train_tar_pts, val_tar_pts = (
            tar_pts[train_idx],
            tar_pts[val_idx],
        )
        val_score = score[val_idx]

        # compute 2D rotation
        if len(relInplane.shape) == 2:
            cos_theta = relInplane[:, 0]
            sin_theta = relInplane[:, 1]
        elif len(relInplane.shape) == 3:
            cos_theta = relInplane[:, :, 0]
            sin_theta = relInplane[:, :, 1]
        else:
            relInplane = relInplane.reshape(-1)
            cos_theta = torch.cos(relInplane)
            sin_theta = torch.sin(relInplane)

        R = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=1)
        R = R.reshape(-1, 2, 2)

        # compute 2D translation
        M_candidates = affine_torch(scale=relScale, rotation=R)  # (N, 3, 3)
        aff_train_src_pts = apply_affine(M_candidates, train_src_pts)
        aff_trans = train_tar_pts - aff_train_src_pts
        # aff_trans[:, [0, 1]] = aff_trans[:, [1, 0]]
        M_candidates[:, :2, 2] = aff_trans

        # compute inliers
        aff_val_src_pts = apply_affine(M_candidates, val_src_pts)
        errors = torch.norm(val_tar_pts - aff_val_src_pts, dim=2)
        inliers = errors <= self.pixel_threshold
        score_inliers = torch.sum(inliers * val_score, dim=1)
        score_cadidate, idx_best = torch.max(score_inliers, dim=0)
        failed = score_cadidate == 0

        # get best M
        mask_inlier = torch.where(inliers[idx_best])[0]
        index_inlier = val_idx[idx_best][mask_inlier]
        return dict(M=M_candidates[idx_best], failed=failed, index_inlier=index_inlier)

    def forward(self, src_pts, tar_pts, relScales, relInplanes, scores=None, direction="src2tar"):
        if scores is None:
            scores = torch.ones_like(src_pts[:, :, 0])

        B, N = src_pts.shape[:2]
        device = src_pts.device
        dtype = src_pts.dtype

        # define inlier points
        inlier_src_pts = torch.full((B, N, 2), -1, dtype=dtype, device=device)
        inlier_scores = torch.full((B, N), 0, dtype=dtype, device=device)
        inlier_tar_pts = inlier_src_pts.clone()

        # best_relScale = torch.zeros(B, device=device)
        # best_relInplane = torch.zeros(B, 2, device=device)
        Ms = torch.eye(3).to(device).unsqueeze(0).repeat(B, 1, 1)
        idx_failed = torch.zeros(B, device=device).bool()

        for b_idx in range(B):
            src_keypoint = src_pts[b_idx]
            tar_keypoint = tar_pts[b_idx]
            score = scores[b_idx]
            relScale = relScales[b_idx]
            relInplane = relInplanes[b_idx]

            # mask = src_keypoint[:, 0] != -1
            mask = torch.any(src_keypoint, dim=1)
            if mask.sum() >= 1:
                m_src_keypoint = src_keypoint[mask]
                m_tar_keypoint = tar_keypoint[mask]
                m_score = score[mask]
                m_relScale = relScale[mask]
                m_relInplane = relInplane[mask]

                # run RANSAC
                output_ransac = self.forward_(
                    src_pts=m_src_keypoint,
                    tar_pts=m_tar_keypoint,
                    score=m_score,
                    relScale=m_relScale,
                    relInplane=m_relInplane,
                )
                # best_relScale[b_idx] = output_ransac["relScale"]
                # best_relInplane[b_idx] = output_ransac["relInplane"]
                Ms[b_idx] = output_ransac["M"]
                idx_failed[b_idx] = output_ransac["failed"]
                # breakpoint()

                # get inlier points
                idx_inlier = output_ransac["index_inlier"]
                inlier_src_pts[b_idx, : len(idx_inlier)] = m_src_keypoint[idx_inlier]
                inlier_tar_pts[b_idx, : len(idx_inlier)] = m_tar_keypoint[idx_inlier]
                inlier_scores[b_idx, : len(idx_inlier)] = m_score[idx_inlier]

        # out_data = PandasTensorCollection(
        #     src_pts=inlier_src_pts,
        #     tar_pts=inlier_tar_pts,
        #     scores=inlier_scores,
        # )
        return dict(Ms=Ms, idx_failed=idx_failed, inlier_src_pts=inlier_src_pts, inlier_tar_pts=inlier_tar_pts, inlier_scores=inlier_scores)


class BatchAffine2DRANSAC(torch.nn.Module):
    def __init__(
        self,
        # template_K,
        # template_Ms,
        # template_poses,
        pixel_threshold=14,
    ):
        super(BatchAffine2DRANSAC, self).__init__()
        # self.template_K = template_K
        # self.template_Ms = template_Ms
        # self.template_poses = template_poses
        self.ransac = RANSAC(pixel_threshold=pixel_threshold)


    def forward_ransac(
        self,
        src_pts,
        tar_pts,
        relScales,
        relInplanes
    ):
        batch_size, k = src_pts.shape[:2]
        device = src_pts.device

        # best_relScales = torch.zeros(batch_size, k, device=device)
        # best_relInplanes = torch.zeros(batch_size, k, 2, device=device)
        # best_pts_idx = torch.zeros(batch_size, k, device=device).long()
        best_Ms = torch.eye(3).to(device).unsqueeze(0).repeat(batch_size, k, 1, 1)
        idx_faileds = torch.zeros(batch_size, k, device=device).bool()
        inlier_src_pts, inlier_tar_pts, inlier_scores = [], [], []

        for idx_k in range(k):
            batch_output_ransac = self.ransac(src_pts=src_pts[:, idx_k],
                                            tar_pts=tar_pts[:, idx_k],
                                            relScales=relScales[:, idx_k],
                                            relInplanes=relInplanes[:, idx_k])
            # best_relScales[:, idx_k] = relScales[batch_output_ransac["pts_idx_best"]]
            # best_relInplanes[:, idx_k] = relInplanes[batch_output_ransac["pts_idx_best"]]
            # best_pts_idx[:, idx_k] = batch_output_ransac["pts_idx_best"]
            best_Ms[:, idx_k] = batch_output_ransac["Ms"]
            idx_faileds[:, idx_k] = batch_output_ransac["idx_failed"]

            inlier_src_pts.append(batch_output_ransac["inlier_src_pts"])
            inlier_tar_pts.append(batch_output_ransac["inlier_tar_pts"])
            inlier_scores.append(batch_output_ransac["inlier_scores"])
        
        # for idx_k in range(k):
        #     inlier_src_pts.append(tmp[idx_k].src_pts)
        #     inlier_tar_pts.append(tmp[idx_k].tar_pts)
        #     inlier_scores.append(tmp[idx_k].scores)
        inlier_src_pts = torch.stack(inlier_src_pts, dim=1)
        inlier_tar_pts = torch.stack(inlier_tar_pts, dim=1)
        inlier_scores = torch.stack(inlier_scores, dim=1)

        return dict(best_Ms=best_Ms, inlier_scores=inlier_scores, inlier_src_pts=inlier_src_pts, inlier_tar_pts=inlier_tar_pts)
