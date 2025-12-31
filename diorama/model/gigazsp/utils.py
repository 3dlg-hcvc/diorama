import numpy as np
import cv2
import torch
import torch.nn.functional as F


# ----------------
# DATA AUGMENTATION
# ----------------
class CropResizePad:
    def __init__(self, target_size=224, patch_size=14):
        self.target_size = target_size
        self.patch_size = patch_size

    def __call__(self, tensors, xyxy_boxes):
        """_summary_

        Args:
            tensors (_type_): (B, C, H, W)
            xyxy_boxes (_type_): (B, 4)

        Returns:
            _type_: _description_
        """
        batch_size = xyxy_boxes.shape[0]
        device = xyxy_boxes.device
        bbox_sizes = torch.zeros((batch_size, 2), dtype=torch.int32)
        bbox_sizes[:, 0] = xyxy_boxes[:, 2] - xyxy_boxes[:, 0]
        bbox_sizes[:, 1] = xyxy_boxes[:, 3] - xyxy_boxes[:, 1]
        scales = self.target_size / torch.max(bbox_sizes, dim=-1)[0]
        
        Ms, crop_resize_images = [], []
        for i in range(batch_size):
            tensor = tensors[i]
            bbox, bbox_size, scale = xyxy_boxes[i], bbox_sizes[i], scales[i]

            M_crop = torch.eye(3, device=device)
            M_resize_pad = torch.eye(3, device=device)

            # crop and scale
            tensor = tensor[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]
            M_crop[:2, 2] = -bbox[:2]
            # M_crop[0, 2], M_crop[1, 2] = -bbox[1], -bbox[0]

            tensor = F.interpolate(tensor.unsqueeze(0), scale_factor=scale.item())[0]
            M_resize_pad[:2, :2] *= scale

            if tensor.shape[-1] / tensor.shape[-2] != 1:
                pad_top = (self.target_size - tensor.shape[-2]) // 2
                pad_bottom = self.target_size - tensor.shape[-2] - pad_top
                pad_bottom = max(pad_bottom, 0)

                pad_left = (self.target_size - tensor.shape[-1]) // 2
                pad_left = max(pad_left, 0)
                pad_right = self.target_size - tensor.shape[-1] - pad_left

                tensor = F.pad(tensor, [pad_left, pad_right, pad_top, pad_bottom])
                M_resize_pad[:2, 2] = torch.tensor([pad_left, pad_top])
                # M_resize_pad[:2, 2] = torch.tensor([pad_top, pad_left])

            M = torch.matmul(M_resize_pad, M_crop)

            # sometimes, 1 pixel is missing due to rounding, so interpolate again
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=(self.target_size, self.target_size)
            )[0]

            Ms.append(M)
            crop_resize_images.append(tensor)

        Ms = torch.stack(Ms)
        crop_resize_images = torch.stack(crop_resize_images)
        return crop_resize_images, Ms

    def forward_image_wrap(self, images, M):
        images_np = images.permute(0, 2, 3, 1).cpu().numpy()
        M_np = M.cpu().numpy()
        new_images = [
            cv2.warpAffine(
                images_np[i], M_np[i][:2, :], (self.target_size, self.target_size)
            )
            for i in range(len(images))
        ]
        assert len(new_images) != 0, f"Issue with warpAffine: {new_images}"
        new_images = torch.from_numpy(np.stack(new_images)).to(images.device)
        return new_images.permute(0, 3, 1, 2).float()


# ----------------
# TRANSFORMATIONS
# ----------------
def affine_torch(rotation, scale=None, translation=None):
    if len(rotation.shape) == 2:
        """
        Create 2D affine transformation matrix
        """
        M = torch.eye(3, device=scale.device, dtype=scale.dtype)
        M[:2, :2] = rotation
        if scale is not None:
            M[:2, :2] *= scale
        if translation is not None:
            M[:2, 2] = translation
        return M
    else:
        Ms = torch.eye(3, device=scale.device, dtype=scale.dtype)
        Ms = Ms.unsqueeze(0).repeat(rotation.shape[0], 1, 1)
        Ms[:, :2, :2] = rotation
        if scale is not None:
            Ms[:, :2, :2] *= scale.unsqueeze(1).unsqueeze(1)
        if translation is not None:
            Ms[:, :2, 2] = translation
        return Ms


def homogenuous(pixel_points):
    """
    Convert pixel coordinates to homogenuous coordinates
    """
    device = pixel_points.device
    if len(pixel_points.shape) == 2:
        one_vector = torch.ones(pixel_points.shape[0], 1).to(device)
        return torch.cat([pixel_points, one_vector], dim=1)
    elif len(pixel_points.shape) == 3:
        one_vector = torch.ones(pixel_points.shape[0], pixel_points.shape[1], 1).to(
            device
        )
        return torch.cat([pixel_points, one_vector], dim=2)
    else:
        raise NotImplementedError


def inverse_affine(M):
    """
    Inverse 2D affine transformation matrix of cropping
    """
    if len(M.shape) == 2:
        M = M.unsqueeze(0)
    if len(M.shape) == 3:
        assert (M[:, 1, 0] == 0).all() and (M[:, 0, 1] == 0).all()
        assert (M[:, 0, 0] == M[:, 1, 1]).all(), f"M: {M}"

        scale = M[:, 0, 0]
        M_inv = torch.eye(3, device=M.device, dtype=M.dtype)
        M_inv = M_inv.unsqueeze(0).repeat(M.shape[0], 1, 1)
        M_inv[:, 0, 0] = 1 / scale  # scale
        M_inv[:, 1, 1] = 1 / scale  # scale
        M_inv[:, :2, 2] = -M[:, :2, 2] / scale.unsqueeze(1)  # translation
    else:
        raise ValueError("M must be 2D or 3D")
    return M_inv


def apply_affine(M, points):
    """
    M: (N, 3, 3)
    points: (N, 2)
    """
    if len(points.shape) == 2:
        transformed_points = torch.einsum(
            "bhc,bc->bh",
            M,
            homogenuous(points),
        )  # (N, 3)
        transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]
    elif len(points.shape) == 3:
        transformed_points = torch.einsum(
            "bhc,bnc->bnh",
            M,
            homogenuous(points),
        )
        transformed_points = transformed_points[:, :, :2] / transformed_points[:, :, 2:]
    else:
        raise NotImplementedError
    return transformed_points


# ----------------
# NORMALIZED CYCLICAL DISTANCES
# ----------------
def normalize_cyclical_dists(dists):

    # Assume dists are in the format B x H x W
    b, h, w = dists.size()
    dists = dists.view(b, h * w)

    # Normalize to [0, 1]
    dists -= dists.min(dim=-1)[0][:, None]
    dists /= dists.max(dim=-1)[0][:, None]

    # Hack to find the mininimum non-negligible value
    dists -= 0.05
    dists[dists < 0] = 3

    # Re-Normalize to [0, 1]
    dists -= dists.min(dim=-1)[0][:, None]
    dists[dists > 1] = 0
    dists /= dists.max(dim=-1)[0][:, None]

    dists = dists.view(b, h, w)

    return dists


# ----------------
# INTERSECTION OVER UNION
# ----------------
def batch_intersection_over_union(tensor_a, tensor_b, threshold=None):

    """
    a is B x H x W
    b is B x H x W
    """

    intersection = (tensor_a * tensor_b).sum(dim=(-1, -2))
    union = (tensor_a + tensor_b).sum(dim=(-1, -2))
    iou = 2 * (intersection / union)

    if threshold is None:

        return iou  # Return shape (B,)

    else:

        tensor_a[tensor_a < threshold] = 0
        tensor_b[tensor_b < threshold] = 0

        tensor_a[tensor_a >= threshold] = 1
        tensor_b[tensor_b >= threshold] = 1

        intersection_map = tensor_a * tensor_b

        return iou, intersection_map


# ----------------
# SCALE BACK POINTS
# ----------------
def scale_points_from_patch(points, vit_image_size=224, num_patches=28):
    points = (points + 0.5) / num_patches * vit_image_size
    
    return points


def scale_points_to_orig(points, image_scaling):
    points *= image_scaling

    return points.int().long()


# ---------------
# RIGID BODY TRANSFORM
# ---------------
def least_squares_solution(x1, x2):
    """
    x1, x2 both define i=1,...,I 3-dimensional points, with known correspondence
    https://www.youtube.com/watch?v=re08BUepRxo
    Assumption: points x1, x2 related by a similarity:
      E(x2_i) = λ*R*x1_i+t, {i = 1, ..., I}
    Task: estimate the parameters, provide uncertainty

    Args:
        x1 (array): Shape (3, N)
        x2 (array): Shape (3, N)

    Returns:
        rot (array): Shape (3, 3), a rotation matrix
        t (array):   Shape (3, 1), a translation vector
        lam (float): Scaling factor

    """
    w1 = (1/0.1**2) * np.ones(x1.shape[1]) # 'weights' are (1/sig^2) - fix for now
    w2 = (1/0.1**2) * np.ones(x2.shape[1]) # 'weights' are (1/sig^2) - fix for now
    # Find centroid (weighted) of the 'observed' (x2) points
    x2_C = (np.sum(x2*w2, axis=1) / np.sum(w2)).reshape(3,1)
    x1_C = (np.sum(x1*w1, axis=1) / np.sum(w1)).reshape(3,1)

    # Unknown params: rotation R, scale λ, modified translation vector u, residuals v_x2_i
    # t = x2_C - λ*R*u

    # Minimising the weighted sum of the residuals, we arrive at
    u = x1_C

    # Approximate solution for lambda, holds for small noise - analytic soln is dependent on R!
    def get_sums(x, xc, w):
        # total = 0
        # for x_i, x_ci, w_i in zip(x, xc, w):
        #     total += w_i * (x_i-x_ci).T @ (x_i-x_ci)
        # return total
        # Or, parallel implementation!
        return np.sum(np.sum((x - xc) * (x - xc), axis=0) * w)
    
    lam_sq = get_sums(x2, x2_C, w2) / get_sums(x1, x1_C, w2)
    lam = np.sqrt(lam_sq)
    # lam = 1

    # Estimation of rotation
    # 1. Centre coordinates
    c_x1 = x1 - x1_C
    c_x2 = x2 - x2_C
    # 2. Create H matrix (3x3)
    H = c_x1 @ np.diag(w2) @ c_x2.T
    # 3. Use SVD to find estimated rotation R
    U, S, Vh = np.linalg.svd(H)
    R = Vh.T @ U.T

    # Get translation parameter using t = x2_C - λ*R*u
    # import pdb
    # pdb.set_trace()
    t = x2_C - lam * R @ u
    return R, t, lam


def rigid_transform_3D(A, B):
    """
    Un-weighted version, from https://github.com/nghiaho12/rigid_transform_3D

    Implementation of "Least-Squares Fitting of Two 3-D Point Sets", Arun, K. S. 
    and Huang, T. S. and Blostein, S. D, [1987]
    """
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    print("R before correcting for reflection:")
    print(R)
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def umeyama(src, dst, estimate_scale=True):
    """Estimate N-D similarity transformation with or without scaling.
    Taken from skimage!

    homo_src = np.hstack((src, np.ones((len(src), 1))))
    homo_dst = np.hstack((src, np.ones((len(src), 1))))

    homo_dst = T @ homo_src, where T is the returned transformation

    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.eye(4), 1
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    
    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    
    return T, scale

    # R = T[:dim, :dim]
    # t = T[:dim, dim:]
    # return R, t, scale


class RigidBodyTransform():

    def estimate(self, world_corr1, world_corr2):
        self.R, self.t, self.lam = least_squares_solution(world_corr1.T, world_corr2.T)

    def residuals(self, world_corr1, world_corr2):
        # E(x2_i) = λ*R*x1_i+t, {i = 1, ..., I}
        world_corr2_est = self.transform(world_corr1)
        res = torch.nn.PairwiseDistance(p=2)(torch.Tensor(world_corr2_est),
                                             torch.Tensor(world_corr2))
        return res.numpy()

    def transform(self, world_corr1):
        return (self.lam * self.R @ world_corr1.T + self.t).T


class RigidBodyUmeyama():
    
    def estimate(self, world_corr1, world_corr2):
        self.T, self.lam = umeyama(world_corr1, world_corr2)
        
    def residuals(self, world_corr1, world_corr2):
        world_corr2_est = self.transform(world_corr1)
        res = torch.nn.PairwiseDistance(p=2)(torch.Tensor(world_corr2_est),
                                             torch.Tensor(world_corr2))
        return res.numpy()
    
    def transform(self, world_corr1):
        w1_homo = np.vstack((world_corr1.T, np.ones((1, (len(world_corr1))))))
        transformed = self.T @ w1_homo
        return (transformed[:3, :]).T
    

def ransac_sample_validation(world_corr1, world_corr2):
    # enforce 1-1 correspondence between world_corr1 and world_corr2
    pair_dist1 = torch.cdist(torch.Tensor(world_corr1), torch.Tensor(world_corr1))
    valid1 = torch.all((pair_dist1 < 1e-6).sum(-1) <= 1)
    pair_dist2 = torch.cdist(torch.Tensor(world_corr2), torch.Tensor(world_corr2))
    valid2 = torch.all((pair_dist2 < 1e-6).sum(-1) <= 1)
    
    return valid1 & valid2
