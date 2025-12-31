from typing import List, Tuple
import numpy as np
import torch


def _to_cartesian(coords: torch.Tensor, shape: Tuple):

    """
    Takes raveled coordinates and returns them in a cartesian coordinate frame
    coords: B x D
    shape: tuple of cartesian dimensions
    return: B x D x 2
    """
    i, j = (torch.from_numpy(inds) for inds in np.unravel_index(coords.cpu(), shape=shape))
    return torch.stack([i, j], dim=-1)


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


# -----------------
# BATCHED VERSION OF CORRESPONDENCE FUNCTION
# Uses cyclical distances rather than mutual nearest neighbours
# -----------------
def find_correspondences_batch(descriptors1, descriptors2, mask1, mask2, num_pairs: int = 10, sim_threshold: float = 0.5, device : torch.device = torch.device('cpu')):
    """
    Finding point correspondences between two images.
    Legend: B: batch, T: total tokens (num_patches ** 2 + 1), D: Descriptor dim per head, H: Num attention heads

    Method: Compute similarity between all pairs of pixel descriptors
            Find nearest neighbours from Image1 --> Image2, and Image2 --> Image1
            Use nearest neighbours to define a cycle from Image1 --> Image2 --> Image1
            Take points in Image1 (and corresponding points in Image2) which have smallest 'cycle distance'
            Also, filter examples which aren't part of the foreground in both images, as determined by masks

    :param descriptors1: ViT features of shape B x 1 x (T - 1) x D * H (i.e, no CLS token)
    :param descriptors2: ViT features of shape B x 1 x (T - 1) x D * H (i.e, no CLS token)
    :param mask1: Mask of shape B x (T - 1) indicating foreground pixels
    :param mask2: Mask of shape B x (T - 1) indicating foreground pixels
    :param num_pairs: number of outputted corresponding pairs.
    """
    # extracting descriptors for each image
    B, _, t_m_1, d_h = descriptors1.size()
    fg_mask1 = mask1 # B x T - 1
    fg_mask2 = mask2 # B x T - 1

    # Hard code
    num_patches1 = (int(np.sqrt(t_m_1)), int(np.sqrt(t_m_1)))
    inf_idx = int(t_m_1)

    # COMPUTE MUTUAL NEAREST NEIGHBOURS
    similarities = chunk_cosine_sim(descriptors1, descriptors2)
    sim_1, nn_1 = torch.max(similarities, dim=-1, keepdim=False)  # nn_1 - indices of desc2 closest to desc1. B x T - 1
    sim_2, nn_2 = torch.max(similarities, dim=-2, keepdim=False)  # nn_2 - indices of desc1 closest to desc2. B x T - 1
    nn_1, nn_2 = nn_1[:, 0, :], nn_2[:, 0, :]

    # Map nn_2 points which are not highlighed by fg_mask to 0
    nn_2[~fg_mask2] = 0     # TODO: Note, this assumes top left pixel is never a point of interest
    cyclical_idxs = torch.gather(nn_2, dim=-1, index=nn_1)  # Intuitively, nn_2[nn_1]

    # prepare cartesian coordinates
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1])[None, :].repeat(B, 1)
    cyclical_idxs_ij = _to_cartesian(cyclical_idxs, shape=num_patches1).to(device)
    image_idxs_ij = _to_cartesian(image_idxs, shape=num_patches1).to(device)

    # Find which points are mapped to 0, artificially map them to a high value
    cyclical_idxs_ij[cyclical_idxs_ij == 0] = inf_idx

    # compute distance between cyclical point and original point
    # View to make sure PairwiseDistance behaviour is consistent across torch versions
    b, hw, ij_dim = cyclical_idxs_ij.shape
    cyclical_dists = torch.nn.PairwiseDistance(p=2)(cyclical_idxs_ij.view(-1, ij_dim), image_idxs_ij.view(-1, ij_dim))
    cyclical_dists = cyclical_dists.view(b, hw)
    cyclical_dists[~fg_mask1] = inf_idx

    # Find the TopK points in Image1 and their correspondences in Image2
    sorted_vals, sorted_points_desc1 = cyclical_dists.sort(dim=-1, descending=False)
    corresp_points_desc1 = torch.zeros((b, num_pairs)).type_as(sorted_points_desc1)
    for i in range(b):
        num_corresp = min((sorted_vals[i] < 2).sum(), num_pairs)
        corresp_points_desc1[i, :num_corresp] = sorted_points_desc1[i, :num_corresp]
    corresp_points_mask = corresp_points_desc1 != 0
    corresp_points_desc2 = torch.gather(nn_1, dim=-1, index=corresp_points_desc1)
    # corresp_points_desc2[~corresp_points_mask] = 0

    # Compute the descriptor distances of the selected points
    sim_selected_12 = torch.gather(sim_1[:, 0, :], dim=-1, index=corresp_points_desc1.to(device))
    sim_selected_mask = sim_selected_12 >= sim_threshold
    corresp_points_mask = corresp_points_mask & sim_selected_mask
    
    corresp_points_desc1[~corresp_points_mask] = 0
    corresp_points_desc2[~corresp_points_mask] = 0
    sim_selected_12[~corresp_points_mask] = 0

    # Convert to cartesian coordinates
    corresp_points_desc1, corresp_points_desc2 = (_to_cartesian(inds, shape=num_patches1) for inds in (corresp_points_desc1, corresp_points_desc2))
    corresp_points_mask = corresp_points_mask.cpu()
    cyclical_dists = cyclical_dists.reshape(-1, num_patches1[0], num_patches1[1])

    return corresp_points_desc1, corresp_points_desc2, corresp_points_mask, cyclical_dists, sim_selected_12
