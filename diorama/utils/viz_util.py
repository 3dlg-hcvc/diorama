import tempfile

import numpy as np
from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines


# RGB:
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)

# https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/colormap.py
def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def denorm_torch_to_pil(image):
    image_norm_mean = (0.485, 0.456, 0.406)
    image_norm_std = (0.229, 0.224, 0.225)
    image = image * torch.Tensor(image_norm_std)[:, None, None]
    image = image + torch.Tensor(image_norm_mean)[:, None, None]
    return Image.fromarray((image.permute(1, 2, 0) * 255).numpy().astype(np.uint8))


def fig_to_pil(fig):
    tmp_file = tempfile.SpooledTemporaryFile(max_size=10*1024*1024) # 10MB 
    fig.savefig(tmp_file, bbox_inches='tight', pad_inches=0)
    image = Image.open(tmp_file)
    image.load()
    tmp_file.close()
    return image


def fig_to_array(fig):
    return np.array(fig_to_pil(fig))


def arrange(images):
    rows = []
    for row in images:
        rows += [np.concatenate(row, axis=1)]
    image = np.concatenate(rows, axis=0)
    return image


def tile_ims_horizontal_highlight_best(ims, gap_px=20, highlight_idx=None):
    cumul_offsets = [0]
    for im in ims:
        cumul_offsets.append(cumul_offsets[-1]+im.width+gap_px)
    max_h = max([im.height for im in ims])
    dst = Image.new('RGB', (cumul_offsets[-1], max_h), (255, 255, 255))
    for i, im in enumerate(ims):
        dst.paste(im, (cumul_offsets[i], (max_h - im.height) // 2))
        
        if i == highlight_idx:
            img1 = ImageDraw.Draw(dst)  
            # shape is defined as [(x1,y1), (x2, y2)]
            shape = [(cumul_offsets[i],(max_h - im.height) // 2), 
                     (cumul_offsets[i]+im.width, max_h-(max_h - im.height) // 2)]
            img1.rectangle(shape, fill = None, outline ="green", width=6)

    return dst


def get_concat_h_cut_center(im1, im2, gap_px=20):
    dst = Image.new('RGB', (im1.width + im2.width + gap_px, min(im1.height, im2.height)), (255, 255, 255))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width + gap_px, (im1.height - im2.height) // 2))
    return dst


def draw_correspondences_lines(points1, points2, image1, image2, ax):
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :param ax: a matplotlib axis object
    :return: the matplotlib axis.
    """
    gap = 20

    im = np.array(get_concat_h_cut_center(image1, image2, gap))
    ax.imshow(im)
    ax.axis('off')
    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)
    cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                       "maroon", "black", "white", "chocolate", "gray", "blueviolet"]*(1+num_points//15))
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 6, 2
    points2 += np.array([0, gap+image1.width])
    for point1, point2, color in zip(points1, points2, colors):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        ax.add_patch(circ1_1)
        ax.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor='white')
        ax.add_patch(circ2_1)
        ax.add_patch(circ2_2)
        l = mlines.Line2D([x1,x2], [y1,y2], c=color, linewidth=0.75)
        ax.add_line(l)
        ax.plot(x1, y1, x2, y2, linestyle='-', c='w')
    return ax
