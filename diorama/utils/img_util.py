from PIL import Image
import numpy as np


def resize_padding(im, desired_size, mode="RGBA", background_color=(0, 0, 0), padding=0):
    """
    Args:
        im (pillow image object): the image to be resized
        desired_size (int): the side length of resized image
        mode (string): image mode for creating PIL Image object

    Returns:
        pillow image object: resized image
    """
    # compute the new size
    old_size = im.size
    if max(old_size) == 0:
        return None
    double_size = [s*2 for s in old_size]
    unpad_size = desired_size - padding*2
    tgt_size = max(double_size) if max(double_size) < unpad_size else unpad_size
    ratio = float(tgt_size) / max(old_size)
    # new_size = tuple(int(x * ratio) for x in old_size)
    w, h = [int(x * ratio) for x in old_size]
    w = w if w >= 2 else 2
    h = h if h >= 2 else 2
    new_size = (w, h)

    # create a new image and paste the resized on it
    if mode == "L":
        im = im.resize(new_size, Image.NEAREST)
        new_im = Image.new(mode, (desired_size, desired_size))
    elif mode in ["RGB", "RGBA"]:
        im = im.resize(new_size)
        new_im = Image.new(mode, (desired_size, desired_size), color=background_color)
    new_im.paste(im,
                ((desired_size - new_size[0]) // 2,
                (desired_size - new_size[1]) // 2))
    return new_im, new_size


def box_ios_batch(boxes: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Self (IoS) of a set of bounding boxes in `(x_min, y_min, x_max, y_max)` format.

    Args:
        boxes (np.ndarray): 2D `np.ndarray` representing boxes.
            `shape = (N, 4)` where `N` is number of true objects.

    Returns:
        np.ndarray: Pairwise IoS of boxes. `shape = (N, N)` where `N` is number of boxes.
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area = box_area(boxes.T)
    top_left = np.maximum(boxes[:, None, :2], boxes[:, :2])
    bottom_right = np.minimum(boxes[:, None, 2:], boxes[:, 2:])
    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    
    return area_inter / area[:, None].T


def multiple_instances_suppression(predictions: np.ndarray, ios_threshold: float = 0.8) -> np.ndarray:
    """
    Perform Multiple Instances Suppression (MIS) on object detection predictions.

    Args:
        predictions (np.ndarray): An array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`.
        ios_threshold (float, optional): The intersection-over-self threshold
            to use for multiple instances suppression.

    Returns:
        np.ndarray: A boolean array indicating which predictions to keep after multiple instances suppression.
    """
    rows, columns = predictions.shape

    # add column #5 - category filled with zeros for agnostic mis
    if columns == 5:
        predictions = np.c_[predictions, np.zeros(rows)]

    # sort predictions column #4 - score
    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    categories = predictions[:, 5]
    ioss = box_ios_batch(boxes)
    ioss = ioss - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, (ios, category) in enumerate(zip(ioss, categories)):
        if not keep[index]:
            continue
        # drop detections with ios > ios_threshold and same category as current detections
        condition = (ios > ios_threshold) & (categories == category)
        keep = keep & ~condition

    return keep[sort_index.argsort()]