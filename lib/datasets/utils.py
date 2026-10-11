''' some auxiliary functions for all datasets '''
import numpy as np
import cv2


num_heading_bin = 12  # hyper param

def angle2class(angle):
    ''' Convert continuous angle to discrete class and residual. '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class2angle(cls, residual, to_label_format=False):
    ''' Inverse function to angle2class. '''
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def gaussian_radius(bbox_size, min_overlap=0.7):
    height, width = bbox_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


def draw_projected_box3d(image, corners2d, color=(255, 255, 255), thickness=1):
    """Draw a box from 8 projected corners or a projected edge path."""
    corners2d = np.asarray(corners2d)
    if corners2d.shape == (8, 2):
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
    elif corners2d.ndim == 2 and corners2d.shape[1] == 2 and corners2d.shape[0] >= 2:
        # project_3d returns vertices in an order where each adjacent pair is
        # one box edge.
        edges = tuple((i, i + 1) for i in range(corners2d.shape[0] - 1))
    else:
        raise ValueError(
            f"Expected projected corners with shape (8,2) or an edge path, got {corners2d.shape}"
        )

    finite = np.isfinite(corners2d).all(axis=1)
    corners2d = np.where(finite[:, None], corners2d, 0).astype(np.int32)
    for start, end in edges:
        if not finite[start] or not finite[end]:
            continue
        cv2.line(
            image,
            tuple(corners2d[start]),
            tuple(corners2d[end]),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
    return image


def draw_calibrated_box3d(
    image,
    corners3d,
    calib,
    color=(255, 255, 255),
    thickness=1,
    *,
    scale=1.0,
    offset_x=0.0,
    offset_y=0.0,
    flip=False,
):
    """Project camera-frame corners through calibration and draw the box.

    ``rect_to_img`` applies the calibration's lens distortion. The optional
    scale, offsets, and flip map projected source-image points into a
    resized/padded or horizontally reflected image.
    """
    corners2d, _ = calib.rect_to_img(np.asarray(corners3d, dtype=np.float32))
    corners2d = np.asarray(corners2d, dtype=np.float32)
    corners2d[:, 0] = corners2d[:, 0] * scale + offset_x
    corners2d[:, 1] = corners2d[:, 1] * scale + offset_y
    if flip:
        corners2d[:, 0] = image.shape[1] - corners2d[:, 0]
    return draw_projected_box3d(image, corners2d, color=color, thickness=thickness)
