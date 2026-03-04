import numpy as np
import torch

def _to_torch_boxes(boxes, device):
    """
    boxes: numpy array [N, 5] -> torch float32 [N, 5] on device
    """
    if isinstance(boxes, np.ndarray):
        t = torch.from_numpy(boxes.astype(np.float32, copy=False))
    else:
        t = boxes.float()
    return t.to(device)


def _rbbox_to_corners(boxes):
    """
    boxes: [N, 5] -> corners: [N, 4, 2]
    format: [cx, cy, w, h, angle] (angle in radians, CCW)
    """
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w  = boxes[:, 2]
    h  = boxes[:, 3]
    a  = boxes[:, 4]

    cos_a = torch.cos(a)
    sin_a = torch.sin(a)

    # four corners in local box coordinates (before rotation)
    # (±w/2, ±h/2)
    wx = w / 2.0
    hy = h / 2.0

    # base corners: (x, y) pairs
    # order: bottom-left, bottom-right, top-right, top-left
    corners_local = torch.stack([
        torch.stack([-wx, -hy], dim=-1),
        torch.stack([ wx, -hy], dim=-1),
        torch.stack([ wx,  hy], dim=-1),
        torch.stack([-wx,  hy], dim=-1),
    ], dim=1)  # [N, 4, 2]

    # rotation matrix per box
    # [ [cos, -sin],
    #   [sin,  cos] ]
    rot = torch.zeros(boxes.size(0), 2, 2, device=boxes.device, dtype=boxes.dtype)
    rot[:, 0, 0] = cos_a
    rot[:, 0, 1] = -sin_a
    rot[:, 1, 0] = sin_a
    rot[:, 1, 1] = cos_a

    # apply rotation
    corners_rot = torch.matmul(corners_local, rot.transpose(1, 2))  # [N, 4, 2]

    # translate by center
    centers = torch.stack([cx, cy], dim=-1).unsqueeze(1)  # [N, 1, 2]
    corners = corners_rot + centers  # [N, 4, 2]

    return corners


def _polygon_area(polygon):
    """
    polygon: [M, 2] tensor (x, y)
    returns scalar area (tensor)
    """
    if polygon.size(0) < 3:
        return polygon.new_tensor(0.0)

    x = polygon[:, 0]
    y = polygon[:, 1]
    # shoelace formula
    area = 0.5 * torch.abs(torch.sum(x * y.roll(-1) - y * x.roll(-1)))
    return area


def _clip_polygon(subject, clip):
    """
    Sutherland–Hodgman polygon clipping.
    subject: [M, 2]
    clip: [K, 2]
    returns: [P, 2] intersection polygon
    """
    def inside(p, edge_start, edge_end):
        # z-component of cross((edge_end - edge_start), (p - edge_start)) >= 0
        return ((edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) -
                (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0])) >= 0

    def compute_intersection(p1, p2, e1, e2):
        # line p1->p2 and e1->e2 intersection
        # using parametric form
        A = p2 - p1
        B = e2 - e1
        C = e1 - p1

        denom = A[0] * B[1] - A[1] * B[0]
        # parallel lines -> no unique intersection (should not happen often)
        if torch.abs(denom) < 1e-7:
            return p1
        t = (C[0] * B[1] - C[1] * B[0]) / denom
        return p1 + t * A

    output = subject
    if output.size(0) == 0:
        return output

    for i in range(clip.size(0)):
        input_list = output
        output = []

        A = clip[i]
        B = clip[(i + 1) % clip.size(0)]

        if input_list.size(0) == 0:
            break

        S = input_list[-1]
        for E in input_list:
            if inside(E, A, B):
                if not inside(S, A, B):
                    output.append(compute_intersection(S, E, A, B))
                output.append(E)
            elif inside(S, A, B):
                output.append(compute_intersection(S, E, A, B))
            S = E

        if len(output) == 0:
            return subject.new_zeros((0, 2))
        output = torch.stack(output, dim=0)

    return output


def _single_pair_iou(corners1, corners2):
    """
    corners1: [4, 2], corners2: [4, 2]
    returns scalar IoU (tensor)
    """
    area1 = _polygon_area(corners1)
    area2 = _polygon_area(corners2)
    if area1 <= 0.0 or area2 <= 0.0:
        return corners1.new_tensor(0.0)

    inter_poly = _clip_polygon(corners1, corners2)
    inter_area = _polygon_area(inter_poly)

    union = area1 + area2 - inter_area
    if union <= 0.0:
        return corners1.new_tensor(0.0)
    return inter_area / union


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """
    Rotated box IoU using PyTorch on GPU.
    Replacement for the old Numba-based version.

    Args:
        boxes: numpy array [N, 5] (cx, cy, w, h, angle)
        query_boxes: numpy array [K, 5]
        criterion: kept for API compatibility (currently unused, IoU only)
        device_id: CUDA device index (default: 0)

    Returns:
        iou: numpy array [N, K], same dtype as input boxes
    """
    box_dtype = boxes.dtype
    boxes = boxes.astype(np.float32, copy=False)
    query_boxes = query_boxes.astype(np.float32, copy=False)

    N = boxes.shape[0]
    K = query_boxes.shape[0]
    if N == 0 or K == 0:
        return np.zeros((N, K), dtype=box_dtype)

    if not torch.cuda.is_available():
        raise RuntimeError("rotate_iou_gpu_eval: CUDA is not available for PyTorch.")

    device = torch.device(f"cuda:{device_id}")

    # to torch on GPU
    t_boxes = _to_torch_boxes(boxes, device)
    t_qboxes = _to_torch_boxes(query_boxes, device)

    # If your angles are in degrees, uncomment this:
    # t_boxes[:, 4] = t_boxes[:, 4] * torch.pi / 180.0
    # t_qboxes[:, 4] = t_qboxes[:, 4] * torch.pi / 180.0

    corners1 = _rbbox_to_corners(t_boxes)     # [N, 4, 2]
    corners2 = _rbbox_to_corners(t_qboxes)    # [K, 4, 2]

    iou = torch.zeros((N, K), device=device, dtype=torch.float32)

    # NOTE: this is a double loop in Python. For very large N, K this can be slow.
    # It’s still GPU-based geometry, just not fully vectorized.
    for i in range(N):
        c1 = corners1[i]
        for j in range(K):
            c2 = corners2[j]
            iou[i, j] = _single_pair_iou(c1, c2)

    # back to numpy, original dtype
    return iou.cpu().numpy().astype(box_dtype, copy=False)
