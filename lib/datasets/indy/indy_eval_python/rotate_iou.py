import numpy as np
import torch

def _to_torch_boxes(boxes, device):
    """Convert numpy [N,5] to torch [N,5] on device."""
    if isinstance(boxes, np.ndarray):
        t = torch.from_numpy(boxes.astype(np.float32, copy=False))
    else:
        t = boxes.float()
    return t.to(device)


def _rbbox_to_corners_numba_compatible(boxes):
    """
    boxes: [N,5] -> corners: [N,4,2]
    Format: [cx, cy, w, h, angle], angle > 0 = CLOCKWISE
    Corner order: BL, TL, TR, BR  (same as rbbox_to_corners)
    """
    device = boxes.device
    dtype = boxes.dtype

    cx = boxes[:, 0:1]  # [N,1]
    cy = boxes[:, 1:2]
    w  = boxes[:, 2:2+1]
    h  = boxes[:, 3:3+1]
    a  = boxes[:, 4:4+1]

    a_cos = torch.cos(a)
    a_sin = torch.sin(a)

    # local box corners: BL, TL, TR, BR
    wx = w / 2.0
    hy = h / 2.0

    # corners_x/y: [N,4]
    corners_x = torch.cat([-wx, -wx,  wx,  wx], dim=1)
    corners_y = torch.cat([-hy,  hy,  hy, -hy], dim=1)

    # apply CLOCKWISE rotation (same as Numba):
    # x' = cos(a) * x + sin(a) * y
    # y' = -sin(a) * x + cos(a) * y
    x_rot = a_cos * corners_x + a_sin * corners_y
    y_rot = -a_sin * corners_x + a_cos * corners_y

    # translate by center
    x_world = x_rot + cx
    y_world = y_rot + cy

    # corners = torch.stack([x_world, y_world], dim=-1)  # [N,4,2]
    corners = torch.stack([x_world, y_world], dim=-1)

    # Convert clockwise → counter-clockwise
    # old order: [BL, TL, TR, BR] = [0,1,2,3] (CW)
    # new order: [BL, BR, TR, TL] = [0,3,2,1] (CCW)
    corners = corners[:, [0, 3, 2, 1], :]

    return corners.to(device=device, dtype=dtype)


def _polygon_area(pts):
    """
    pts: [M,2] tensor
    area via shoelace formula (abs, orientation-independent)
    """
    if pts.size(0) < 3:
        return pts.new_tensor(0.0)

    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * torch.abs(torch.sum(x * y.roll(-1) - y * x.roll(-1)))


def _sutherland_hodgman_clip(subject, clip):
    """
    Generic convex polygon clipping (Sutherland–Hodgman).
    subject: [M,2], clip: [K,2]
    returns intersection polygon [P,2]
    """

    def inside(p, a, b):
        # keep points on the left of edge a->b
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersect(p1, p2, a, b):
        # intersection of segments p1->p2 and a->b
        A = p2 - p1
        B = b - a
        C = a - p1
        denom = A[0] * B[1] - A[1] * B[0]
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
                    output.append(intersect(S, E, A, B))
                output.append(E)
            elif inside(S, A, B):
                output.append(intersect(S, E, A, B))
            S = E

        if len(output) == 0:
            return subject.new_zeros((0, 2))
        output = torch.stack(output, dim=0)

    return output


def _pair_measure(c1, c2, box1, box2, criterion=-1):
    """
    c1, c2: [4,2] corners of two boxes
    box1, box2: [5] [cx,cy,w,h,angle]
    criterion: as in devRotateIoUEval
    """
    inter_poly = _sutherland_hodgman_clip(c1, c2)
    inter_area = _polygon_area(inter_poly)

    area1 = box1[2] * box1[3]  # w*h
    area2 = box2[2] * box2[3]

    if criterion == -1:
        denom = area1 + area2 - inter_area
        return inter_area / denom if denom > 0 else c1.new_tensor(0.0)
    elif criterion == 0:
        return inter_area / area1 if area1 > 0 else c1.new_tensor(0.0)
    elif criterion == 1:
        return inter_area / area2 if area2 > 0 else c1.new_tensor(0.0)
    else:
        return inter_area


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """
    PyTorch CUDA replacement for Numba rotate_iou_gpu_eval.

    Args:
        boxes:       np.ndarray [N,5] [cx,cy,w,h,angle], angle>0 clockwise
        query_boxes: np.ndarray [K,5]
        criterion:   -1 (IoU), 0, 1, or other (intersection area), same as devRotateIoUEval
        device_id:   CUDA device index

    Returns:
        np.ndarray [N,K] of type boxes.dtype
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

    t_boxes = _to_torch_boxes(boxes, device)
    t_qboxes = _to_torch_boxes(query_boxes, device)

    c1 = _rbbox_to_corners_numba_compatible(t_boxes)   # [N,4,2]
    c2 = _rbbox_to_corners_numba_compatible(t_qboxes)  # [K,4,2]

    out = torch.zeros((N, K), device=device, dtype=torch.float32)

    for i in range(N):
        for j in range(K):
            out[i, j] = _pair_measure(c1[i], c2[j], t_boxes[i], t_qboxes[j], criterion)

    return out.cpu().numpy().astype(box_dtype, copy=False)
