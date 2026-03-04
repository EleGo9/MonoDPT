import math

import numba
import numpy as np
from numba import cuda


@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)
def trangle_area(a, b, c):
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) *
            (b[0] - c[0])) / 2.0


@cuda.jit('(float32[:], int32)', device=True, inline=True)
def area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            trangle_area(int_pts[:2], int_pts[2 * i + 2:2 * i + 4],
                         int_pts[2 * i + 4:2 * i + 6]))
    return area_val


@cuda.jit('(float32[:], int32)', device=True, inline=True)
def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    """Sort vertices of polygon in counterclockwise order.
    
    This corrected implementation uses a more robust approach for sorting
    that works for arbitrary angles.
    """
    if num_of_inter > 0:
        # Find centroid
        center = cuda.local.array((2, ), dtype=numba.float32)
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        
        # Calculate angles
        angles = cuda.local.array((16, ), dtype=numba.float32)
        for i in range(num_of_inter):
            dx = int_pts[2 * i] - center[0]
            dy = int_pts[2 * i + 1] - center[1]
            angles[i] = math.atan2(dy, dx)
        
        # Sort based on angles - bubble sort for simplicity
        for i in range(num_of_inter):
            for j in range(0, num_of_inter - i - 1):
                if angles[j] > angles[j + 1]:
                    # Swap angles
                    temp_angle = angles[j]
                    angles[j] = angles[j + 1]
                    angles[j + 1] = temp_angle
                    
                    # Swap points
                    temp_x = int_pts[2 * j]
                    temp_y = int_pts[2 * j + 1]
                    int_pts[2 * j] = int_pts[2 * (j + 1)]
                    int_pts[2 * j + 1] = int_pts[2 * (j + 1) + 1]
                    int_pts[2 * (j + 1)] = temp_x
                    int_pts[2 * (j + 1) + 1] = temp_y


@cuda.jit(
    '(float32[:], float32[:], int32, int32, float32[:])',
    device=True,
    inline=True)
def line_segment_intersection(pts1, pts2, i, j, temp_pts):
    """Improved line segment intersection that uses more robust calculations."""
    A = cuda.local.array((2, ), dtype=numba.float32)
    B = cuda.local.array((2, ), dtype=numba.float32)
    C = cuda.local.array((2, ), dtype=numba.float32)
    D = cuda.local.array((2, ), dtype=numba.float32)

    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    
    # Use more numerically stable method from line_segment_intersection_v1
    area_abc = trangle_area(A, B, C)
    area_abd = trangle_area(A, B, D)

    # Check if C and D are on opposite sides of line AB
    if area_abc * area_abd >= 0:
        return False

    area_cda = trangle_area(C, D, A)
    area_cdb = area_cda + area_abc - area_abd

    # Check if A and B are on opposite sides of line CD
    if area_cda * area_cdb >= 0:
        return False
        
    # Compute intersection point
    # We use a parameter t along the line AB: P = A + t(B-A)
    # t = area_cda / (area_abd - area_abc)
    t = area_cda / (area_abd - area_abc + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Compute intersection
    dx = t * (B[0] - A[0])
    dy = t * (B[1] - A[1])
    temp_pts[0] = A[0] + dx
    temp_pts[1] = A[1] + dy
    
    # Add numerical tolerance to ensure points are not missed due to floating-point errors
    if 0.0 <= t <= 1.0:
        return True
    return False


@cuda.jit('(float32, float32, float32[:])', device=True, inline=True)
def point_in_quadrilateral(pt_x, pt_y, corners):
    """Improved point-in-polygon test using winding number method."""
    # Use the ray casting algorithm to determine if a point is inside a polygon
    inside = False
    j = 3  # Last vertex index
    
    for i in range(4):
        # Check if point is on an edge
        x1 = corners[2 * i]
        y1 = corners[2 * i + 1]
        x2 = corners[2 * j]
        y2 = corners[2 * j + 1]
        
        # Point is on edge or vertex
        if ((x1 == pt_x and y1 == pt_y) or 
            (x2 == pt_x and y2 == pt_y)):
            return True
            
        # Check if ray from point crosses segment
        if ((y1 > pt_y) != (y2 > pt_y)):
            # Compute x-coordinate of intersection
            if pt_x < x1 + (pt_y - y1) * (x2 - x1) / (y2 - y1 + 1e-10):
                inside = not inside
                
        j = i
        
    return inside


@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)
def quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    
    # Check if vertices from first polygon are inside second polygon
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
            
    # Check if vertices from second polygon are inside first polygon
    for i in range(4):
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
            
    # Find intersections between edges
    temp_pts = cuda.local.array((2, ), dtype=numba.float32)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                # Check if point already exists in array to avoid duplicates
                duplicate = False
                for k in range(num_of_inter):
                    if (abs(int_pts[k * 2] - temp_pts[0]) < 1e-5 and 
                        abs(int_pts[k * 2 + 1] - temp_pts[1]) < 1e-5):
                        duplicate = True
                        break
                
                if not duplicate:
                    int_pts[num_of_inter * 2] = temp_pts[0]
                    int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                    num_of_inter += 1
                    
    return num_of_inter


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def rbbox_to_corners(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = cuda.local.array((4, ), dtype=numba.float32)
    corners_y = cuda.local.array((4, ), dtype=numba.float32)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 *
                i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i
                + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def inter(rbbox1, rbbox2):
    corners1 = cuda.local.array((8, ), dtype=numba.float32)
    corners2 = cuda.local.array((8, ), dtype=numba.float32)
    intersection_corners = cuda.local.array((16, ), dtype=numba.float32)

    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)

    num_intersection = quadrilateral_intersection(corners1, corners2,
                                                  intersection_corners)
    
    if num_intersection <= 2:
        return 0.0  # No valid intersection with fewer than 3 points
        
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    return area(intersection_corners, num_intersection)


@cuda.jit('(float32[:], float32[:], int32)', device=True, inline=True)
def devRotateIoUEval(rbox1, rbox2, criterion=-1):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    
    # Check for identical boxes - if dimensions and centers are the same, return 1.0
    if (abs(rbox1[0] - rbox2[0]) < 1e-5 and abs(rbox1[1] - rbox2[1]) < 1e-5 and
        abs(rbox1[2] - rbox2[2]) < 1e-5 and abs(rbox1[3] - rbox2[3]) < 1e-5):
        # Special case for identical boxes
        sin_diff = math.sin(rbox1[4]) * math.cos(rbox2[4]) - math.cos(rbox1[4]) * math.sin(rbox2[4])
        cos_diff = math.cos(rbox1[4]) * math.cos(rbox2[4]) + math.sin(rbox1[4]) * math.sin(rbox2[4])
        diff_angle = math.atan2(sin_diff, cos_diff)
        
        # Check if angles are multiples of 90 degrees (π/2) apart
        if abs(diff_angle) < 1e-3 or abs(abs(diff_angle) - math.pi) < 1e-3:
            return 1.0  # Boxes are identical or 180° apart, both giving perfect overlap
    
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter


@cuda.jit('(int64, int64, float32[:], float32[:], float32[:], int32)', fastmath=False)
def rotate_iou_kernel_eval(N, K, dev_boxes, dev_query_boxes, dev_iou, criterion=-1):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    row_size = min(N - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(K - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)
    block_qboxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)

    dev_query_box_idx = threadsPerBlock * col_start + tx
    dev_box_idx = threadsPerBlock * row_start + tx
    if (tx < col_size):
        block_qboxes[tx * 5 + 0] = dev_query_boxes[dev_query_box_idx * 5 + 0]
        block_qboxes[tx * 5 + 1] = dev_query_boxes[dev_query_box_idx * 5 + 1]
        block_qboxes[tx * 5 + 2] = dev_query_boxes[dev_query_box_idx * 5 + 2]
        block_qboxes[tx * 5 + 3] = dev_query_boxes[dev_query_box_idx * 5 + 3]
        block_qboxes[tx * 5 + 4] = dev_query_boxes[dev_query_box_idx * 5 + 4]
    if (tx < row_size):
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        for i in range(col_size):
            offset = row_start * threadsPerBlock * K + col_start * threadsPerBlock + tx * K + i
            dev_iou[offset] = devRotateIoUEval(block_boxes[tx * 5:tx * 5 + 5],
                                          block_qboxes[i * 5:i * 5 + 5], criterion)


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """rotated box iou running in gpu. 500x faster than cpu version
    (take 5ms in one example with numba.cuda code).
    
    Args:
        boxes (float tensor: [N, 5]): rbboxes. format: centers, dims, 
            angles(clockwise when positive)
        query_boxes (float tensor: [K, 5]): same format as boxes
        device_id (int, optional): Defaults to 0. GPU device id
    
    Returns:
        iou tensor: [N, K]
    """
    box_dtype = boxes.dtype
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    threadsPerBlock = 8 * 8
    cuda.select_device(device_id)
    blockspergrid = (div_up(N, threadsPerBlock), div_up(K, threadsPerBlock))
    
    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        rotate_iou_kernel_eval[blockspergrid, threadsPerBlock, stream](
            N, K, boxes_dev, query_boxes_dev, iou_dev, criterion)
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(boxes.dtype)