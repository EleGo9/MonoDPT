import os, sys
sys.path.append(os.getcwd())

import numpy as np
import math
import cv2
import shutil
import torch
from PIL import Image
from lib.helpers.file_io import read_csv

def mkdir_if_missing(directory, delete_if_exist=False):
    """
    Recursively make a directory structure even if missing.

    if delete_if_exist=True then we will delete it first
    which can be useful when better control over initialization is needed.
    """
    if delete_if_exist and os.path.exists(directory): shutil.rmtree(directory)

    # check if not exist, then make
    if not os.path.exists(directory):
        os.makedirs(directory)

def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    if type(x3d) == np.ndarray:

        p2_batch = np.zeros([x3d.shape[0], 4, 4])
        p2_batch[:, :, :] = p2[np.newaxis, :, :]

        ry3d_cos = np.cos(ry3d)
        ry3d_sin = np.sin(ry3d)

        R = np.zeros([x3d.shape[0], 4, 3])
        R[:, 0, 0] = ry3d_cos
        R[:, 0, 2] = ry3d_sin
        R[:, 1, 1] = 1
        R[:, 2, 0] = -ry3d_sin
        R[:, 2, 2] = ry3d_cos

        corners_3d = np.zeros([x3d.shape[0], 3, 8])

        # setup X
        corners_3d[:, 0, :] = -l3d[:, np.newaxis] / 2
        corners_3d[:, 0, 1:5] = l3d[:, np.newaxis] / 2

        # setup Y
        corners_3d[:, 1, :] = -h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 2:4] = h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 6:8] = h3d[:, np.newaxis] / 2

        # setup Z
        corners_3d[:, 2, :] = -w3d[:, np.newaxis] / 2
        corners_3d[:, 2, 3:7] = w3d[:, np.newaxis] / 2

        # rotate
        corners_3d = R @ corners_3d

        # translate
        corners_3d[:, 0, :] += x3d[:, np.newaxis]
        corners_3d[:, 1, :] += y3d[:, np.newaxis]
        corners_3d[:, 2, :] += z3d[:, np.newaxis]
        corners_3d[:, 3, :] = 1

        # project to 2D
        corners_2d = p2_batch @ corners_3d

        corners_2d[:, :2, :] /= corners_2d[:, 2, :][:, np.newaxis, :]

        verts3d = corners_2d

    elif type(x3d) == torch.Tensor:

        p2_batch = torch.zeros(x3d.shape[0], 4, 4)
        p2_batch[:, :, :] = p2[np.newaxis, :, :]

        ry3d_cos = torch.cos(ry3d)
        ry3d_sin = torch.sin(ry3d)

        R = torch.zeros(x3d.shape[0], 4, 3)
        R[:, 0, 0] = ry3d_cos
        R[:, 0, 2] = ry3d_sin
        R[:, 1, 1] = 1
        R[:, 2, 0] = -ry3d_sin
        R[:, 2, 2] = ry3d_cos

        corners_3d = torch.zeros(x3d.shape[0], 3, 8)

        # setup X
        corners_3d[:, 0, :] = -l3d[:, np.newaxis] / 2
        corners_3d[:, 0, 1:5] = l3d[:, np.newaxis] / 2

        # setup Y
        corners_3d[:, 1, :] = -h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 2:4] = h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 6:8] = h3d[:, np.newaxis] / 2

        # setup Z
        corners_3d[:, 2, :] = -w3d[:, np.newaxis] / 2
        corners_3d[:, 2, 3:7] = w3d[:, np.newaxis] / 2

        # rotate
        corners_3d = torch.bmm(R, corners_3d)

        corners_3d = corners_3d.to(x3d.device)
        p2_batch = p2_batch.to(x3d.device)

        # translate
        corners_3d[:, 0, :] += x3d[:, np.newaxis]
        corners_3d[:, 1, :] += y3d[:, np.newaxis]
        corners_3d[:, 2, :] += z3d[:, np.newaxis]
        corners_3d[:, 3, :] = 1

        # project to 2D
        corners_2d = torch.bmm(p2_batch, corners_3d)

        corners_2d[:, :2, :] /= corners_2d[:, 2, :][:, np.newaxis, :]

        verts3d = corners_2d

    else:

        # compute rotational matrix around yaw axis
        R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                      [0, 1, 0],
                      [-math.sin(ry3d), 0, +math.cos(ry3d)]])

        # 3D bounding box corners
        x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0], dtype=np.float64)
        y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d], dtype=np.float64)
        z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0], dtype=np.float64)

        x_corners += -l3d / 2
        y_corners += -h3d / 2
        z_corners += -w3d / 2

        # bounding box in object co-ordinate
        corners_3d = np.array([x_corners, y_corners, z_corners])

        # rotate
        corners_3d = R.dot(corners_3d)

        # translate
        corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

        corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
        corners_2D = p2.dot(corners_3D_1)
        corners_2D = corners_2D / corners_2D[2]

        # corners_2D = np.zeros([3, corners_3d.shape[1]])
        # for i in range(corners_3d.shape[1]):
        #    a, b, c, d = argoverse.utils.calibration.proj_cam_to_uv(corners_3d[:, i][np.newaxis, :], p2)
        #    corners_2D[:2, i] = a
        #    corners_2D[2, i] = corners_3d[2, i]

        bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

        verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d
    else:
        return verts3d

def draw_line(im, v1, v2, color=(0, 200, 200), thickness=1):

    cv2.line(im, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), color, thickness)

def draw_circle(im, pos, radius=5, thickness=1, color=(250, 100, 100), fill=True):

    if fill: thickness = -1

    cv2.circle(im, (int(pos[0]), int(pos[1])), radius, color=color, thickness=thickness)

def draw_2d_boxes(im, boxes, color=(0, 255, 0), thickness=2):
    img = im.copy()    
    # Handle single box case
    if not isinstance(boxes[0], (list, tuple)):
        boxes = [boxes]
    
    for box in boxes:
        x1, y1, x2, y2 = [int(val) for val in box]
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    return img


def draw_transparent_box(im, box, blend=0.5, color=(0, 255, 255)):

    x_s = int(np.clip(min(box[0], box[2]), a_min=0, a_max=im.shape[1]))
    x_e = int(np.clip(max(box[0], box[2]), a_min=0, a_max=im.shape[1]))
    y_s = int(np.clip(min(box[1], box[3]), a_min=0, a_max=im.shape[0]))
    y_e = int(np.clip(max(box[1], box[3]), a_min=0, a_max=im.shape[0]))

    im[y_s:y_e + 1, x_s:x_e + 1, 0] = im[y_s:y_e + 1, x_s:x_e + 1, 0] * blend + color[0] * (1 - blend)
    im[y_s:y_e + 1, x_s:x_e + 1, 1] = im[y_s:y_e + 1, x_s:x_e + 1, 1] * blend + color[1] * (1 - blend)
    if __name__ == '__main__':
        im[y_s:y_e + 1, x_s:x_e + 1, 2] = im[y_s:y_e + 1, x_s:x_e + 1, 2] * blend + color[2] * (1 - blend)



def draw_3d_box(im, verts, color=(0, 200, 200), thickness=1):

    for lind in range(0, verts.shape[0] - 1):
        v1 = verts[lind]
        v2 = verts[lind + 1]
        cv2.line(im, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), color, thickness)

    # draw_transparent_polygon(im, verts[5:9, :], blend=0.5, color=color)
    return im

def get_polygon_grid(im, poly_verts):

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.path import Path

    nx = im.shape[1]
    ny = im.shape[0]
    #poly_verts = [(1, 1), (5, 1), (5, 9), (3, 2), (1, 1)]

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = Path(poly_verts)
    grid = path.contains_points(points)
    grid = grid.reshape((ny, nx))

    return grid

def draw_transparent_polygon(im, verts, blend=0.5, color=(0, 255, 255)):

    mask = get_polygon_grid(im, verts[:4, :])

    im[mask, 0] = im[mask, 0] * blend + (1 - blend) * color[0]
    im[mask, 1] = im[mask, 1] * blend + (1 - blend) * color[1]
    im[mask, 2] = im[mask, 2] * blend + (1 - blend) * color[2]

def interp_color(dist, bounds=[0, 1], color_lo=(0,0, 250), color_hi=(0, 250, 250)):

    percent = (dist - bounds[0]) / (bounds[1] - bounds[0])
    b = color_lo[0] * (1 - percent) + color_hi[0] * percent
    g = color_lo[1] * (1 - percent) + color_hi[1] * percent
    r = color_lo[2] * (1 - percent) + color_hi[2] * percent

    return (b, g, r)

def create_colorbar(height, width, color_lo=(0,0, 250), color_hi=(0, 250, 250)):

    im = np.zeros([height, width, 3])

    for h in range(0, height):

        color = interp_color(h + 0.5, [0, height], color_hi, color_lo)
        im[h, :, 0] = (color[0])
        im[h, :, 1] = (color[1])
        im[h, :, 2] = (color[2])

    return im.astype(np.uint8)

def draw_tick_marks(im, ticks):

    ticks_loc = list(range(0, im.shape[0] + 1, int((im.shape[0]) / (len(ticks) - 1))))

    for tind, tick in enumerate(ticks):
        y = min(max(ticks_loc[tind], 50), im.shape[0] - 10)
        x = im.shape[1] - 115

        draw_text(im, '-{}m'.format(tick), (x, y), lineType=2, scale=1.1, bg_color=None)
    return im

def draw_text(im, text, pos, scale=0.4, color=(0, 0, 0), font=cv2.FONT_HERSHEY_SIMPLEX, bg_color=(0, 255, 255),
              blend=0.33, lineType=1):

    pos = [int(pos[0]), int(pos[1])]

    if bg_color is not None:

        text_size, _ = cv2.getTextSize(text, font, scale, lineType)
        x_s = int(np.clip(pos[0], a_min=0, a_max=im.shape[1]))
        x_e = int(np.clip(pos[0] + text_size[0] - 1 + 4, a_min=0, a_max=im.shape[1]))
        y_s = int(np.clip(pos[1] - text_size[1] - 2, a_min=0, a_max=im.shape[0]))
        y_e = int(np.clip(pos[1] + 1 - 2, a_min=0, a_max=im.shape[0]))

        im[y_s:y_e + 1, x_s:x_e + 1, 0] = im[y_s:y_e + 1, x_s:x_e + 1, 0]*blend + bg_color[0] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 1] = im[y_s:y_e + 1, x_s:x_e + 1, 1]*blend + bg_color[1] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 2] = im[y_s:y_e + 1, x_s:x_e + 1, 2]*blend + bg_color[2] * (1 - blend)

        pos[0] = int(np.clip(pos[0] + 2, a_min=0, a_max=im.shape[1]))
        pos[1] = int(np.clip(pos[1] - 2, a_min=0, a_max=im.shape[0]))

    cv2.putText(im, text, tuple(pos), font, scale, color, lineType)

def draw_bev(canvas_bev, z3d, l3d, w3d, x3d, ry3d, color=(0, 200, 200), scale=1, thickness=2, text= None):

    w = l3d * scale
    l = w3d * scale
    x = x3d * scale
    z = z3d * scale
    r = ry3d*-1

    corners1 = np.array([
        [-w / 2, -l / 2, 1],
        [+w / 2, -l / 2, 1],
        [+w / 2, +l / 2, 1],
        [-w / 2, +l / 2, 1]
    ])

    ry = np.array([
        [+math.cos(r), -math.sin(r), 0],
        [+math.sin(r), math.cos(r), 0],
        [0, 0, 1],
    ])

    corners2 = ry.dot(corners1.T).T

    corners2[:, 0] += x + canvas_bev.shape[1] / 2
    corners2[:, 1] += z

    draw_line(canvas_bev, corners2[0], corners2[1], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[1], corners2[2], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[2], corners2[3], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[3], corners2[0], color=color, thickness=thickness)

    if text is not None:
        thickness=2
        cv2.putText(canvas_bev, text, (int(corners2[0, 0]), int(corners2[0, 1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), thickness, cv2.LINE_AA)
    return canvas_bev

def imhstack(im1, im2):

    sf = im1.shape[0] / im2.shape[0]

    if sf > 1:
        im2 = cv2.resize(im2, (int(im2.shape[1] / sf), im1.shape[0]))
    elif sf < 1:
        im1 = cv2.resize(im1, (int(im1.shape[1] / sf), im2.shape[0]))


    im_concat = np.hstack((im1, im2))

    return im_concat

def get_bounds(img_shape_x, img_shape_y, new_shape_x= None, new_shape_y= None, curr_scale= None):
    cx      = img_shape_x //2
    cy      = img_shape_y //2

    if curr_scale is not None:
        new_shape_x = int(img_shape_x/float(curr_scale))
        new_shape_y = int(img_shape_y/float(curr_scale))

    half_new_x   = new_shape_x//2
    half_new_y   = new_shape_y//2
    left_x       = cx - half_new_x
    right_x      = cx + half_new_x + new_shape_x % 2
    top_y        = cy - half_new_y
    bottom_y     = cy + half_new_y + new_shape_y % 2

    return left_x, right_x, top_y, bottom_y

def get_downsampled_image(img, curr_scale, height_first_flag= True):
    if height_first_flag:
        cx      = img.shape[0] //2
        cy      = img.shape[1] //2
        new_shape    = np.array(img.shape)
        new_shape[:2]= new_shape[:2]/float(curr_scale)
        new_shape    = new_shape.astype(np.int)


        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        new_img_pil = img_pil.resize((new_shape[1], new_shape[0]), Image.ANTIALIAS)
        new_img = cv2.cvtColor(np.array(new_img_pil), cv2.COLOR_RGB2BGR)

        # Paste the input image
        left_x, right_x, left_y, right_y = get_bounds(img_shape_x= img.shape[0], img_shape_y= img.shape[1], new_shape_x= new_shape[0], new_shape_y= new_shape[1])
        # half_new_x   = new_shape[0]//2
        # half_new_y   = new_shape[1]//2
        # left_x       = cx - half_new_x
        # right_x      = cx + half_new_x + new_shape[0] % 2
        # left_y       = cy - half_new_y
        # right_y      = cy + half_new_y + new_shape[1] % 2
        img_out      = np.zeros(img.shape).astype(np.uint8)
        img_out[left_x : right_x, left_y : right_y] = new_img

    else:
        from skimage.transform import resize
        # Channels are first
        cx      = img.shape[1] //2
        cy      = img.shape[2] //2
        new_shape    = np.array(img.shape)
        new_shape[1:]= new_shape[1:]/float(curr_scale)
        new_shape    = new_shape.astype(np.int)

        # Bring it in height first format
        img_temp     = img.transpose(1,2,0)          #CHW --> HWC
        new_img_temp = resize(img_temp, (new_shape[1], new_shape[2], new_shape[0])) #HWC
        new_img      = new_img_temp.transpose(2,0,1) #HWC --> CHW

        # Paste the input image
        half_new_x   = new_shape[1]//2
        half_new_y   = new_shape[2]//2
        left_x       = cx - half_new_x
        right_x      = cx + half_new_x + new_shape[1] % 2
        left_y       = cy - half_new_y
        right_y      = cy + half_new_y + new_shape[2] % 2
        img_out      = np.zeros(img.shape).astype(np.uint8)
        img_out[:, left_x : right_x, left_y : right_y] = new_img

    return img_out

def is_slanted(label_file, ego_height= 1.65):
    # print(label_file)
    gt_img     = read_csv(label_file, ignore_warnings= True, use_pandas= True)

    if gt_img is not None:
        car_index  = np.logical_or(gt_img[:, 0] == "Car", gt_img[:, 0] == "car")
        gt_other   = gt_img[:, 1:].astype(float)

        if np.sum(car_index) > 0:
            #       0  1   2      3   4   5   6    7    8    9   10   11   12    13     14
            # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, lidar
            cars_filter_y3d = gt_other[car_index, 11]
            good_cars_count = np.sum( np.logical_and(cars_filter_y3d >= 0.0, cars_filter_y3d <= ego_height + 1.0))

            if good_cars_count != np.sum(car_index):
                return True

    return False

def plot_on_image_from_txt(
    img, 
    predictions_img, 
    p2, 
    box_colors, 
    canvas_bev,
    bev_scale=1,):

    if predictions_img is not None and predictions_img.size > 0:
        # Add dimension if there is a single point
        if predictions_img.ndim == 1:
            predictions_img = predictions_img[np.newaxis, :]
        
        N = predictions_img.shape[0]
        
        # Extract 2D box information
        # Format: (cls, _, _, alpha, x1, y1, x2, y2, ...)
        cls = predictions_img[:, 0]        
        x1 = predictions_img[:, 4].astype(np.int32)
        y1 = predictions_img[:, 5].astype(np.int32)
        x2 = predictions_img[:, 6].astype(np.int32)
        y2 = predictions_img[:, 7].astype(np.int32)
        h3d = predictions_img[:, 8]
        w3d = predictions_img[:, 9]
        l3d = predictions_img[:, 10]
        x3d = predictions_img[:, 11]
        y3d = predictions_img[:, 12]  - h3d/2
        z3d = predictions_img[:, 13]
        ry3d = predictions_img[:,14]
        print('DEPTH', z3d)
        
        
        # Get scores if available (index 15)
        if predictions_img.shape[1] > 15:
            scores = predictions_img[:, 15]
            
        else:
            scores = np.ones((predictions_img.shape[0],))

        for j in range(N):
            # Get class name
            # if isinstance(cls[j], (int, np.integer)) and 0 <= cls_indices[j] < len(box_class_list):
            #     box_class = box_class_list[cls_indices[j]]
            # else:
            try:
                box_class = cls[j].lower()
            except:
                box_class = "unknown"

            font_scale=0.5
            thickness=2



            if box_class == "dontcare":
                box_color = box_colors[0]
                font_scale=0.3
                thickness=1
            elif box_class == "car":
                box_color = box_colors[1]
            elif box_class == "pedestrian":
                box_color = box_colors[2]
            elif box_class == "cyclist":
                box_color = box_colors[3]
            else:
                box_color =box_colors[4]

            if scores[j]<0.1:
                return img, canvas_bev

            
            if box_class=='car':
                cv2.rectangle(img, (x1[j], y1[j]), (x2[j], y2[j]), color=box_color, thickness=thickness)
                label_parts = []
                label_parts.append(box_class)
                score_text = f"{scores[j]:.2f}" if isinstance(scores[j], float) else f"{int(scores[j])}"
                label_parts.append(score_text)
                
                label = ": ".join(label_parts)
                
                if label:
                    # Draw background for text
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                    cv2.rectangle(img, (x1[j], y1[j] - text_height - 4), (x1[j] + text_width, y1[j]), box_color, -1)
                    
                    # Draw text
                    cv2.putText(img, label, (x1[j], y1[j] - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                                font_scale, (255, 255, 255), 1, cv2.LINE_AA)
                verts_cur, _ = project_3d(p2, x3d[j], y3d[j], z3d[j], w3d[j], h3d[j], l3d[j], ry3d[j], return_3d=True)
                img = draw_3d_box(img, verts_cur, color= box_color, thickness= thickness)
                bev_img = draw_bev(canvas_bev, z3d[j], l3d[j], w3d[j], x3d[j], ry3d[j], color= box_color, scale = bev_scale, thickness= thickness, text= None)
            # img = imhstack(img, bev_img)
            else:
                bev_img = canvas_bev
    return img, bev_img