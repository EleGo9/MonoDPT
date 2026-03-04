import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import ImageFont

np.set_printoptions   (precision= 4, suppress= True)
torch.set_printoptions(precision= 4)

# from plot.common_operations import *  # Module doesn't exist - functions imported below
# import plot.plotting_params as params
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import cv2
import imutils
import random

from lib.helpers.file_io import imread, imwrite, read_csv
from lib.datasets.kitti.kitti_utils import get_calib_from_file, get_objects_from_label
# from lib.datasets.utils import convertRot2Alpha
from lib.visualization import *
from lib.visualization import draw_3d_box, draw_transparent_box,draw_2d_boxes, project_3d, plot_on_image_from_txt, imhstack, create_colorbar, draw_tick_marks

# from lib.math_3d import *
# from lib.util import create_colorbar, draw_bev, draw_tick_marks, imhstack, draw_3d_box

import glob
import argparse


# compression_ratio = args.compression

# show_ground_truth = False
# show_baseline     = True
def main(args):
    dataset  = args.dataset
    folder   = args.folder
    random_sampling_flag = False
    # show_gt_in_image  = args.show_gt_in_image
    num_files_to_plot = 200

    seed = 0
    np.random.seed(seed)
    random.seed(seed)
    # bev_scale    = 1   # Pixels per meter
    # bev_max_w    = 20   # Max along positive X direction. # This corresponds to the camera-view of (-max, max)
    # bev_w        = 2 * bev_max_w * bev_scale
    # bev_max_z= 100
    # ticks = [100, 80, 60, 40,0]

    bev_scale    = 8   # Pixels per meter
    bev_max_w    = 15   # Max along positive X direction. # This corresponds to the camera-view of (-max, max)
    bev_w        = 2 * bev_max_w * bev_scale
    bev_max_z= 120
    ticks = [120, 90, 60, 30, 0]

    # Parse multiple ground truth folders (comma-separated)
    if args.gt_folder != "":
        gt_folders = [f.strip() for f in args.gt_folder.split(',')]
    else:
        gt_folders = []



    # ==================================================================================================
    # Dataset Specific Settings
    # ==================================================================================================
    if dataset == "kitti" or dataset == "nuscenes" or dataset == "indy":
        box_class_list= ["car"] # , "cyclist", "pedestrian", 'truck', 'van']
        # lidar_points_in_gt = False

        if len(gt_folders) == 0:
            gt_folders = ["data/KITTI/training"]
        print("Using KITTI dataset with ground truth folders:", gt_folders)

    if dataset == "mantruck":
        box_class_list= ['Truck','Pedestrian', 'Car', 'Cyclist']
        print("Using MANTRUCK dataset with ground truth folders:", gt_folders)

    if dataset == 'iveco':
        box_class_list= ['Bulldozer','Dump','Excavator', 'Flatbed', 'Tractor', 'Truck', 'VTLM']
        print("Using IVECO dataset with ground truth folders:", gt_folders)


    print(box_class_list)

    # ==================================================================================================
    # Output
    # ==================================================================================================
    # img_save_folder   = "images/qualitative/" + dataset + "/" + args.folder.split('/')[1] #"waymo_with_gt_segment-14739149465358076158_4740_000_4760_000_with_camera_labels_new"
    # print(img_save_folder)
    # os.makedirs(img_save_folder,exist_ok= True)

    bev_c1       = (0, 250, 250)
    bev_c2       = (0, 175, 250)
    # c_gts        = (0, 255, 0)
    # c            = (255, 0, 0) #(255,48,51)#(255,0,0)#(114,211,254) #(252,221,152 # RGB

    # # color_gt     = (153,255,51)#(0, 255 , 0)
    # color_gt =  (0, 255, 0)
    # color_pred_2 =  (255, 0, 0) 
    # # color_pred_2 = (59, 221, 255)#(51,153,255)#(94,45,255)#(255, 128, 0)

    # use_classwise_color = False

    files_list   = sorted(glob.glob(folder + "/*.txt"))
    num_files    = len(files_list)

    print("Choosing {} files out of {} files...".format(num_files_to_plot, num_files))

    # Extract basenames from output files
    basenames = [os.path.basename(f).split(".")[0] for f in files_list]

    if random_sampling_flag:
        indices = np.sort(np.random.choice(range(num_files), min(num_files_to_plot, num_files), replace=False))
        basenames = [basenames[i] for i in indices]
        files_list = [files_list[i] for i in indices]

    print("File number:", len(basenames))


    # if dataset == "kitti":
    #     file_index = [669, 983, 1022, 1198, 1232, 1355]
    # elif dataset == "nusc_kitti":
    #     file_index = [2268, 828, 451, 5247, 5129, 4399, 3930]
    # elif dataset == "waymo":
    #     file_index = [689, 1074, 4061, 8694, 11314, 12874]
    # file_index = np.array(file_index)

    def find_file_in_folders(basename, gt_folders, subfolder):
        """Search for a file across multiple GT folders"""
        for gt_folder in gt_folders:
            file_path = os.path.join(gt_folder, subfolder, basename + ".txt")
            if os.path.exists(file_path):
                return file_path
        return None

    def find_image_in_folders(basename, gt_folders, subfolder):
        """Search for an image file across multiple GT folders"""
        for gt_folder in gt_folders:
            for ext in [".png", ".jpg", ".jpeg"]:
                file_path = os.path.join(gt_folder, subfolder, basename + ext)
                if os.path.exists(file_path):
                    return file_path
        return None

    for i, (basename, pred_file) in enumerate(zip(basenames, files_list)):
        print(f"Processing {i+1}/{len(basenames)}: {basename}")

        # Search for files across all GT folders
        cal_file   = find_file_in_folders(basename, gt_folders, "calib")
        img_file   = find_image_in_folders(basename, gt_folders, "image_2")
        label_file = find_file_in_folders(basename, gt_folders, "label_2")

        if cal_file is None:
            print(f"  Warning: Calibration file not found for {basename}, skipping...")
            continue
        if img_file is None:
            print(f"  Warning: Image file not found for {basename}, skipping...")
            continue

        p2         = get_calib_from_file(cal_file)['P2']
        gt_img     = read_csv(label_file, ignore_warnings= True, use_pandas= True) if label_file else None

        if gt_img is not None:
            gt_other      = gt_img[:, 1:].astype(float)
            #       0  1   2      3   4   5   6    7    8    9   10   11   12    13     14
            # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, lidar
            # if lidar_points_in_gt:
            #     gt_bad_index  = np.logical_or(np.logical_or(np.logical_or(gt_other[:, 14] == 0, gt_other[:, 3]-gt_other[:, 5] >=0), gt_other[:, 4]-gt_other[:, 6] >=0), gt_other[:, 12] <=0 )
            # else:
            gt_bad_index  = np.logical_or(np.logical_or(gt_other[:, 3]-gt_other[:, 5] >=0, gt_other[:, 4]-gt_other[:, 6] >=0), gt_other[:, 12] <=0 )

            gt_good_index = np.logical_not(gt_bad_index)
            gt_img        = gt_img[gt_good_index]


        img = imread(img_file)
        # if resolution and test images are not the same:
        from lib.datasets.kitti.kitti_utils import get_affine_transform
        from PIL import Image
        # img = Image.open(img_file) 
        # img_size = np.array(img.size)        
        # print('img_size', img_size)
        # center = np.array(img_size) / 2
        # crop_size, crop_scale = img_size, 1
        # trans, trans_inv = get_affine_transform(center, crop_size, 0, np.array([1280, 384]), inv=1)
        # img = img.transform(tuple(np.array([1280, 384]).tolist()),
        #                     method=Image.AFFINE,
        #                     data=tuple(trans_inv.reshape(-1).tolist()),
        #                     resample=Image.BILINEAR)
        # print('Transformed image size:', img.size)
        # img = np.array(img)
        print('Image shape:', img.shape)

        bev_img = create_colorbar(bev_max_z * bev_scale, bev_w, color_lo=bev_c1, color_hi=bev_c2)

        predictions_img  = read_csv(pred_file, ignore_warnings= True, use_pandas= True)

        if gt_img is not None:
            img, bev_img = plot_on_image_from_txt(img, 
                    gt_img,
                    p2, 
                    box_colors = [(0, 255, 0), (0, 50, 0), (0, 200, 0), (0, 150, 0), (0, 100, 0)], 
                    canvas_bev=bev_img,
                    bev_scale=bev_scale,)
        else:
            print("No ground truth for image:", img_file)
                    
        if predictions_img is not None:
            # print(predictions_img)
            # print(len(predictions_img))
            # print(predictions_img[-1])
            # if len(predictions_img)>15:
            #     print()
            #     if float(predictions_img[15])>0.35:
            img, bev_img = plot_on_image_from_txt(img, 
                predictions_img, 
                p2, 
                box_colors = [(0, 0, 255),(0, 0, 200),(0, 0, 150),(0, 0, 100),(0, 50, 200)],
                canvas_bev=bev_img,
                bev_scale=bev_scale,)
        else:
            print("No predictions for image:", img_file)
        bev_img = cv2.flip(bev_img, 0)
        # draw tick marks
        bev_img = draw_tick_marks(bev_img, ticks)
        img = imhstack(img, bev_img)


        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(args.folder, basename + ".png"), img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='implementation of GUPNet')
    parser.add_argument('--dataset', type=str, default = "kitti", help='one of kitti,nusc_kitti,nuscenes,waymo, indy, mantruck')
    parser.add_argument('--folder' , type=str, default = "output/gup_nuscenes/results_test/data", help='folder containing prediction output files')
    parser.add_argument('--gt_folder',type=str, default = "", help='comma-separated list of ground truth folders (e.g., "data/KITTI/training,data/nuScenes/val")')
    args = parser.parse_args()
    main(args)
