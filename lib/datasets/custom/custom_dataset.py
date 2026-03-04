import os
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.kitti.kitti_utils import get_objects_from_label
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
# from lib.datasets.kitti.kitti_eval_python.eval import get_distance_eval_result
import lib.datasets.indy.indy_eval_python.indy_common as indy
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
from lib.datasets.indy.indy_eval_python.eval import get_indy_eval_result
from lib.helpers.rpn_util import get_MAE

import cv2
from lib.helpers.visualization import draw_3d_box, draw_transparent_box,draw_2d_boxes, project_3d
import copy
from .pd import PhotometricDistort
DEBUG = False
from tqdm.auto import tqdm
class Custom_Dataset(data.Dataset):
    def __init__(self, split, cfg, root_dir=None, dataset_id=0):

        # basic configuration
        if root_dir is None:
            self.root_dir = cfg.dataset.root_dir
        else:
            self.root_dir = root_dir
        self.split = split

        # Configurable class parameters
        self.class_name = cfg.dataset.class_name
        self.cls2id = cfg.dataset.cls2id
        if self.cls2id is None:
            # Auto-generate cls2id from class_name if not provided
            self.cls2id = {name: idx for idx, name in enumerate(self.class_name)}
        self.num_classes = len(self.class_name)

        # Configurable resolution (W * H)
        self.resolution = np.array(cfg.dataset.resolution)

        # Configurable max objects
        self.max_objs = cfg.dataset.max_objs

        # Configurable filename format (e.g., '%06d' for 000000, '%08d' for 00000000)
        self.filename_format = cfg.dataset.filename_format[dataset_id] if cfg.dataset.filename_format is not None else '%06d'

        self.use_3d_center = cfg.dataset.use_3d_center
        self.writelist = cfg.dataset.writelist
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.dataset.bbox2d_type
        assert self.bbox2d_type in ['anno', 'proj']
        self.meanshape = cfg.dataset.meanshape
        self.class_merging = cfg.dataset.class_merging
        self.use_dontcare = cfg.dataset.use_dontcare
        self.depth_threshold = cfg.dataset.depth_threshold
        self.distortion = cfg.dataset.distortion
        # self.depth_max = cfg.get('model', 'depth_max')
        # print('Depth_max', self.depth_threshold)

        if self.class_merging:
            self.writelist.extend(['Van', 'Truck'])
        if self.use_dontcare:
            self.writelist.extend(['DontCare'])

        # data split loading
        assert self.split in ['train', 'val', 'trainval', 'test', 'all']
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]

        # path configuration
        # self.data_dir = os.path.join(self.root_dir, 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.root_dir, 'image_2')
        self.calib_dir = os.path.join(self.root_dir, 'calib')
        self.label_dir = os.path.join(self.root_dir, 'label_2')
        self.original_resolution = cfg.dataset.original_resolution
        print(self.original_resolution)
        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval', 'all'] else False

        # CRITICAL: Only filter invalid projections for training, NOT for test/val
        # For evaluation, we need ALL images to match ground truth annotations
        if self.split in ['train', 'trainval', 'all']:
            self.idx_list = self.filter_invalid_projections(self.idx_list)
        else:
            print(f"Skipping filtering for {self.split} split to maintain full dataset for evaluation")

        self.aug_pd = cfg.dataset.aug_pd
        self.aug_crop = cfg.dataset.aug_crop
        self.aug_calib = cfg.dataset.aug_calib

        self.random_flip = cfg.dataset.random_flip
        self.random_crop = cfg.dataset.random_crop
        self.random_mixup3d = cfg.dataset.random_mixup3d
        self.scale = cfg.dataset.scale
        self.shift = cfg.dataset.shift

        self.depth_scale = cfg.dataset.depth_scale

        self.cls2id = cfg.dataset.cls2id

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Configurable class mean sizes (default: Pedestrian, Car, Cyclist)
        default_cls_mean_size = np.array([[1.76255119    ,0.66068622   , 0.84422524   ],
                                          [1.52563191462 ,1.62856739989, 3.88311640418],
                                          [1.73698127    ,0.59706367   , 1.76282397   ]])
        self.cls_mean_size = cfg.dataset.cls_mean_size
        if isinstance(self.cls_mean_size, list):
            self.cls_mean_size = np.array(self.cls_mean_size, dtype=np.float32)

        # Validate cls_mean_size matches number of classes
        if self.cls_mean_size.shape[0] != self.num_classes:
            print(f"Warning: cls_mean_size has {self.cls_mean_size.shape[0]} entries but {self.num_classes} classes defined. Padding/truncating...")
            if self.cls_mean_size.shape[0] < self.num_classes:
                # Pad with zeros
                padding = np.zeros((self.num_classes - self.cls_mean_size.shape[0], 3), dtype=np.float32)
                self.cls_mean_size = np.vstack([self.cls_mean_size, padding])
            else:
                # Truncate
                self.cls_mean_size = self.cls_mean_size[:self.num_classes]

        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 32
        self.pd = PhotometricDistort()
        self.clip_2d = cfg.dataset.clip_2d
        self.kitti_official_eval = cfg.dataset.kitti_official_eval if cfg.dataset.kitti_official_eval is not None else False
        # print('KITTI OFFICIAL EVAL', self.kitti_official_eval)
    def filter_invalid_projections(self, idx_list):
        """Filter out images with 3D projections that fall outside the image boundaries."""
        print(f"Original dataset size: {len(idx_list)}")
        valid_idx_list = []
        print('Remove all samples with depth > ', self.depth_threshold)
        
        for item in tqdm(idx_list, desc="Filtering invalid projections"):
            index = int(item)  # Extract the index from the filename
            objects = self.get_label(index)
            calib = self.get_calib(index)
            has_valid_objects = False
            for obj in objects:
                # Apply the same filtering criteria as in __getitem__
                # if obj.cls_type not in self.writelist:
                #     continue
                # if obj.cls_type == 'DontCare':
                #     continue
                if obj.pos[-1] < 1:
                    continue
                if obj.pos[-1] > self.depth_threshold:
                    print(obj.pos[-1])
                    continue
                
                # Check if the 3D center projects inside the image
                center_3d = obj.pos + [0, -obj.h / 2, 0]  # real 3D center in 3D space
                center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                center_3d = center_3d[0]  # shape adjustment
                
                # Check if the projected center is inside the image
                if (0 <= center_3d[0] < self.original_resolution[0] and 
                    0 <= center_3d[1] < self.original_resolution[1]):
                    has_valid_objects = True

                    break
            
            if has_valid_objects:
                valid_idx_list.append(item)
        
        print(f"Filtered dataset size: {len(valid_idx_list)}")
        return valid_idx_list

    def get_image(self, idx):
        # Try both .png and .jpg formats
        img_file_png = os.path.join(self.image_dir, f'{self.filename_format}.png' % idx)
        img_file_jpg = os.path.join(self.image_dir, f'{self.filename_format}.jpg' % idx)

        if os.path.exists(img_file_png):
            img_file = img_file_png
        elif os.path.exists(img_file_jpg):
            img_file = img_file_jpg
        else:
            raise FileNotFoundError(f"Image not found: {img_file_png} or {img_file_jpg}")
        return Image.open(img_file)    # (H, W, 3) RGB mode

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, f'{self.filename_format}.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, f'{self.filename_format}.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def eval(self, results_dir, logger):
        # self.kitti_official_eval = True
        if self.kitti_official_eval==True:
            print('Starting KITTI official evaluation')
            img_ids = [int(id) for id in self.idx_list]
            dt_annos = kitti.get_label_annos(results_dir)
            gt_annos = kitti.get_label_annos(self.label_dir, img_ids)

            test_id = {'Car': 0, 'Pedestrian':1, 'Cyclist': 2}

            print('==> Evaluating (official) ...')
            car_moderate = 0
            for category in self.writelist:
                results_str, results_dict, mAP3d_R40 = get_official_eval_result(gt_annos, dt_annos, test_id[category])
                if category == 'Car':
                    car_moderate = mAP3d_R40
                print(results_str)
                return car_moderate, results_dict['2d mAP']
        else:
            if logger is not None:
                logger.info("==> Loading detections and GTs...")
            else:
                print("==> Loading detections and GTs...")
            error_avg = get_MAE(results_folder = results_dir, gt_folder= self.label_dir, use_logging= True, logger= logger, visualize=True)
            if img_ids==None:
                img_ids = [int(id) for id in self.idx_list]
            else:
                img_ids = cur_img_ids
            dt_annos = indy.get_label_annos(results_dir, filename_format=self.filename_format)
            gt_annos = indy.get_label_annos(self.label_dir, img_ids, filename_format=self.filename_format)

            test_id = self.cls2id
            if logger is not None:
                logger.info('==> Evaluating (official) ...')
            else:
                print('==> Evaluating (official) ...')

            car_moderate = 0
            results_str, results_dict = get_indy_eval_result(gt_annos, dt_annos, self.writelist, max_depth= self.depth_threshold, iou_thres=0.7)

            if logger is not None:
                logger.info(results_str)
            else:
                print(results_str)
            return results_dict['3d mAP'], results_dict['2d mAP']

    def __len__(self):
        return self.idx_list.__len__()

    def letterbox_resize(self, img, target_size):
        """
        Resize with letterbox (padding) to maintain aspect ratio.

        Args:
            img: PIL Image
            target_size: Target size (W, H)

        Returns:
            img: Resized and padded image
            scale: Scale factor applied
            pad_w, pad_h: Padding added (left/top)
        """
        orig_w, orig_h = img.size
        target_w, target_h = target_size

        # Calculate scale to fit image in target size (maintaining aspect ratio)
        scale = min(target_w / orig_w, target_h / orig_h)

        # New size after scaling
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize image
        # img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        img = np.array(img)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_resized = Image.fromarray(img_resized)

        # Create new image with target size (filled with mean color)
        img_padded = Image.new('RGB', (target_w, target_h),
                              color=tuple((self.mean * 255).astype(np.uint8).tolist()))

        # Calculate padding to center the image
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        # Paste resized image onto padded background
        img_padded.paste(img_resized, (pad_w, pad_h))

        return img_padded, scale, pad_w, pad_h

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # image loading
        img = self.get_image(index)
        orig_ds = str(self.image_dir.split('/')[-2])
        orig_img_size = np.array(img.size)  # Original size (W, H)
        features_size = self.resolution // self.downsample    # W * H

        # DISABLED: affine transform variables (not used with letterbox resize)
        # center = np.array(img.size) / 2
        # crop_size, crop_scale = img.size, 1
        # random_crop_flag = False

        random_flip_flag = False
        random_mix_flag = False
        calib = self.get_calib(index)


        if self.data_augmentation:

            if np.random.random() < self.random_mixup3d:
                random_mix_flag = True

            if self.aug_pd:
                img = np.array(img).astype(np.float32)
                img = self.pd(img).astype(np.uint8)
                img = Image.fromarray(img)

            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # DISABLED aug_crop: Not compatible with letterbox resize
            # if self.aug_crop:
            #     if np.random.random() < self.random_crop:
            #         random_crop_flag = True
            #         crop_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
            #         crop_size = img_size * crop_scale
            #         center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
            #         center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        if random_mix_flag == True:
            count_num = 0
            random_mix_flag = False
            while count_num < 50:
                count_num += 1
                random_index = int(np.random.choice(self.idx_list))
                calib_temp = self.get_calib(random_index)

                if calib_temp.cu == calib.cu and calib_temp.cv == calib.cv and calib_temp.fu == calib.fu and calib_temp.fv == calib.fv:
                    img_temp = self.get_image(random_index)
                    img_size_temp = np.array(img_temp.size)
                    if img_size_temp[0] == orig_img_size[0] and img_size_temp[1] == orig_img_size[1]:
                        objects_1 = self.get_label(index)
                        objects_2 = self.get_label(random_index)
                        if len(objects_1) + len(objects_2) < self.max_objs:
                            random_mix_flag = True
                            if random_flip_flag == True:
                                img_temp = img_temp.transpose(Image.FLIP_LEFT_RIGHT)
                            img_blend = Image.blend(img, img_temp, alpha=0.5)
                            img = img_blend
                            break


        # DISABLED: Affine transformation (replaced with letterbox resize)
        # trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        # img_transform = img.transform(tuple(self.resolution.tolist()),
        #                     method=Image.AFFINE,
        #                     data=tuple(trans_inv.reshape(-1).tolist()),
        #                     resample=Image.BILINEAR)
        # cv2.imshow('img', np.array(img_transform))
        # cv2.waitKey(0)

        # NEW: Letterbox resize (maintains aspect ratio with padding)
        img, resize_scale, pad_w, pad_h = self.letterbox_resize(img, tuple(self.resolution.tolist()))


        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W
        info = {'img_id': index,
                'img_size': self.resolution,  # After resize
                'orig_img_size': orig_img_size,  # Before resize
                'resize_scale': resize_scale,
                'pad_w': pad_w,
                'pad_h': pad_h,
                'bbox_downsample_ratio': self.resolution / features_size,
                'orig_ds': orig_ds}
        # print('INFO',info)

        if self.split == 'test':
            calib = self.get_calib(index)
            # Update calibration for letterbox resize
            calib_updated = copy.deepcopy(calib)
            calib_updated.cu = calib.cu * resize_scale + pad_w
            calib_updated.cv = calib.cv * resize_scale + pad_h
            calib_updated.fu *= resize_scale
            calib_updated.fv *= resize_scale
            calib_updated.P2[0, 0] = calib_updated.fu
            calib_updated.P2[1, 1] = calib_updated.fv
            calib_updated.P2[0, 2] = calib_updated.cu
            calib_updated.P2[1, 2] = calib_updated.cv
            return img, calib_updated.P2, img, info

        #  ============================   get labels   ==============================
        objects = self.get_label(index)
        calib = self.get_calib(index)

        # IMPORTANT: Save original calibration before adjusting
        # We need the ORIGINAL calibration for projecting 3D centers to image plane
        # Then we'll manually apply resize/padding to the projected coordinates
        import copy
        calib_original = copy.deepcopy(calib)

        # Update calibration for letterbox resize
        # This adjusted calibration will be passed to the model
        calib.cu = calib.cu * resize_scale + pad_w
        calib.cv = calib.cv * resize_scale + pad_h
        calib.fu *= resize_scale
        calib.fv *= resize_scale
        calib.P2[0, 0] = calib.fu
        calib.P2[1, 1] = calib.fv
        calib.P2[0, 2] = calib.cu
        calib.P2[1, 2] = calib.cv

        # data augmentation for labels
        if random_flip_flag:
            # Flip after resize
            if self.aug_calib:
                calib.flip(self.resolution)
            for object in objects:
                # Apply resize first, then flip
                object.box2d[0] = object.box2d[0] * resize_scale + pad_w
                object.box2d[1] = object.box2d[1] * resize_scale + pad_h
                object.box2d[2] = object.box2d[2] * resize_scale + pad_w
                object.box2d[3] = object.box2d[3] * resize_scale + pad_h

                [x1, y1, x2, y2] = object.box2d
                object.box2d[0],  object.box2d[2] = self.resolution[0] - x2, self.resolution[0] - x1
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if self.aug_calib:
                    object.pos[0] *= -1
                if object.alpha > np.pi:  object.alpha -= 2 * np.pi  # check range
                if object.alpha < -np.pi: object.alpha += 2 * np.pi
                if object.ry > np.pi:  object.ry -= 2 * np.pi
                if object.ry < -np.pi: object.ry += 2 * np.pi
        else:
            # No flip, just apply resize
            for object in objects:
                object.box2d[0] = object.box2d[0] * resize_scale + pad_w
                object.box2d[1] = object.box2d[1] * resize_scale + pad_h
                object.box2d[2] = object.box2d[2] * resize_scale + pad_w
                object.box2d[3] = object.box2d[3] * resize_scale + pad_h

        # labels encoding
        calibs = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=bool)
        labels = np.zeros((self.max_objs), dtype=np.int8)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32) 
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)

        obj_region = np.zeros((img.shape[1], img.shape[2]), dtype=bool) # (H, W)

        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
        
        for i in range(object_num):
            # filter objects by writelist
            if objects[i].cls_type not in self.writelist:
                # print('!',objects[i].cls_type)
                continue
            if objects[i].cls_type == 'DontCare':
                # print('!',objects[i].cls_type)
                continue
            
            # ignore the samples beyond the threshold
            if objects[i].pos[-1] > self.depth_threshold:
                # print('Too far:', objects[i].pos[-1])
                continue

            # process 2d bbox & get 2d center (already resized above)
            bbox_2d = objects[i].box2d.copy()

            # DISABLED: Affine transformation (replaced with letterbox resize)
            # bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            # bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

            # process 3d center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H

            # create object region
            ymin, ymax = int(max(bbox_2d[1], 0)), int(min(bbox_2d[3], img.shape[1]))
            xmin, xmax = int(max(bbox_2d[0], 0)), int(min(bbox_2d[2], img.shape[2]))
            obj_region[ymin:ymax, xmin:xmax] = 1

            corner_2d = bbox_2d.copy()

            center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)

            # CRITICAL FIX: Use ORIGINAL calibration for projection, not adjusted one
            # This matches the approach in kitti_dataset.py which uses original calib for projection
            # then applies affine transform. Here we use original calib then apply resize/padding.
            center_3d, _ = calib_original.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment

            # Now apply letterbox resize transformation to the projected coordinates
            center_3d[0] = center_3d[0] * resize_scale + pad_w
            center_3d[1] = center_3d[1] * resize_scale + pad_h

            # Handle flipping if applied (flip should be applied AFTER resize)
            # DISABLED: Affine transformation (replaced with letterbox resize)
            # if random_flip_flag and not self.aug_calib:  # random flip for center3d
            #     center_3d[0] = orig_img_size[0] - center_3d[0]
            # center_3d = affine_transform(center_3d.reshape(-1), trans)

            # filter 3d center out of img
            proj_inside_img = True

            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]: 
                proj_inside_img = False
            if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]: 
                proj_inside_img = False

            if proj_inside_img == False:
                # print('proj outside img')
                # print(index)
                # print('-----------------------')
                if DEBUG:
                    img_vis = img.copy()
                    img_vis = np.transpose(img_vis, (1, 2, 0))
                    img_vis = img_vis*self.std + self.mean
                    img_vis = (img_vis * 255).astype(np.uint8)
                    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
                    cv2.imshow('Image with proj outside', img_vis)
                    cv2.waitKey(0)
                continue

            # class
            cls_id = self.cls2id[objects[i].cls_type]
            labels[i] = cls_id

            # encoding 2d/3d boxes
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[i] = 1. * w, 1. * h

            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[i] / self.resolution

            corner_2d_norm = corner_2d
            corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
            corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
            center_3d_norm = center_3d / self.resolution

            l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l = np.clip(l, 0, 1)
                    r = np.clip(r, 0, 1)
                    t = np.clip(t, 0, 1)
                    b = np.clip(b, 0, 1)
                else:
                    continue		

            boxes[i] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[i] = center_3d_norm[0], center_3d_norm[1], l, r, t, b
            depth[i] = objects[i].pos[-1]  # No scaling with letterbox resize

            
            ry_input = objects[i].ry
            heading_angle = calib.ry2alpha(ry_input, center_2d[0])  # Use resized center_2d
            if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi: heading_angle += 2 * np.pi
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding size_3d
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size

            if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                mask_2d[i] = 1

            # DISABLED: Affine calibration transformation (replaced with letterbox resize)
            # M = np.eye(3)
            # M[0:2, 0:3] = trans
            # # Trasformazione della calibrazione
            # # calib.P2 è la matrice 3x4 originale
            # new_P2 = M @ calib.P2
            # calibs[i] = calib.P2
            # print('calibs[ original', calibs[i])
            # calibs[i] = new_P2
            # print('calibs new', calibs[i])

            # Store calibration matrix (already updated for letterbox resize)
            calibs[i] = calib.P2

            # print('calibs[i]', calibs[i])
            # print(calib)

        if random_mix_flag == True:
            # if False:
                objects = self.get_label(random_index)
                # data augmentation for labels
                if random_flip_flag:
                    for object in objects:
                        # Apply resize first
                        object.box2d[0] = object.box2d[0] * resize_scale + pad_w
                        object.box2d[1] = object.box2d[1] * resize_scale + pad_h
                        object.box2d[2] = object.box2d[2] * resize_scale + pad_w
                        object.box2d[3] = object.box2d[3] * resize_scale + pad_h
                        # Then flip
                        [x1, _, x2, _] = object.box2d
                        object.box2d[0],  object.box2d[2] = self.resolution[0] - x2, self.resolution[0] - x1
                        object.ry = np.pi - object.ry
                        if self.aug_calib:
                            object.pos[0] *= -1
                        if object.ry > np.pi:  object.ry -= 2 * np.pi
                        if object.ry < -np.pi: object.ry += 2 * np.pi
                else:
                    # No flip, just apply resize
                    for object in objects:
                        object.box2d[0] = object.box2d[0] * resize_scale + pad_w
                        object.box2d[1] = object.box2d[1] * resize_scale + pad_h
                        object.box2d[2] = object.box2d[2] * resize_scale + pad_w
                        object.box2d[3] = object.box2d[3] * resize_scale + pad_h

                object_num_temp = len(objects) if len(objects) < (self.max_objs - object_num) else (self.max_objs - object_num)
                for i in range(object_num_temp):
                    if objects[i].cls_type not in self.writelist:
                        continue

                    if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                        continue
                    # process 2d bbox & get 2d center (already resized above)
                    bbox_2d = objects[i].box2d.copy()
                    # DISABLED: add affine transformation for 2d boxes.
                    # bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                    # bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

                    # process 3d center
                    center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H

                    # create object region
                    ymin, ymax = int(max(bbox_2d[1], 0)), int(min(bbox_2d[3], img.shape[1]))
                    xmin, xmax = int(max(bbox_2d[0], 0)), int(min(bbox_2d[2], img.shape[2]))
                    obj_region[ymin:ymax, xmin:xmax] = 1

                    corner_2d = bbox_2d.copy()

                    center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                    center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                    center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                    center_3d = center_3d[0]  # shape adjustment
                    # DISABLED: affine transformation
                    # if random_flip_flag and not self.aug_calib:  # random flip for center3d
                    #     center_3d[0] = orig_img_size[0] - center_3d[0]
                    # center_3d = affine_transform(center_3d.reshape(-1), trans)

                    # filter 3d center out of img
                    proj_inside_img = True

                    if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]: 
                        proj_inside_img = False
                    if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]: 
                        proj_inside_img = False

                    if proj_inside_img == False:
                            continue

                    # class
                    cls_id = self.cls2id[objects[i].cls_type]
                    labels[i + object_num] = cls_id

        
                    # encoding 2d/3d boxes
                    w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                    size_2d[i + object_num] = 1. * w, 1. * h

                    center_2d_norm = center_2d / self.resolution
                    size_2d_norm = size_2d[i + object_num] / self.resolution

                    corner_2d_norm = corner_2d
                    corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
                    corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
                    center_3d_norm = center_3d / self.resolution

                    l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
                    t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

                    if l < 0 or r < 0 or t < 0 or b < 0:
                        if self.clip_2d:
                            l = np.clip(l, 0, 1)
                            r = np.clip(r, 0, 1)
                            t = np.clip(t, 0, 1)
                            b = np.clip(b, 0, 1)
                        else:
                            continue		

                    boxes[i + object_num] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
                    # print(boxes)
                    boxes_3d[i + object_num] = center_3d_norm[0], center_3d_norm[1], l, r, t, b

                    # encoding depth
                    # DISABLED with letterbox resize: no crop_scale anymore
                    # if self.depth_scale == 'normal':
                    #     depth[i + object_num] = objects[i].pos[-1] * crop_scale
                    # elif self.depth_scale == 'inverse':
                    #     depth[i + object_num] = objects[i].pos[-1] / crop_scale
                    # elif self.depth_scale == 'none':
                    #     depth[i + object_num] = objects[i].pos[-1]
                    depth[i + object_num] = objects[i].pos[-1]  # No scaling with letterbox resize

                    # encoding heading angle
                    heading_angle = calib.ry2alpha(objects[i].ry, center_2d[0])  # Use resized center_2d
                    if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                    if heading_angle < -np.pi: heading_angle += 2 * np.pi
                    heading_bin[i + object_num], heading_res[i + object_num] = angle2class(heading_angle)

                    #offset_3d[i + object_num] = center_3d - center_heatmap
                    src_size_3d[i + object_num] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                    mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                    size_3d[i + object_num] = src_size_3d[i + object_num] - mean_size

                    if objects[i].trucation <=0.5 and objects[i].occlusion<=2:
                        mask_2d[i + object_num] = 1
                    
                    calibs[i + object_num] = calib.P2


        # collect return data
        inputs = img
        targets = {
                   'calibs': calibs,
                   'indices': indices,
                   'img_size': self.resolution,  # After resize
                   'labels': labels,
                   'boxes': boxes,
                   'boxes_3d': boxes_3d,
                   'depth': depth,
                   'size_2d': size_2d,
                   'size_3d': size_3d,
                   'src_size_3d': src_size_3d,
                   'heading_bin': heading_bin,
                   'heading_res': heading_res,
                   'mask_2d': mask_2d,
                   'obj_region': obj_region}

        info = {'img_id': index,
                'img_size': self.resolution,  # After resize
                'orig_img_size': orig_img_size,  # Before resize
                'resize_scale': resize_scale,
                'pad_w': pad_w,
                'pad_h': pad_h,
                'bbox_downsample_ratio': self.resolution / features_size,
                'orig_ds': orig_ds}
        # print('targets',targets.keys())
        if DEBUG:
            from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcylrtb_to_xyxy
            
            import torch
            from lib.datasets.utils import class2angle

            # 2D visualization
            for box in boxes:
                cx, cy, w, h = box
                if cx == 0 and cy == 0 and w == 0 and h == 0:
                    continue
                # Convert normalized coordinates to pixel coordinates
                cx_px = int(cx * self.resolution[0])
                cy_px = int(cy * self.resolution[1])
                w_px = int(w * self.resolution[0])
                h_px = int(h * self.resolution[1])
                img_vis = img.copy()
                img_vis = np.transpose(img_vis, (1, 2, 0))
                xyxy_box = box_cxcywh_to_xyxy(torch.tensor([[cx_px, cy_px, w_px, h_px]]))
                xyxy_box = xyxy_box.numpy()[0]
                img_vis = draw_2d_boxes(img_vis, xyxy_box, color=(255, 0, 0))
                img_vis = img_vis*self.std + self.mean
                img_vis = (img_vis * 255).astype(np.uint8)
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
                cv2.imshow('2D boxes', img_vis)
                cv2.waitKey(0)

            # 3d boxes visualization
            for box3d, cal_mat, hwl, head_bin, head_res, dpt  in zip(boxes_3d, calibs, size_3d, heading_bin, heading_res, depth):
                img_vis3d = img.copy()
                img_vis3d = np.transpose(img_vis3d, (1, 2, 0))
                p2 = cal_mat
                cx, cy, l, r, t, b = box3d
                if cx == 0 and cy == 0 and w == 0 and h == 0:
                    continue
                cx_px = int(cx * self.resolution[0])
                cy_px = int(cy * self.resolution[1])   
                l = int(l * self.resolution[0])
                r = int(r * self.resolution[0])
                t = int(t * self.resolution[1])
                b = int(b * self.resolution[1])
                b2d_from_3d = box_cxcylrtb_to_xyxy(torch.tensor([[cx_px, cy_px, l, r, t, b]]))
                b2d_from_3d = b2d_from_3d.numpy()[0]
                img_vis3d = draw_2d_boxes(img_vis3d, b2d_from_3d, color=(255, 0, 0))
                img_vis3d = img_vis3d*self.std + self.mean
                img_vis3d = (img_vis3d * 255).astype(np.uint8)
                img_vis3d = cv2.cvtColor(img_vis3d, cv2.COLOR_RGB2BGR)
                # cv2.imshow('Projected 3D boxes', img_vis3d)
                # cv2.waitKey(0)
                dimens = hwl
                locations = calib.img_to_rect(cx_px, cy_px, dpt[0]).reshape(-1)
                locations[1] += hwl[0] / 2
                alpha = class2angle(head_bin, head_res, to_label_format=True)
                ry = calib.alpha2ry(alpha, b2d_from_3d[0])
                verts_cur, _ = project_3d(p2, locations[0], locations[1]- dimens[0]/2, locations[2], dimens[1], dimens[0], dimens[2], ry[0], return_3d=True)
                try:
                    img_vis3d = draw_3d_box(img_vis3d, verts_cur, color= (255,0,0), thickness= 2)
                except:
                    print('draw_3d_box error')
                    continue
                cv2.imshow(f'3D visualization ry={ry_input}', img_vis3d)
                cv2.waitKey(0)

        return inputs, calib.P2, targets, info #TODO: check when this matrciz is used, it is maybe wrong!!!


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from types import SimpleNamespace

    # Create a mock config object that mimics the Pydantic structure
    dataset_cfg = SimpleNamespace(
        root_dir='/media/franco/hdd/dataset/dataset_3d/IVECO_SIM_MONO3D_forest/ugv_camera_fc_f_n_img',
        train_split='train',
        test_split='val',
        random_flip=0.0,
        random_crop=1.0,
        scale=0.8,
        shift=0.1,
        use_dontcare=False,
        class_merging=False,
        writelist=['Car','Truck', 'Dump', 'Bulldozer','Excavator'],
        use_3d_center=False,
        depth_threshold=500,
        bbox2d_type='anno',
        meanshape=False,
        clip_2d=False,
        aug_pd=False,
        aug_crop=False,
        aug_calib=False,
        random_mixup3d=0.0,
        depth_scale='normal',
        distortion=False,
        # Custom_Dataset specific
        class_name=['Car', 'Truck', 'Dump', 'Bulldozer', 'Excavator'],
        cls2id={'Car':0, 'Truck':1, 'Dump':2, 'Bulldozer':3, 'Excavator':4}, 
        resolution=[1022, 770],
        max_objs=50,
        cls_mean_size=[[1.52, 1.63, 3.88], [1.52, 1.63, 3.88], [1.52, 1.63, 3.88],[1.52, 1.63, 3.88],[1.52, 1.63, 3.88]],
        filename_format='%015d'
    )

    cfg = SimpleNamespace(dataset=dataset_cfg)

    dataset = Custom_Dataset('train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)
    max_iter = 25
    for batch_idx, (inputs, calib_mat, targets, info) in enumerate(dataloader):
        # test image
        if random.randint(0,4)<2:
            continue
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        if batch_idx > max_iter:
            break
