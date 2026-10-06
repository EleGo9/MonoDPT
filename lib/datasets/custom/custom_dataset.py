import os
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os, sys
sys.path.append(os.getcwd())
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
from lib.datasets.custom.pd import PhotometricDistort
from lib.datasets.custom.image_geometry import crop_image, make_image_transform, validate_model_resolution
from tqdm.auto import tqdm

DEBUG = True

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
        self.resolution = np.asarray(cfg.dataset.resolution, dtype=np.int32) # W * H

        # Configurable max objects
        self.max_objs = cfg.dataset.max_objs

        # Configurable filename format (e.g., '%06d' for 000000, '%08d' for 00000000)
        self.filename_format = cfg.dataset.filename_format[dataset_id] if cfg.dataset.filename_format is not None else '%06d'

        self.use_3d_center = cfg.dataset.use_3d_center
        self.writelist = cfg.dataset.writelist
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.dataset.bbox2d_type
        assert self.bbox2d_type in ['anno', 'proj']
        self.use_meanshape = cfg.dataset.use_meanshape
        self.class_merging = cfg.dataset.class_merging
        self.use_dontcare = cfg.dataset.use_dontcare
        self.depth_threshold = cfg.dataset.depth_threshold
        self.distortion = cfg.dataset.distortion

        if self.class_merging:
            self.writelist.extend(['Van', 'Truck'])
        if self.use_dontcare:
            self.writelist.extend(['DontCare'])

        # data split loading
        assert self.split in ['train', 'val', 'trainval', 'test', 'all']
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        if os.path.exists(self.split_file):
            self.idx_list = [x.strip() for x in open(self.split_file).readlines() if x.strip()]
        else:
            image_dir = os.path.join(self.root_dir, 'image_2')
            self.idx_list = [os.path.splitext(name)[0]
                             for name in sorted(os.listdir(image_dir))
                             if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not self.idx_list:
                raise FileNotFoundError(
                    f'No split file at {self.split_file} and no images found in {image_dir}')
            print(f'Split file not found: {self.split_file}; using {len(self.idx_list)} images from {image_dir}')

        # path configuration
        # self.data_dir = os.path.join(self.root_dir, 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.root_dir, 'image_2')
        self.calib_dir = os.path.join(self.root_dir, 'calib')
        self.label_dir = os.path.join(self.root_dir, 'label_2')
        self.original_resolution = np.asarray(cfg.dataset.original_resolution, dtype=np.int32)
        self.scale_factor = float(getattr(cfg.dataset, "scale_factor", 1.0))
        fits = bool(np.all(self.resolution <= self.original_resolution))
        if self.scale_factor != 1.0 and not fits:
            raise ValueError("scale_factor != 1 requires resolution <= original_resolution")
        self.crop_enabled = fits and (
            bool(np.any(self.resolution < self.original_resolution)) or self.scale_factor != 1.0
        )
        ref = self._image_transform(tuple(self.original_resolution))
        self.input_size = np.array([ref.target_width, ref.target_height], dtype=np.int32)
        validate_model_resolution(self.input_size, patch_size=14)
        print(
            f"[Custom_Dataset] crop {ref.crop_width:g}x{ref.crop_height:g} "
            f"at ({ref.crop_x0:g},{ref.crop_y0:g}) -> input {tuple(self.input_size)}, "
            f"scale {ref.scale:g}"
        )
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
        self.debug_geometry = bool(getattr(cfg.dataset, "debug_geometry", False))

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
        self.cls_mean_size = np.array(cfg.dataset.cls_mean_size, dtype=np.float32)  # [H, W, L]

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

        if not self.use_meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 32
        self.pd = PhotometricDistort()
        self.clip_2d = cfg.dataset.clip_2d
        self.kitti_official_eval = cfg.dataset.kitti_official_eval if cfg.dataset.kitti_official_eval is not None else False
    def _image_transform(self, source_size):
        return make_image_transform(
            source_size,
            self.resolution,
            scale_factor=self.scale_factor,
            allow_crop=self.crop_enabled,
        )

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
            image_size = self.get_image(index).size
            image_transform = self._image_transform(image_size)
            for obj in objects:
                # Apply the same filtering criteria as in __getitem__
                # if obj.cls_type not in self.writelist:
                #     continue
                # if obj.cls_type == 'DontCare':
                #     continue
                if obj.pos[-1] < 1:
                    continue
                if obj.pos[-1] > self.depth_threshold:
                    # print(obj.pos[-1])
                    continue
                
                center_3d = obj.pos + [0, -obj.h / 2, 0]
                center_3d = center_3d.reshape(-1, 3)
                center_3d, _ = calib.rect_to_img(center_3d)
                center_3d = image_transform.transform_points(center_3d)[0]

                if (0 <= center_3d[0] < self.input_size[0] and
                    0 <= center_3d[1] < self.input_size[1]):
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
        load_calib = Calibration(calib_file)
        if load_calib.D is None or np.all(load_calib.D == 0):
            print(f"Warning: No distortion parameters found in calibration file {calib_file}. Using default values.")
        return load_calib

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
            results_str, results_dict = get_indy_eval_result(gt_annos, dt_annos, self.writelist, max_depth= self.depth_threshold, iou_thres=getattr(self.cfg.dataset, "iou_thres", 0.5))

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
            pixel_multiple: If specified, pad the image so that its dimensions are multiples of this value.

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

    def preprocess_image(self, img):
        """Crop to the requested size when possible; otherwise keep letterboxing."""
        image_transform = self._image_transform(img.size)
        if image_transform.is_crop:
            img = crop_image(img, image_transform)
        else:
            img, _, _, _ = self.letterbox_resize(img, tuple(self.input_size.tolist()))
        return img, image_transform

    def transform_object_boxes(self, objects, image_transform, flip):
        """Transform source boxes once and report which crop boxes remain visible."""
        visible = []
        for obj in objects:
            box = image_transform.transform_box(
                obj.box2d,
                clip_crop=True,
                flip=flip,
            )
            visible.append(box is not None)
            obj.box2d = box if box is not None else np.zeros((4,), dtype=np.float32)

            if flip:
                obj.alpha = np.pi - obj.alpha
                obj.ry = np.pi - obj.ry
                if self.aug_calib:
                    obj.pos[0] *= -1
                if obj.alpha > np.pi:
                    obj.alpha -= 2 * np.pi
                if obj.alpha < -np.pi:
                    obj.alpha += 2 * np.pi
                if obj.ry > np.pi:
                    obj.ry -= 2 * np.pi
                if obj.ry < -np.pi:
                    obj.ry += 2 * np.pi
        return visible

    @staticmethod
    def debug_check_image_transform(source_calib, transformed_calib, image_transform):
        """Assert that P2 projection and unprojection follow the image transform."""
        points_rect = np.array(
            [[0.5, 0.1, 8.0], [-1.2, -0.4, 14.0], [2.1, 0.7, 32.0]],
            dtype=np.float32,
        )
        source_pixels, _ = source_calib.rect_to_img(points_rect)
        expected_pixels = image_transform.transform_points(source_pixels)
        transformed_pixels, _ = transformed_calib.rect_to_img(points_rect)
        np.testing.assert_allclose(transformed_pixels, expected_pixels, rtol=1e-5, atol=1e-4)

        if source_calib.D is None or not np.any(source_calib.D):
            recovered = transformed_calib.img_to_rect(
                transformed_pixels[:, 0], transformed_pixels[:, 1], points_rect[:, 2]
            )
            np.testing.assert_allclose(recovered, points_rect, rtol=1e-5, atol=1e-4)

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # image loading
        img = self.get_image(index)
        orig_ds = str(self.image_dir.split('/')[-2])
        orig_img_size = np.array(img.size)  # Original size (W, H)
        crop_before_augmentation = self._image_transform(img.size).is_crop
        features_size = self.input_size // self.downsample    # W * H

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
                # Cropped images are flipped after cropping so the crop offset
                # and the transformed labels refer to exactly the same pixels.
                if not crop_before_augmentation:
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
                            if random_flip_flag and not crop_before_augmentation:
                                img_temp = img_temp.transpose(Image.FLIP_LEFT_RIGHT)
                            img_blend = Image.blend(img, img_temp, alpha=0.5)
                            img = img_blend
                            break


        # DISABLED: Affine transformation (replaced with letterbox resize)
        # trans, trans_inv = get_affine_transform(center, crop_size, 0, self.input_size, inv=1)
        # img_transform = img.transform(tuple(self.input_size.tolist()),
        #                     method=Image.AFFINE,
        #                     data=tuple(trans_inv.reshape(-1).tolist()),
        #                     resample=Image.BILINEAR)
        # cv2.imshow('img', np.array(img_transform))
        # cv2.waitKey(0)

        # Crop and letterbox share one source-to-model image coordinate transform.
        img, image_transform = self.preprocess_image(img)
        if random_flip_flag and image_transform.is_crop:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        resize_scale = image_transform.scale
        # Legacy names are signed affine offsets: negative for crop, positive for pad.
        pad_w = image_transform.offset_x
        pad_h = image_transform.offset_y


        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W
        info = {'img_id': index,
                'img_size': self.input_size,  # After resize
                'orig_img_size': orig_img_size,  # Before resize
                'resize_scale': resize_scale,
                'pad_w': pad_w,
                'pad_h': pad_h,
                'image_offset_x': pad_w,
                'image_offset_y': pad_h,
                'crop_x0': image_transform.crop_x0,
                'crop_y0': image_transform.crop_y0,
                'crop_width': image_transform.crop_width,
                'crop_height': image_transform.crop_height,
                'crop_applied': image_transform.is_crop,
                'bbox_downsample_ratio': self.input_size / features_size,
                'orig_ds': orig_ds}
        # print('INFO',info)

        if self.split == 'test':
            calib = self.get_calib(index)
            source_calib = copy.deepcopy(calib)
            calib.apply_image_transform(resize_scale, pad_w, pad_h)
            if self.debug_geometry:
                self.debug_check_image_transform(source_calib, calib, image_transform)
            return img, calib.P2, img, info

        #  ============================   get labels   ==============================
        objects = self.get_label(index)
        calib = self.get_calib(index)

        source_calib = copy.deepcopy(calib)
        calib.apply_image_transform(resize_scale, pad_w, pad_h)
        if self.debug_geometry:
            self.debug_check_image_transform(source_calib, calib, image_transform)
        if random_flip_flag and self.aug_calib:
            calib.flip(self.input_size)
        objects_visible = self.transform_object_boxes(
            objects, image_transform, flip=random_flip_flag
        )

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

        # Dataset statistics for evaluation diagnostics
        stat_gt_cars = sum(1 for obj in objects if obj.cls_type == 'Car')
        stat_removed_by_depth = 0
        stat_rejected_proj = 0
        stat_skipped_encoding = 0

        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
        
        for i in range(object_num):
            if not objects_visible[i]:
                continue
            # filter objects by writelist
            if objects[i].cls_type not in self.writelist:
                # print('!',objects[i].cls_type)
                continue
            if objects[i].cls_type == 'DontCare':
                # print('!',objects[i].cls_type)
                continue
            
            # ignore the samples beyond the threshold
            if objects[i].pos[-1] > self.depth_threshold:
                if objects[i].cls_type in self.writelist:
                    stat_removed_by_depth += 1
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

            if image_transform.is_crop:
                # The transformed calibration maps directly into crop coordinates.
                center_3d, _ = calib.rect_to_img(center_3d)
                center_3d = center_3d[0]
                if random_flip_flag and not self.aug_calib:
                    center_3d[0] = self.input_size[0] - center_3d[0]
            else:
                # Keep the pre-existing center target path when no crop is needed.
                center_3d, _ = source_calib.rect_to_img(center_3d)
                center_3d = image_transform.transform_points(center_3d)[0]

            # Handle flipping if applied (flip should be applied AFTER resize)
            # DISABLED: Affine transformation (replaced with letterbox resize)
            # if random_flip_flag and not self.aug_calib:  # random flip for center3d
            #     center_3d[0] = orig_img_size[0] - center_3d[0]
            # center_3d = affine_transform(center_3d.reshape(-1), trans)

            # filter 3d center out of img
            proj_inside_img = True

            if center_3d[0] < 0 or center_3d[0] >= self.input_size[0]:
                proj_inside_img = False
            if center_3d[1] < 0 or center_3d[1] >= self.input_size[1]:
                proj_inside_img = False

            if proj_inside_img == False:
                stat_rejected_proj += 1
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

            center_2d_norm = center_2d / self.input_size
            size_2d_norm = size_2d[i] / self.input_size

            corner_2d_norm = corner_2d
            corner_2d_norm[0: 2] = corner_2d[0: 2] / self.input_size
            corner_2d_norm[2: 4] = corner_2d[2: 4] / self.input_size
            center_3d_norm = center_3d / self.input_size

            l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l = np.clip(l, 0, 1)
                    r = np.clip(r, 0, 1)
                    t = np.clip(t, 0, 1)
                    b = np.clip(b, 0, 1)
                else:
                    stat_skipped_encoding += 1
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
                objects_temp_visible = self.transform_object_boxes(
                    objects, image_transform, flip=random_flip_flag
                )

                object_num_temp = len(objects) if len(objects) < (self.max_objs - object_num) else (self.max_objs - object_num)
                for i in range(object_num_temp):
                    if not objects_temp_visible[i]:
                        continue
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
                    center_3d, _ = calib.rect_to_img(center_3d)
                    center_3d = center_3d[0]
                    if image_transform.is_crop and random_flip_flag and not self.aug_calib:
                        center_3d[0] = self.input_size[0] - center_3d[0]
                    # DISABLED: affine transformation
                    # if random_flip_flag and not self.aug_calib:  # random flip for center3d
                    #     center_3d[0] = orig_img_size[0] - center_3d[0]
                    # center_3d = affine_transform(center_3d.reshape(-1), trans)

                    # filter 3d center out of img
                    proj_inside_img = True

                    if center_3d[0] < 0 or center_3d[0] >= self.input_size[0]:
                        proj_inside_img = False
                    if center_3d[1] < 0 or center_3d[1] >= self.input_size[1]:
                        proj_inside_img = False

                    if proj_inside_img == False:
                            continue

                    # class
                    cls_id = self.cls2id[objects[i].cls_type]
                    labels[i + object_num] = cls_id

        
                    # encoding 2d/3d boxes
                    w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                    size_2d[i + object_num] = 1. * w, 1. * h

                    center_2d_norm = center_2d / self.input_size
                    size_2d_norm = size_2d[i + object_num] / self.input_size

                    corner_2d_norm = corner_2d
                    corner_2d_norm[0: 2] = corner_2d[0: 2] / self.input_size
                    corner_2d_norm[2: 4] = corner_2d[2: 4] / self.input_size
                    center_3d_norm = center_3d / self.input_size

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
                   'img_size': self.input_size,  # After resize
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
                'img_size': self.input_size,  # After resize
                'orig_img_size': orig_img_size,  # Before resize
                'resize_scale': resize_scale,
                'pad_w': pad_w,
                'pad_h': pad_h,
                'image_offset_x': pad_w,
                'image_offset_y': pad_h,
                'crop_x0': image_transform.crop_x0,
                'crop_y0': image_transform.crop_y0,
                'crop_width': image_transform.crop_width,
                'crop_height': image_transform.crop_height,
                'crop_applied': image_transform.is_crop,
                'bbox_downsample_ratio': self.input_size / features_size,
                'orig_ds': orig_ds,
                'stat_gt_cars': stat_gt_cars,
                'stat_removed_by_depth': stat_removed_by_depth,
                'stat_rejected_proj': stat_rejected_proj,
                'stat_skipped_encoding': stat_skipped_encoding}
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
                cx_px = int(cx * self.input_size[0])
                cy_px = int(cy * self.input_size[1])
                w_px = int(w * self.input_size[0])
                h_px = int(h * self.input_size[1])
                img_vis = img.copy()
                img_vis = np.transpose(img_vis, (1, 2, 0))
                xyxy_box = box_cxcywh_to_xyxy(torch.tensor([[cx_px, cy_px, w_px, h_px]]))
                xyxy_box = xyxy_box.numpy()[0]
                img_vis = draw_2d_boxes(img_vis, xyxy_box, color=(255, 0, 0))
                img_vis = img_vis*self.std + self.mean
                img_vis = (img_vis * 255).astype(np.uint8)
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
                cv2.imshow('2D boxes', img_vis)
                # cv2.waitKey(0)

            # 3d boxes visualization
            for box_2d, box3d, cal_mat, hwl, head_bin, head_res, dpt  in zip(boxes, boxes_3d, calibs, src_size_3d, heading_bin, heading_res, depth):
                img_vis3d = img.copy()
                img_vis3d = np.transpose(img_vis3d, (1, 2, 0))
                p2 = cal_mat
                cx, cy, l, r, t, b = box3d
                if not np.any(box_2d):
                    continue
                cx_px = int(cx * self.input_size[0])
                cy_px = int(cy * self.input_size[1])
                l = int(l * self.input_size[0])
                r = int(r * self.input_size[0])
                t = int(t * self.input_size[1])
                b = int(b * self.input_size[1])
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
                denorm_box_2d_center = box_2d[0] * self.input_size[0]
                ry = calib.alpha2ry(alpha, denorm_box_2d_center)
                
                verts_cur, _ = project_3d(calib, locations[0], locations[1]- dimens[0]/2, locations[2], dimens[1], dimens[0], dimens[2], ry[0], return_3d=True)
                try:
                    img_vis3d = draw_3d_box(img_vis3d, verts_cur, color= (255,0,0), thickness= 2)
                except:
                    print('draw_3d_box error')
                    continue
                cv2.imshow(f'3D visualization', img_vis3d)
                cv2.waitKey(0)

        return inputs, calib.P2, targets, info #TODO: check when this matrciz is used, it is maybe wrong!!!


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from lib.helpers.config_helper import Config
    from omegaconf import OmegaConf
    import argparse
    
    parser = argparse.ArgumentParser(description='Monocular 3D Object Detection with Decoupled-Query and Geometry-Error Priors')
    parser.add_argument('--config', dest='config', help='settings of detection in yaml format', default="configs/monodpt.yaml")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    # cfg = {
    #        'root_dir': '/path/to/dataset',
    #        'random_flip': 0.0, 'random_crop': 1.0, 'scale': 0.8, 'shift': 0.1, 'use_dontcare': False,
    #        'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False, 'depth_threshold': 100,}
    
    # Build Configuration Object
    config_file = OmegaConf.load(args.config)
    cfg = Config(**config_file)

    dataset = Custom_Dataset('train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)
    max_iter = 2000
    for batch_idx, (inputs, calib_mat, targets, info) in enumerate(dataloader):
        # test image
        if random.randint(0,4)<2:
            continue
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        if batch_idx > max_iter:
            break
