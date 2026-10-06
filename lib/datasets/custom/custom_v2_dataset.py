"""Clean CUSTOM dataset implementation for MonoDPT.

Coordinate conventions used here:

* image sizes are ``(width, height)``;
* annotation boxes are ``xyxy`` pixel coordinates;
* dimensions are ``(height, width, length)`` in meters;
* annotation locations are camera-frame ``(x, y, z)`` meters at the box's
  bottom center; projected target centers are the geometric box centers.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset


# Resolve project imports from this file's location, so direct execution works
# from any current working directory.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from lib.datasets.custom.image_geometry import (  # noqa: E402
    crop_image,
    make_image_transform,
    validate_model_resolution,
)
from lib.datasets.kitti.kitti_utils import (  # noqa: E402
    Calibration,
    Object3d,
    get_objects_from_label,
)
from lib.datasets.utils import angle2class  # noqa: E402


ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class SampleRecord:
    """Paths and stable numeric ID for one fully labeled sample."""

    sample_id: int
    file_stem: str
    image_path: Path
    calibration_path: Path
    label_path: Path
    source_name: str


@dataclass(frozen=True)
class ObjectLabel:
    """Named semantic view of one label using the model's camera convention."""

    class_name: str
    truncation: float
    occlusion: float
    alpha_camera_rad: float
    bbox_xyxy_pixels: np.ndarray
    dimensions_hwl_m: np.ndarray
    location_bottom_xyz_m: np.ndarray
    rotation_y_camera_rad: float

    @classmethod
    def from_object3d(cls, obj: Object3d) -> "ObjectLabel":
        return cls(
            class_name=obj.cls_type,
            truncation=float(obj.trucation),
            occlusion=float(obj.occlusion),
            alpha_camera_rad=float(obj.alpha),
            bbox_xyxy_pixels=np.asarray(obj.box2d, dtype=np.float32).copy(),
            dimensions_hwl_m=np.asarray([obj.h, obj.w, obj.l], dtype=np.float32),
            location_bottom_xyz_m=np.asarray(obj.pos, dtype=np.float32).copy(),
            rotation_y_camera_rad=float(obj.ry),
        )


@dataclass
class EncodedTargets:
    """Fixed-size model targets before conversion to the public mapping."""

    labels: np.ndarray
    boxes: np.ndarray
    boxes_3d: np.ndarray
    depth: np.ndarray
    size_3d: np.ndarray
    src_size_3d: np.ndarray
    heading_bin: np.ndarray
    heading_res: np.ndarray
    mask_2d: np.ndarray
    obj_region: np.ndarray
    img_size: np.ndarray

    def as_mapping(self) -> dict[str, np.ndarray]:
        return {
            "labels": self.labels,
            "boxes": self.boxes,
            "boxes_3d": self.boxes_3d,
            "depth": self.depth,
            "size_3d": self.size_3d,
            "src_size_3d": self.src_size_3d,
            "heading_bin": self.heading_bin,
            "heading_res": self.heading_res,
            "mask_2d": self.mask_2d,
            "obj_region": self.obj_region,
            "img_size": self.img_size,
        }


class CustomV2Dataset(Dataset):
    """Labeled CUSTOM data for train/test with consistent image geometry."""

    IMAGENET_RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, split, cfg, root_dir=None, dataset_id=0):
        if split not in {"train", "test"}:
            raise ValueError(f"CustomV2Dataset supports only 'train' and 'test', got {split!r}")

        self.cfg = cfg
        self.split = split
        configured_root = cfg.dataset.root_dir if root_dir is None else root_dir
        if isinstance(configured_root, (list, tuple)):
            if len(configured_root) != 1:
                raise ValueError("Pass one root_dir per CustomV2Dataset instance")
            configured_root = configured_root[0]
        self.root_dir = Path(configured_root).expanduser()
        self.source_name = self.root_dir.name
        self.image_dir = self.root_dir / "image_2"
        self.calib_dir = self.root_dir / "calib"
        self.label_dir = self.root_dir / "label_2"
        self.split_file = self.root_dir / "ImageSets" / f"{split}.txt"

        dataset_cfg = cfg.dataset
        self.class_name = list(dataset_cfg.class_name or [])
        if not self.class_name or len(set(self.class_name)) != len(self.class_name):
            raise ValueError("dataset.class_name must contain unique class names")
        configured_cls2id = dataset_cfg.cls2id
        self.cls2id = (
            {name: index for index, name in enumerate(self.class_name)}
            if configured_cls2id is None
            else {str(name): int(index) for name, index in configured_cls2id.items()}
        )
        expected_cls2id = {name: index for index, name in enumerate(self.class_name)}
        if self.cls2id != expected_cls2id:
            raise ValueError(
                "cls2id must map class_name entries to matching contiguous indices; "
                f"expected {expected_cls2id}, got {self.cls2id}"
            )
        self.num_classes = len(self.class_name)
        model_num_classes = getattr(cfg.model, "num_classes", self.num_classes)
        if int(model_num_classes) != self.num_classes:
            raise ValueError(
                f"Model has {model_num_classes} classes but dataset.class_name has "
                f"{self.num_classes}"
            )

        self.writelist = list(dataset_cfg.writelist or [])
        unknown_writable = [name for name in self.writelist if name not in self.cls2id]
        if unknown_writable:
            raise ValueError(f"writelist contains unmapped classes: {unknown_writable}")
        self.max_objs = int(dataset_cfg.max_objs)
        if self.max_objs <= 0:
            raise ValueError("dataset.max_objs must be positive")
        self.depth_threshold = float(dataset_cfg.depth_threshold)
        self.clip_2d = bool(dataset_cfg.clip_2d)
        self.bbox2d_type = getattr(dataset_cfg, "bbox2d_type", "anno")
        if self.bbox2d_type not in {"anno", "proj"}:
            raise ValueError("dataset.bbox2d_type must be 'anno' or 'proj'")

        self.use_meanshape = bool(dataset_cfg.use_meanshape)
        if self.use_meanshape:
            means = np.asarray(dataset_cfg.cls_mean_size, dtype=np.float32)
            if means.shape != (self.num_classes, 3):
                raise ValueError(
                    "cls_mean_size must have one [H,W,L] row per class_name entry "
                    f"(class order is {self.class_name}); got shape {means.shape}"
                )
            if not np.isfinite(means).all() or np.any(means <= 0):
                raise ValueError("cls_mean_size must contain finite positive [H,W,L] values")
            self.cls_mean_size = means.copy()
        else:
            self.cls_mean_size = np.zeros((self.num_classes, 3), dtype=np.float32)

        self.resolution = np.asarray(dataset_cfg.resolution, dtype=np.int32)
        self.original_resolution = np.asarray(dataset_cfg.original_resolution, dtype=np.int32)
        if self.resolution.shape != (2,) or self.original_resolution.shape != (2,):
            raise ValueError("resolution and original_resolution must be [W,H]")
        if np.any(self.resolution <= 0) or np.any(self.original_resolution <= 0):
            raise ValueError("resolution and original_resolution must be positive")
        self.scale_factor = float(getattr(dataset_cfg, "scale_factor", 1.0))
        if not np.isfinite(self.scale_factor) or self.scale_factor <= 0:
            raise ValueError("scale_factor must be finite and positive")
        reference_transform = self._image_transform(tuple(self.original_resolution))
        self.input_size = np.asarray(
            [reference_transform.target_width, reference_transform.target_height], dtype=np.int32
        )
        validate_model_resolution(self.input_size, patch_size=14)

        self.filename_format = self._filename_format_for_root(dataset_cfg, dataset_id)
        self.kitti_official_eval = bool(getattr(dataset_cfg, "kitti_official_eval", False))
        self.iou_thres = float(getattr(dataset_cfg, "iou_thres", 0.5))
        self.data_augmentation = split == "train"
        self.random_flip = float(getattr(dataset_cfg, "random_flip", 0.0))
        if not 0.0 <= self.random_flip <= 1.0:
            raise ValueError("random_flip must be in [0, 1]")
        self._reject_unsupported_augmentations(dataset_cfg)

        self.records = self._index_samples()
        self.idx_list = [record.file_stem for record in self.records]
        self._records_by_id = {record.sample_id: record for record in self.records}
        if not self.records:
            raise ValueError(f"No fully labeled samples found under {self.root_dir}")

    @staticmethod
    def _filename_format_for_root(dataset_cfg, dataset_id):
        formats = getattr(dataset_cfg, "filename_format", None)
        if formats is None:
            return "%06d"
        if isinstance(formats, str):
            return formats
        if dataset_id >= len(formats):
            raise ValueError(
                f"No filename_format configured for dataset index {dataset_id}"
            )
        return str(formats[dataset_id])

    @staticmethod
    def _reject_unsupported_augmentations(dataset_cfg):
        unsupported_flags = []
        for name in ("aug_pd", "aug_crop", "aug_calib"):
            if bool(getattr(dataset_cfg, name, False)):
                unsupported_flags.append(name)
        if float(getattr(dataset_cfg, "random_mixup3d", 0.0)) > 0:
            unsupported_flags.append("random_mixup3d")
        if unsupported_flags:
            raise ValueError(
                "CustomV2Dataset supports horizontal flip only; disable unsupported "
                f"augmentation settings: {', '.join(unsupported_flags)}"
            )

    def _image_transform(self, source_size):
        return make_image_transform(
            source_size,
            tuple(int(value) for value in self.resolution),
            scale_factor=self.scale_factor,
            allow_crop=bool(np.all(self.resolution <= np.asarray(source_size))),
            patch_size=14,
        )

    def _index_samples(self):
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.calib_dir.is_dir():
            raise FileNotFoundError(f"Calibration directory does not exist: {self.calib_dir}")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"Label directory does not exist: {self.label_dir}")

        images_by_stem = {}
        for path in sorted(self.image_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            if path.stem in images_by_stem:
                raise ValueError(f"Multiple image files share sample stem {path.stem!r}")
            images_by_stem[path.stem] = path
        if self.split_file.is_file():
            requested_stems = [
                Path(line.strip()).stem
                for line in self.split_file.read_text().splitlines()
                if line.strip()
            ]
            split_list_exists = True
        else:
            requested_stems = sorted(images_by_stem)
            split_list_exists = False

        records = []
        missing = []
        seen_ids = set()
        for requested_stem in requested_stems:
            try:
                sample_id = int(requested_stem)
            except ValueError as exc:
                raise ValueError(
                    f"Sample ID {requested_stem!r} is not numeric; tester metadata "
                    "requires numeric img_id values"
                ) from exc
            if sample_id in seen_ids:
                raise ValueError(f"Split {self.split!r} contains duplicate sample ID {sample_id}")
            seen_ids.add(sample_id)
            formatted_stem = self.filename_format % sample_id
            image_path = images_by_stem.get(requested_stem) or images_by_stem.get(formatted_stem)
            actual_stem = image_path.stem if image_path is not None else formatted_stem
            if split_list_exists:
                candidate_stems = list(dict.fromkeys((requested_stem, actual_stem, formatted_stem)))
            else:
                # Without ImageSets, index the physical matching file stems.
                candidate_stems = [requested_stem]
            calibration_path = next(
                (self.calib_dir / f"{stem}.txt" for stem in candidate_stems
                 if (self.calib_dir / f"{stem}.txt").is_file()),
                self.calib_dir / f"{actual_stem}.txt",
            )
            label_path = next(
                (self.label_dir / f"{stem}.txt" for stem in candidate_stems
                 if (self.label_dir / f"{stem}.txt").is_file()),
                self.label_dir / f"{actual_stem}.txt",
            )
            missing_paths = []
            if image_path is None:
                missing_paths.append("image")
            if not calibration_path.is_file():
                missing_paths.append("calibration")
            if not label_path.is_file():
                missing_paths.append("label")
            if missing_paths:
                missing.append(f"{requested_stem}: {', '.join(missing_paths)}")
                continue
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    file_stem=formatted_stem,
                    image_path=image_path,
                    calibration_path=calibration_path,
                    label_path=label_path,
                    source_name=self.source_name,
                )
            )
        if missing:
            preview = "; ".join(missing[:8])
            remaining = len(missing) - min(8, len(missing))
            suffix = f"; and {remaining} more" if remaining else ""
            raise FileNotFoundError(
                f"Split {self.split!r} has samples without matching image/calibration/label "
                f"files under {self.root_dir}: {preview}{suffix}"
            )
        return records

    def __len__(self):
        return len(self.records)

    def _record(self, sample_id):
        sample_id = int(sample_id)
        try:
            return self._records_by_id[sample_id]
        except KeyError as exc:
            raise KeyError(f"Sample ID {sample_id} is not in split {self.split!r}") from exc

    def get_image(self, sample_id):
        record = self._record(sample_id)
        with Image.open(record.image_path) as image:
            return image.convert("RGB")

    def get_label(self, sample_id):
        return self._load_labels_with_camera_alpha(sample_id, self.get_calib(sample_id))

    def get_calib(self, sample_id):
        return Calibration(str(self._record(sample_id).calibration_path))

    @staticmethod
    def _reflect_calibration(calib, image_width):
        """Reflect projected pixels and camera X in one analytic update."""
        pixel_reflection = np.asarray(
            [[-1.0, 0.0, float(image_width)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        camera_reflection = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        calib.set_projection_matrix(pixel_reflection @ calib.P2 @ camera_reflection)
        # Brown-Conrady p2 changes sign under X reflection; radial and p1 terms do not.
        if calib.D is not None and len(calib.D) > 3:
            calib.D = np.asarray(calib.D, dtype=np.float32).copy()
            calib.D[3] *= -1.0

    @staticmethod
    def _wrapped_angle(angle):
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _load_labels_with_camera_alpha(self, sample_id, source_calib):
        """Load KITTI-convention labels and derive alpha from camera yaw."""
        objects = get_objects_from_label(str(self._record(sample_id).label_path))
        for obj in objects:
            if (
                obj.cls_type == "DontCare"
                or not np.isfinite(obj.ry)
                or abs(obj.ry) > 2.0 * np.pi
            ):
                continue

            # The source alpha column is a placeholder (zero for all current
            # CUSTOM cars). Derive observation angle from KITTI camera yaw and
            # the projected geometric center instead.
            if (
                not np.isfinite(obj.h)
                or obj.h <= 0
                or not np.isfinite(obj.pos).all()
                or obj.pos[2] <= 0
            ):
                continue
            center_xyz = obj.pos.astype(np.float32, copy=True)
            center_xyz[1] -= obj.h / 2.0  # labels store the bottom center
            center_pixel, center_depth = source_calib.rect_to_img(center_xyz[None, :])
            if (
                np.isfinite(center_pixel[0]).all()
                and np.isfinite(center_depth[0])
                and center_depth[0] > 0
            ):
                obj.alpha = self._wrapped_angle(
                    obj.ry
                    - np.arctan2(
                        center_pixel[0, 0] - source_calib.cu, source_calib.fu
                    )
                )
        return objects

    def _recompute_eval_ground_truth_alpha(self, annotations, image_ids):
        """Derive evaluator alpha values while preserving KITTI camera yaw."""
        for sample_id, annotation in zip(image_ids, annotations):
            names = np.asarray(annotation["name"])
            source_yaws = np.asarray(annotation["rotation_y"], dtype=np.float32)
            alphas = np.asarray(annotation["alpha"], dtype=np.float32).copy()
            dimensions_lhw = np.asarray(annotation["dimensions"], dtype=np.float32)
            locations = np.asarray(annotation["location"], dtype=np.float32)
            calib = self.get_calib(sample_id)

            for index, (name, source_yaw) in enumerate(zip(names, source_yaws)):
                if (
                    name == "DontCare"
                    or not np.isfinite(source_yaw)
                    or abs(source_yaw) > 2.0 * np.pi
                ):
                    continue
                rotation_y = float(source_yaw)
                height = float(dimensions_lhw[index, 1])
                if (
                    not np.isfinite(height)
                    or height <= 0
                    or not np.isfinite(locations[index]).all()
                    or locations[index, 2] <= 0
                ):
                    continue
                center_xyz = locations[index].copy()
                center_xyz[1] -= height / 2.0
                center_pixel, center_depth = calib.rect_to_img(center_xyz[None, :])
                if (
                    np.isfinite(center_pixel[0]).all()
                    and np.isfinite(center_depth[0])
                    and center_depth[0] > 0
                ):
                    theta = np.arctan2(
                        center_pixel[0, 0] - calib.cu, calib.fu
                    )
                    alphas[index] = self._wrapped_angle(
                        rotation_y - theta
                    )

            annotation["alpha"] = alphas
        return annotations

    def _source_box(self, obj, label, source_calib):
        if self.bbox2d_type == "anno":
            return label.bbox_xyxy_pixels.copy()
        corners = obj.generate_corners3d().astype(np.float32)
        pixels, _ = source_calib.rect_to_img(corners)
        return np.asarray(
            [pixels[:, 0].min(), pixels[:, 1].min(), pixels[:, 0].max(), pixels[:, 1].max()],
            dtype=np.float32,
        )

    def _empty_targets(self, width, height):
        return EncodedTargets(
            labels=np.zeros((self.max_objs,), dtype=np.int64),
            boxes=np.zeros((self.max_objs, 4), dtype=np.float32),
            boxes_3d=np.zeros((self.max_objs, 6), dtype=np.float32),
            depth=np.zeros((self.max_objs, 1), dtype=np.float32),
            size_3d=np.zeros((self.max_objs, 3), dtype=np.float32),
            src_size_3d=np.zeros((self.max_objs, 3), dtype=np.float32),
            heading_bin=np.zeros((self.max_objs, 1), dtype=np.int64),
            heading_res=np.zeros((self.max_objs, 1), dtype=np.float32),
            mask_2d=np.zeros((self.max_objs,), dtype=np.bool_),
            obj_region=np.zeros((height, width), dtype=np.bool_),
            img_size=np.asarray([width, height], dtype=np.int32),
        )

    def _encode_targets(
        self, objects, source_calib, transform, transformed_calib, image_size, flip
    ):
        width, height = image_size
        targets = self._empty_targets(width, height)
        slot = 0
        stat_gt = 0
        stat_depth = 0
        stat_projection = 0
        stat_encoding = 0

        for obj in objects:
            if obj.cls_type == "Car":
                stat_gt += 1
            if obj.cls_type not in self.writelist or obj.cls_type not in self.cls2id:
                continue

            label = ObjectLabel.from_object3d(obj)
            box = transform.transform_box(
                self._source_box(obj, label, source_calib),
                clip_crop=True,
                flip=flip,
            )
            if box is None:
                continue
            box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(width))
            box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(height))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            bottom_xyz = label.location_bottom_xyz_m.copy()
            heading_alpha = label.alpha_camera_rad
            if flip:
                bottom_xyz[0] *= -1.0
                heading_alpha = self._wrapped_angle(np.pi - heading_alpha)

            depth = float(bottom_xyz[2])
            if not np.isfinite(depth) or depth <= 0 or depth > self.depth_threshold:
                stat_depth += 1
                continue
            dimensions_hwl = label.dimensions_hwl_m
            if not np.isfinite(dimensions_hwl).all() or np.any(dimensions_hwl <= 0):
                stat_encoding += 1
                continue

            # Source label pos is at the ground contact point. The model learns
            # the geometric center of the 3D cuboid, half a height above it.
            center_xyz = bottom_xyz.copy()
            center_xyz[1] -= dimensions_hwl[0] / 2.0
            projected_center, _ = transformed_calib.rect_to_img(center_xyz.reshape(1, 3))
            center_xy = np.asarray(projected_center[0], dtype=np.float32)
            if not np.isfinite(center_xy).all() or not (
                0.0 <= center_xy[0] < width and 0.0 <= center_xy[1] < height
            ):
                stat_projection += 1
                continue

            x1, y1, x2, y2 = box
            box_width, box_height = x2 - x1, y2 - y1
            if box_width <= 0 or box_height <= 0:
                continue
            box_center = np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
            box_norm = np.asarray(
                [box_center[0] / width, box_center[1] / height, box_width / width, box_height / height],
                dtype=np.float32,
            )
            box_edges_norm = np.asarray([x1 / width, y1 / height, x2 / width, y2 / height])
            center_norm = center_xy / np.asarray([width, height], dtype=np.float32)
            left = center_norm[0] - box_edges_norm[0]
            right = box_edges_norm[2] - center_norm[0]
            top = center_norm[1] - box_edges_norm[1]
            bottom = box_edges_norm[3] - center_norm[1]
            offsets = np.asarray([left, right, top, bottom], dtype=np.float32)
            if np.any(offsets < 0):
                if not self.clip_2d:
                    stat_encoding += 1
                    continue
                offsets = np.clip(offsets, 0.0, 1.0)

            if slot >= self.max_objs:
                continue
            cls_id = self.cls2id[label.class_name]
            mean_hwl = self.cls_mean_size[cls_id]
            heading_alpha = self._wrapped_angle(heading_alpha)
            heading_class, heading_residual = angle2class(heading_alpha)

            targets.labels[slot] = cls_id
            targets.boxes[slot] = box_norm
            targets.boxes_3d[slot] = np.asarray(
                [center_norm[0], center_norm[1], *offsets], dtype=np.float32
            )
            targets.depth[slot, 0] = depth
            targets.src_size_3d[slot] = dimensions_hwl
            targets.size_3d[slot] = dimensions_hwl - mean_hwl
            targets.heading_bin[slot, 0] = heading_class
            targets.heading_res[slot, 0] = heading_residual
            targets.mask_2d[slot] = label.truncation <= 0.5 and label.occlusion <= 2

            left_px = max(0, int(np.floor(x1)))
            right_px = min(width, int(np.ceil(x2)))
            top_px = max(0, int(np.floor(y1)))
            bottom_px = min(height, int(np.ceil(y2)))
            targets.obj_region[top_px:bottom_px, left_px:right_px] = True
            slot += 1

        stats = {
            "stat_gt_cars": stat_gt,
            "stat_removed_by_depth": stat_depth,
            "stat_rejected_proj": stat_projection,
            "stat_skipped_encoding": stat_encoding,
        }
        return targets, stats

    def _transform_image(self, image, transform, flip):
        if transform.is_crop:
            image = crop_image(image, transform)
        else:
            resized_width = int(transform.source_width * transform.scale)
            resized_height = int(transform.source_height * transform.scale)
            image = image.resize(
                (resized_width, resized_height), Image.Resampling.BILINEAR
            )
            fill = tuple(
                np.rint(self.IMAGENET_RGB_MEAN * 255.0).astype(np.uint8).tolist()
            )
            canvas = Image.new(
                "RGB", (transform.target_width, transform.target_height), color=fill
            )
            canvas.paste(image, (int(transform.offset_x), int(transform.offset_y)))
            image = canvas
        if flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image

    def __getitem__(self, item):
        record = self.records[item]
        image = self.get_image(record.sample_id)
        original_size = tuple(image.size)
        transform = self._image_transform(original_size)

        flip = self.data_augmentation and np.random.random() < self.random_flip
        source_calib = self.get_calib(record.sample_id)
        transformed_calib = copy.deepcopy(source_calib)
        transformed_calib.apply_image_transform(
            transform.scale, transform.offset_x, transform.offset_y
        )
        if flip:
            self._reflect_calibration(transformed_calib, transform.target_width)

        objects = self._load_labels_with_camera_alpha(record.sample_id, source_calib)
        encoded, stats = self._encode_targets(
            objects,
            source_calib,
            transform,
            transformed_calib,
            (transform.target_width, transform.target_height),
            flip,
        )

        image = self._transform_image(image, transform, flip)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - self.IMAGENET_RGB_MEAN) / self.IMAGENET_RGB_STD
        image_array = np.ascontiguousarray(image_array.transpose(2, 0, 1), dtype=np.float32)

        info = {
            "img_id": record.sample_id,
            "img_size": encoded.img_size.copy(),
            "orig_img_size": np.asarray(original_size, dtype=np.int32),
            "resize_scale": np.float32(transform.scale),
            "pad_w": np.float32(transform.offset_x),
            "pad_h": np.float32(transform.offset_y),
            "image_offset_x": np.float32(transform.offset_x),
            "image_offset_y": np.float32(transform.offset_y),
            "crop_x0": np.float32(transform.crop_x0),
            "crop_y0": np.float32(transform.crop_y0),
            "crop_width": np.float32(transform.crop_width),
            "crop_height": np.float32(transform.crop_height),
            "crop_applied": bool(transform.is_crop),
            "flip": bool(flip),
            "orig_ds": record.source_name,
            **{key: int(value) for key, value in stats.items()},
        }
        return image_array, transformed_calib.P2.astype(np.float32, copy=True), encoded.as_mapping(), info

    def eval(self, results_dir, logger=None):
        """Evaluate result files using the configured custom/KITTI evaluator."""
        from lib.datasets.indy.indy_eval_python import indy_common
        from lib.datasets.indy.indy_eval_python.eval import get_indy_eval_result

        image_ids = [record.sample_id for record in self.records]
        if self.kitti_official_eval:
            from lib.datasets.kitti.kitti_eval_python import kitti_common
            from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result

            gt_annos = kitti_common.get_label_annos(self.label_dir, image_ids)
            gt_annos = self._recompute_eval_ground_truth_alpha(gt_annos, image_ids)
            dt_annos = kitti_common.get_label_annos(results_dir, image_ids)
            class_ids = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
            car_moderate = 0.0
            average_2d = 0.0
            evaluated = 0
            for class_name in self.writelist:
                if class_name not in class_ids:
                    raise ValueError(
                        f"KITTI official evaluation does not define class {class_name!r}"
                    )
                result_text, result_dict, map_3d = get_official_eval_result(
                    gt_annos, dt_annos, class_ids[class_name]
                )
                if logger is not None:
                    logger.info(result_text)
                else:
                    print(result_text)
                if class_name == "Car":
                    car_moderate = map_3d
                average_2d += float(
                    result_dict.get(f"{class_name}_image_moderate_R40", 0.0)
                )
                evaluated += 1
            return car_moderate, average_2d / max(evaluated, 1)

        gt_annos = indy_common.get_label_annos(
            self.label_dir, image_ids, filename_format=self.filename_format
        )
        gt_annos = self._recompute_eval_ground_truth_alpha(gt_annos, image_ids)
        dt_annos = indy_common.get_label_annos(
            results_dir, image_ids, filename_format=self.filename_format
        )
        result_text, result_dict = get_indy_eval_result(
            gt_annos,
            dt_annos,
            self.writelist,
            max_depth=self.depth_threshold,
            iou_thres=self.iou_thres,
        )
        if logger is not None:
            logger.info(result_text)
        else:
            print(result_text)
        return result_dict["3d mAP"], result_dict["2d mAP"]


def render_debug_sample(image_chw, objects, source_calib, info, *, class_names=None):
    """Render transformed 2D and projected 3D labels without opening a GUI."""
    image = np.asarray(image_chw, dtype=np.float32).transpose(1, 2, 0)
    image = np.clip(
        (
            image * CustomV2Dataset.IMAGENET_RGB_STD
            + CustomV2Dataset.IMAGENET_RGB_MEAN
        )
        * 255.0,
        0,
        255,
    )
    canvas = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    width, height = map(int, info["img_size"])
    scale = float(info["resize_scale"])
    offset_x, offset_y = float(info["pad_w"]), float(info["pad_h"])
    flip = bool(info.get("flip", False))
    edge_pairs = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )

    for obj in objects:
        box = np.asarray(obj.box2d, dtype=np.float32).copy()
        box[[0, 2]] = box[[0, 2]] * scale + offset_x
        box[[1, 3]] = box[[1, 3]] * scale + offset_y
        if flip:
            box[0], box[2] = width - box[2], width - box[0]
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
        if box[2] > box[0] and box[3] > box[1]:
            cv2.rectangle(
                canvas,
                (int(round(box[0])), int(round(box[1]))),
                (int(round(box[2])), int(round(box[3]))),
                (0, 220, 0),
                2,
            )
            if class_names is not None and obj.cls_type in class_names:
                cv2.putText(
                    canvas,
                    obj.cls_type,
                    (int(round(box[0])), max(12, int(round(box[1])) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 220, 0),
                    1,
                    cv2.LINE_AA,
                )

        corners3d = obj.generate_corners3d().astype(np.float32)
        # Use the same calibrated projection as target-center encoding; this
        # includes the camera's distortion coefficients for the source image.
        corners2d, _ = source_calib.rect_to_img(corners3d)
        corners2d = corners2d * scale + np.asarray([offset_x, offset_y], dtype=np.float32)
        if flip:
            # Reflecting projected vertices is equivalent to negating camera X
            # and changing yaw from ry to pi-ry before projection.
            corners2d[:, 0] = width - corners2d[:, 0]
        corners2d[:, 0] = np.clip(corners2d[:, 0], -width, 2 * width)
        corners2d[:, 1] = np.clip(corners2d[:, 1], -height, 2 * height)
        corners2d = np.round(corners2d).astype(np.int32)
        for start, end in edge_pairs:
            cv2.line(canvas, tuple(corners2d[start]), tuple(corners2d[end]), (0, 180, 255), 2)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _load_config(path):
    from lib.helpers.config_helper import Config

    return Config.load_from_yaml(str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Interactively inspect CUSTOM dataset samples")
    parser.add_argument(
        "--config", type=Path, default=_REPOSITORY_ROOT / "configs" / "monodpt_custom.yaml"
    )
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--start", type=int, default=0, help="Sample position within the split")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    roots = cfg.dataset.root_dir_train if args.split == "train" else cfg.dataset.root_dir_test
    if not roots:
        roots = cfg.dataset.root_dir
    root = roots[0] if isinstance(roots, (list, tuple)) else roots
    dataset = CustomV2Dataset(args.split, cfg, root_dir=root)
    index = args.start % len(dataset)
    window = "CustomV2Dataset: ← previous, → next, q/esc quit (p/n also work)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            image, _, _, info = dataset[index]
            sample_id = int(info["img_id"])
            rendered = render_debug_sample(
                image,
                dataset.get_label(sample_id),
                dataset.get_calib(sample_id),
                info,
                class_names=dataset.class_name,
            )
            cv2.imshow(window, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
            key = cv2.waitKeyEx(0)
            if key in (ord("q"), ord("Q"), 27):
                break
            # waitKeyEx returns platform-specific virtual key values for arrows.
            # Keep the short legacy codes too, for OpenCV backends that return them.
            if key in (ord("n"), 83, 2555904, 65363, 63235):
                index = (index + 1) % len(dataset)
            elif key in (ord("p"), 81, 2424832, 65361, 63234):
                index = (index - 1) % len(dataset)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
