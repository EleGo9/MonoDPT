import copy
import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from lib.helpers.config_helper import Config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "monodpt_custom.yaml"
DATA_ROOT = Path("/media/matte/PinimSandisk/urRecs/tesi/export_small/training")
CONFIGS = [
    ((1274, 714), 1.0),
    ((1232, 672), 0.75),
    ((1260, 700), 0.5),
]


def load_training_config(resolution=None, scale_factor=None, *, legacy=False):
    # Match tools/train_val.py: OmegaConf.load followed by Config(**config_file).
    omega_cfg = OmegaConf.load(CONFIG_PATH)
    if resolution is not None:
        omega_cfg.dataset.resolution = list(resolution)
    if scale_factor is not None:
        omega_cfg.dataset.scale_factor = float(scale_factor)
    omega_cfg.dataset.debug_geometry = True
    omega_cfg.dataset.random_mixup3d = 0.0
    omega_cfg.dataset.aug_pd = False
    omega_cfg.dataset.aug_crop = False
    if legacy:
        config_data = OmegaConf.to_container(omega_cfg, resolve=True)
        config_data["dataset"].pop("scale_factor", None)
        return Config(**config_data)
    return Config(**omega_cfg)


def angle_error(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def valid_slots(targets):
    boxes = np.asarray(targets["boxes"])
    return np.flatnonzero((boxes[:, 2] > 0) & (boxes[:, 3] > 0))


def assert_sample(dataset, position, flip, aug_calib):
    image, _, targets, info = dataset[position]
    width, height = map(int, dataset.input_size)
    assert image.shape == (3, height, width)
    np.testing.assert_array_equal(info["img_size"], dataset.input_size)
    np.testing.assert_array_equal(targets["img_size"], dataset.input_size)
    assert np.isfinite(np.asarray(info["bbox_downsample_ratio"])).all()

    valid = valid_slots(targets)
    objects = dataset.get_label(int(info["img_id"]))
    source_calib = dataset.get_calib(int(info["img_id"]))
    transform = dataset._image_transform(tuple(info["orig_img_size"]))
    max_center_error = 0.0
    for slot in valid:
        center_3d = objects[slot].pos + [0, -objects[slot].h / 2, 0]
        source_pixel, _ = source_calib.rect_to_img(np.asarray(center_3d).reshape(1, 3))
        expected = transform.transform_points(source_pixel)[0]
        if flip:
            expected[0] = width - expected[0]
        target_pixel = np.asarray(targets["boxes_3d"][slot, :2]) * np.asarray([width, height])
        error = float(np.max(np.abs(target_pixel - expected)))
        max_center_error = max(max_center_error, error)
        if not aug_calib:
            assert error <= 1e-3, (
                f"center error {error} at image {info['img_id']}, slot {slot}, "
                f"flip={flip}, scale_factor={dataset.scale_factor}"
            )

        box = np.asarray(targets["boxes"][slot])
        box_center = box[:2] * np.asarray([width, height])
        box_size = np.asarray(targets["size_2d"][slot])
        assert 0 <= box_center[0] <= width
        assert 0 <= box_center[1] <= height
        assert box_size[0] > 0 and box_size[1] > 0
    return max_center_error


def test_scaled_custom_dataset_real_data():
    if not DATA_ROOT.is_dir():
        pytest.skip(f"custom dataset is unavailable at {DATA_ROOT}")

    from lib.datasets.custom.custom_dataset import Custom_Dataset

    init_seconds = {}
    getitem_seconds_200 = {}
    augmented_center_errors = {}
    datasets = {}

    for resolution, scale_factor in CONFIGS:
        started = time.perf_counter()
        dataset = Custom_Dataset(
            "train", load_training_config(resolution, scale_factor)
        )
        init_seconds[scale_factor] = time.perf_counter() - started
        datasets[scale_factor] = dataset
        assert len(dataset) >= 200
        dataset.random_flip = 0.0
        dataset.aug_calib = False
        started = time.perf_counter()
        for position in range(200):
            dataset[position]
        getitem_seconds_200[scale_factor] = time.perf_counter() - started
        ref = dataset._image_transform(tuple(dataset.original_resolution))
        print(
            f"INIT s={scale_factor:g}: [Custom_Dataset] crop "
            f"{ref.crop_width:g}x{ref.crop_height:g} at "
            f"({ref.crop_x0:g},{ref.crop_y0:g}) -> input "
            f"{tuple(dataset.input_size)}, scale {ref.scale:g}"
        )
        width, height = map(int, dataset.input_size)
        print(
            f"DEPTH s={scale_factor:g}: input={(width, height)}, "
            f"depth_target={(height // 16, width // 16)}, "
            f"decode_img_size={(width, height)}"
        )

        for flip in (0.0, 1.0):
            dataset.random_flip = flip
            for aug_calib in (False, True):
                dataset.aug_calib = aug_calib
                errors = []
                for position in range(200):
                    errors.append(
                        assert_sample(dataset, position, bool(flip), aug_calib)
                    )
                if aug_calib:
                    augmented_center_errors[(scale_factor, flip)] = max(errors)
        dataset.random_flip = 0.0
        dataset.aug_calib = False

        # The test split returns P2 after the same shared image transform.
        test_dataset = Custom_Dataset(
            "test", load_training_config(resolution, scale_factor)
        )
        _, returned_P2, _, test_info = test_dataset[0]
        source_calib = test_dataset.get_calib(int(test_info["img_id"]))
        source_calib.apply_image_transform(
            float(test_info["resize_scale"]),
            float(test_info["pad_w"]),
            float(test_info["pad_h"]),
        )
        np.testing.assert_array_equal(returned_P2, source_calib.P2)

        # The tester's inverse box operation recovers source boxes without crop clipping.
        recovered = 0
        for position in range(len(dataset)):
            sample_info = dataset[position][3]
            transform = dataset._image_transform(tuple(sample_info["orig_img_size"]))
            objects = dataset.get_label(int(sample_info["img_id"]))
            for obj in objects:
                x1, y1, x2, y2 = map(float, obj.box2d)
                if (
                    x1 < transform.crop_x0
                    or y1 < transform.crop_y0
                    or x2 > transform.crop_x0 + transform.crop_width
                    or y2 > transform.crop_y0 + transform.crop_height
                ):
                    continue
                p = transform.transform_box(obj.box2d).astype(np.float64)
                p[[0, 2]] = (p[[0, 2]] - transform.offset_x) / transform.scale
                p[[1, 3]] = (p[[1, 3]] - transform.offset_y) / transform.scale
                np.testing.assert_allclose(p, obj.box2d, rtol=0, atol=1e-3)
                recovered += 1
                if recovered >= 100:
                    break
            if recovered >= 100:
                break
        assert recovered == 100

        # Round-trip alpha through the transformed calibration on 100 real cars.
        checked = 0
        max_ry_error = 0.0
        for position in range(len(dataset)):
            info = dataset[position][3]
            transform = dataset._image_transform(tuple(info["orig_img_size"]))
            calib = dataset.get_calib(int(info["img_id"]))
            adjusted = copy.deepcopy(calib)
            adjusted.apply_image_transform(
                transform.scale, transform.offset_x, transform.offset_y
            )
            for obj in dataset.get_label(int(info["img_id"])):
                if obj.cls_type != "Car":
                    continue
                u = (float(obj.box2d[0]) + float(obj.box2d[2])) / 2
                alpha = calib.ry2alpha(obj.ry, u)
                transformed_u = transform.scale * u + transform.offset_x
                recovered_ry = adjusted.alpha2ry(alpha, transformed_u)
                max_ry_error = max(max_ry_error, angle_error(recovered_ry, obj.ry))
                checked += 1
                if checked == 100:
                    break
            if checked == 100:
                break
        assert checked == 100
        assert max_ry_error < 1e-6

    # Compare all six decoded calibration values from the old manual tester code
    # with apply_image_transform for 100 real calibrations at scale 1.
    s1_dataset = datasets[1.0]
    for position in range(100):
        index = int(s1_dataset.idx_list[position])
        calib_orig = s1_dataset.get_calib(index)
        transform = s1_dataset._image_transform(
            s1_dataset.get_image(index).size
        )
        old = copy.deepcopy(calib_orig)
        old.cu = calib_orig.cu * transform.scale + transform.offset_x
        old.cv = calib_orig.cv * transform.scale + transform.offset_y
        old.fu = calib_orig.fu * transform.scale
        old.fv = calib_orig.fv * transform.scale
        old.P2[0, 0] = old.fu
        old.P2[1, 1] = old.fv
        old.P2[0, 2] = old.cu
        old.P2[1, 2] = old.cv
        new = copy.deepcopy(calib_orig)
        new.apply_image_transform(
            transform.scale, transform.offset_x, transform.offset_y
        )
        for name in ("cu", "cv", "fu", "fv", "tx", "ty"):
            assert getattr(old, name) == getattr(new, name), (
                f"{name} differs for calibration {index}"
            )

    # Default DataLoader collation batches scalar metadata per sample.
    batch = next(iter(DataLoader(datasets[0.5], batch_size=2)))
    batch_info = batch[3]
    for key in ("pad_w", "pad_h", "resize_scale"):
        assert tuple(batch_info[key].shape) == (2,)
        assert torch.is_floating_point(batch_info[key])
    assert tuple(batch_info["crop_x0"].shape) == (2,)

    # Legacy configuration without scale_factor still uses the s=1 crop.
    legacy = Custom_Dataset("val", load_training_config(legacy=True))
    legacy_sample = legacy[0]
    assert legacy_sample[3]["pad_w"] == -3

    # Full post-filter training split counts: cropped boxes plus rejected centers,
    # and output boxes smaller than five pixels.
    for scale_factor, dataset in datasets.items():
        crop_removed = 0
        center_rejected = 0
        narrow_width = 0
        short_height = 0
        either_small = 0
        for position in range(len(dataset)):
            _, _, targets, info = dataset[position]
            center_rejected += int(info["stat_rejected_proj"])
            transform = dataset._image_transform(tuple(info["orig_img_size"]))
            objects = dataset.get_label(int(info["img_id"]))
            for obj in objects:
                if transform.transform_box(obj.box2d, clip_crop=True) is None:
                    crop_removed += 1
            slots = valid_slots(targets)
            sizes = np.asarray(targets["size_2d"])[slots]
            narrow_width += int(np.count_nonzero(sizes[:, 0] < 5))
            short_height += int(np.count_nonzero(sizes[:, 1] < 5))
            either_small += int(np.count_nonzero((sizes[:, 0] < 5) | (sizes[:, 1] < 5)))
        print(
            f"COUNTS s={scale_factor:g}: crop_removed={crop_removed}, "
            f"center_rejected={center_rejected}, total_removed={crop_removed + center_rejected}, "
            f"w_lt_5={narrow_width}, h_lt_5={short_height}, either_lt_5={either_small}"
        )
        print(
            f"TIME s={scale_factor:g}: init={init_seconds[scale_factor]:.3f}s, "
            f"getitem_200={getitem_seconds_200[scale_factor]:.3f}s"
        )

    for (scale_factor, flip), error in augmented_center_errors.items():
        print(
            f"AUG_CALIB_CENTER s={scale_factor:g} flip={int(flip)} "
            f"max_abs_error_px={error:.6f}"
        )


def test_default_config_and_test_calibration_equality():
    if not DATA_ROOT.is_dir():
        pytest.skip(f"custom dataset is unavailable at {DATA_ROOT}")
    from lib.datasets.custom.custom_dataset import Custom_Dataset

    cfg = load_training_config((1274, 714), 1.0)
    dataset = Custom_Dataset("test", cfg)
    _, returned_P2, _, info = dataset[0]
    source_calib = dataset.get_calib(int(info["img_id"]))
    source_calib.apply_image_transform(
        float(info["resize_scale"]), float(info["pad_w"]), float(info["pad_h"])
    )
    np.testing.assert_array_equal(returned_P2, source_calib.P2)
