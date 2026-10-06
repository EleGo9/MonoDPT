from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from lib.datasets.custom.custom_v2_dataset import (
    CustomV2Dataset,
    render_debug_sample,
)
from lib.datasets.custom.image_geometry import make_image_transform
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.utils import class2angle


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "monodpt_custom.yaml"
DATA_ROOT = Path("/media/matte/PinimSandisk/urRecs/tesi/export_small")
TARGET_KEYS = {
    "labels",
    "boxes",
    "boxes_3d",
    "depth",
    "size_3d",
    "src_size_3d",
    "heading_bin",
    "heading_res",
    "mask_2d",
    "obj_region",
    "img_size",
}


def make_cfg(
    root,
    *,
    resolution=(1274, 714),
    scale_factor=1.0,
    use_meanshape=True,
    random_flip=0.0,
    max_objs=5,
    aug_pd=False,
    aug_crop=False,
    aug_calib=False,
    random_mixup3d=0.0,
    filename_format="%06d",
    cls2id=None,
):
    dataset = SimpleNamespace(
        root_dir=Path(root),
        class_name=["Car"],
        cls2id=cls2id,
        writelist=["Car"],
        max_objs=max_objs,
        depth_threshold=120.0,
        clip_2d=False,
        bbox2d_type="anno",
        use_meanshape=use_meanshape,
        cls_mean_size=[[1.0, 1.92, 5.235]],
        resolution=list(resolution),
        original_resolution=[1280, 720],
        scale_factor=scale_factor,
        filename_format=[filename_format],
        kitti_official_eval=False,
        iou_thres=0.5,
        random_flip=random_flip,
        aug_pd=aug_pd,
        aug_crop=aug_crop,
        aug_calib=aug_calib,
        random_mixup3d=random_mixup3d,
    )
    model = SimpleNamespace(num_classes=1)
    return SimpleNamespace(dataset=dataset, model=model)


def write_calibration(path):
    p2 = "800 0 640 0 0 800 360 0 0 0 1 0"
    p3 = "800 0 640 0 0 800 360 0 0 0 1 0"
    r0 = "1 0 0 0 1 0 0 0 1"
    tr = "1 0 0 0 0 1 0 0 0 0 1 0"
    path.write_text(
        "P0: 0 0 0 0\n"
        "P1: 0 0 0 0\n"
        f"P2: {p2}\n"
        f"P3: {p3}\n"
        f"R0_rect: {r0}\n"
        f"Tr_velo_to_cam: {tr}\n"
        "unused: 0\n"
        "D: 0 0 0 0 0\n"
    )


def label_line(
    *, x=0.0, y=1.6, z=20.0, h=1.6, w=1.8, length=4.0,
    box=(570.0, 320.0, 710.0, 460.0), truncation=0.0, occlusion=0.0,
    alpha=0.0, rotation_y=0.0,
):
    return (
        f"Car {truncation} {occlusion} {alpha} "
        f"{box[0]} {box[1]} {box[2]} {box[3]} "
        f"{h} {w} {length} {x} {y} {z} {rotation_y}\n"
    )


@pytest.fixture
def labeled_roots(tmp_path):
    roots = {}
    for split in ("training", "testing"):
        root = tmp_path / split
        for directory in ("image_2", "calib", "label_2"):
            (root / directory).mkdir(parents=True)
        roots[split] = root

        # The full color field makes RGB order and normalization easy to check.
        Image.new("RGB", (1280, 720), (255, 0, 0)).save(root / "image_2" / "000001.png")
        Image.new("RGB", (1280, 720), (10, 30, 90)).save(root / "image_2" / "000002.png")
        write_calibration(root / "calib" / "000001.txt")
        write_calibration(root / "calib" / "000002.txt")
        (root / "label_2" / "000001.txt").write_text(label_line())
        # A labeled test image with no trainable objects remains a real sample.
        (root / "label_2" / "000002.txt").write_text("")
    return roots


@pytest.mark.parametrize(
    "resolution,scale_factor,expected_size",
    [((1274, 714), 1.0, (1274, 714)), ((1232, 672), 0.75, (924, 504)), ((1260, 700), 0.5, (630, 350))],
)
def test_transform_scale_centers_calibration_and_fixed_targets(
    labeled_roots, resolution, scale_factor, expected_size
):
    dataset = CustomV2Dataset(
        "train",
        make_cfg(labeled_roots["training"], resolution=resolution, scale_factor=scale_factor),
    )
    assert len(dataset) == 2
    image, p2, targets, info = dataset[0]
    width, height = expected_size
    assert image.shape == (3, height, width)
    assert image.dtype == np.float32
    assert p2.shape == (3, 4) and p2.dtype == np.float32
    assert set(targets) == TARGET_KEYS
    assert targets["labels"].shape == (5,) and targets["labels"].dtype == np.int64
    assert targets["boxes"].shape == (5, 4) and targets["boxes"].dtype == np.float32
    assert targets["boxes_3d"].shape == (5, 6)
    assert targets["depth"].shape == (5, 1)
    assert targets["size_3d"].shape == (5, 3)
    assert targets["src_size_3d"].shape == (5, 3)
    assert targets["heading_bin"].shape == (5, 1)
    assert targets["heading_res"].shape == (5, 1)
    assert targets["mask_2d"].dtype == np.bool_
    assert targets["obj_region"].shape == (height, width)
    np.testing.assert_array_equal(targets["img_size"], [width, height])
    np.testing.assert_array_equal(info["img_size"], [width, height])

    # The model input is RGB and ImageNet normalized.
    np.testing.assert_allclose(
        image[:, height // 2, width // 2],
        [(1.0 - 0.485) / 0.229, (0.0 - 0.456) / 0.224, (0.0 - 0.406) / 0.225],
        atol=1e-5,
    )
    np.testing.assert_array_equal(targets["mask_2d"], [True, False, False, False, False])
    np.testing.assert_array_equal(targets["labels"][:1], [0])
    np.testing.assert_allclose(targets["depth"][0], [20.0])
    np.testing.assert_allclose(targets["src_size_3d"][0], [1.6, 1.8, 4.0])
    np.testing.assert_allclose(targets["size_3d"][0], [0.6, -0.12, -1.235], atol=1e-5)

    transform = make_image_transform((1280, 720), resolution, scale_factor=scale_factor)
    source_calib = dataset.get_calib(1)
    geometric_center = np.asarray([[0.0, 1.6 - 1.6 / 2, 20.0]], dtype=np.float32)
    source_pixel, _ = source_calib.rect_to_img(geometric_center)
    expected_pixel = transform.transform_points(source_pixel)[0]
    target_pixel = targets["boxes_3d"][0, :2] * np.asarray([width, height])
    np.testing.assert_allclose(target_pixel, expected_pixel, rtol=0, atol=1e-4)
    np.testing.assert_allclose(targets["depth"][0, 0], geometric_center[0, 2])

    transformed = Calibration(str(dataset._record(1).calibration_path))
    transformed.apply_image_transform(transform.scale, transform.offset_x, transform.offset_y)
    np.testing.assert_allclose(p2, transformed.P2, atol=1e-6)
    projected, _ = transformed.rect_to_img(geometric_center)
    np.testing.assert_allclose(projected[0], expected_pixel, rtol=0, atol=1e-4)


def test_empty_targets_and_default_dataloader_collation(labeled_roots):
    dataset = CustomV2Dataset("test", make_cfg(labeled_roots["testing"]))
    image, _, targets, info = dataset[1]
    assert image.shape == (3, 714, 1274)
    assert not targets["mask_2d"].any()
    assert not targets["obj_region"].any()
    assert info["img_id"] == 2

    for batch_size in (1, 2):
        batch = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=False)))
        images, calibs, batched_targets, batched_info = batch
        assert images.shape == (batch_size, 3, 714, 1274)
        assert calibs.shape == (batch_size, 3, 4)
        assert batched_targets["labels"].shape == (batch_size, 5)
        assert batched_targets["obj_region"].shape == (batch_size, 714, 1274)
        assert batched_info["img_size"].shape == (batch_size, 2)
        assert len(batched_info["orig_ds"]) == batch_size
        assert torch.is_floating_point(images)


def test_custom_dataloader_factory_uses_v2_dataset(labeled_roots):
    from lib.helpers.dataloader_helper import build_dataloader

    cfg = make_cfg(labeled_roots["training"])
    cfg.dataset.type = "CUSTOM"
    cfg.dataset.root_dir_train = [labeled_roots["training"]]
    cfg.dataset.root_dir_test = [labeled_roots["testing"]]
    cfg.dataset.train_split = "train"
    cfg.dataset.test_split = "test"
    cfg.dataset.batch_size = 1

    class FabricPassthrough:
        @staticmethod
        def setup_dataloaders(loader):
            return loader

    train_loader, test_loader = build_dataloader(FabricPassthrough(), cfg, workers=0)
    assert isinstance(train_loader.dataset.datasets[0], CustomV2Dataset)
    assert isinstance(test_loader.dataset.datasets[0], CustomV2Dataset)
    batch = next(iter(test_loader))
    assert batch[0].shape == (1, 3, 714, 1274)


def test_flip_reflects_pixels_boxes_center_calibration_and_heading(labeled_roots):
    unflipped_dataset = CustomV2Dataset(
        "train", make_cfg(labeled_roots["training"], random_flip=0.0)
    )
    dataset = CustomV2Dataset(
        "train", make_cfg(labeled_roots["training"], random_flip=1.0)
    )
    image, p2, targets, info = dataset[0]
    assert info["flip"]
    width, height = map(int, info["img_size"])
    unflipped_image = unflipped_dataset[0][0]
    np.testing.assert_array_equal(image, unflipped_image[:, :, ::-1])

    transform = make_image_transform((1280, 720), (1274, 714))
    box = transform.transform_box([570.0, 320.0, 710.0, 460.0], clip_crop=True, flip=True)
    np.testing.assert_allclose(
        targets["boxes"][0],
        [(box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height),
         (box[2] - box[0]) / width, (box[3] - box[1]) / height],
        atol=1e-6,
    )

    source_calib = dataset.get_calib(1)
    source_center = np.asarray([[0.0, 0.8, 20.0]], dtype=np.float32)
    source_pixel, _ = source_calib.rect_to_img(source_center)
    expected_pixel = transform.transform_points(source_pixel)[0]
    expected_pixel[0] = width - expected_pixel[0]
    reflected_center = source_center.copy()
    reflected_center[0, 0] *= -1
    flipped_calib = Calibration(str(dataset._record(1).calibration_path))
    flipped_calib.apply_image_transform(transform.scale, transform.offset_x, transform.offset_y)
    dataset._reflect_calibration(flipped_calib, width)
    projected, _ = flipped_calib.rect_to_img(reflected_center)
    np.testing.assert_allclose(projected[0], expected_pixel, rtol=0, atol=1e-4)
    np.testing.assert_allclose(p2, flipped_calib.P2, atol=1e-6)
    np.testing.assert_allclose(
        targets["boxes_3d"][0, :2] * [width, height], expected_pixel, atol=1e-4
    )

    angle = class2angle(int(targets["heading_bin"][0, 0]), targets["heading_res"][0, 0])
    source_alpha = dataset.get_label(1)[0].alpha
    expected_alpha = (np.pi - source_alpha + np.pi) % (2 * np.pi) - np.pi
    assert abs(np.arctan2(np.sin(angle - expected_alpha), np.cos(angle - expected_alpha))) < 1e-6


@pytest.mark.parametrize("flip", [False, True])
def test_heading_target_uses_kitti_yaw_and_recomputed_alpha(labeled_roots, flip):
    (labeled_roots["training"] / "label_2" / "000001.txt").write_text(
        label_line(alpha=0.6, rotation_y=0.2)
    )
    dataset = CustomV2Dataset(
        "train",
        make_cfg(labeled_roots["training"], random_flip=float(flip)),
    )
    _, _, targets, info = dataset[0]
    assert bool(info["flip"]) is flip
    encoded_alpha = class2angle(
        int(targets["heading_bin"][0, 0]), targets["heading_res"][0, 0]
    )
    object_label = dataset.get_label(1)[0]
    expected_alpha = object_label.alpha
    if flip:
        expected_alpha = (np.pi - expected_alpha + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(object_label.ry, 0.2)
    error = np.arctan2(
        np.sin(encoded_alpha - expected_alpha), np.cos(encoded_alpha - expected_alpha)
    )
    assert abs(error) < 1e-6


def test_eval_ground_truth_preserves_kitti_yaw_and_recomputes_alpha(labeled_roots):
    dataset = CustomV2Dataset("test", make_cfg(labeled_roots["testing"]))
    annotation = {
        "name": np.asarray(["Car"]),
        "rotation_y": np.asarray([0.2], dtype=np.float32),
        "alpha": np.asarray([0.6], dtype=np.float32),
        # Evaluator dimensions are [length, height, width].
        "dimensions": np.asarray([[4.0, 1.6, 1.8]], dtype=np.float32),
        "location": np.asarray([[0.0, 1.6, 20.0]], dtype=np.float32),
    }
    dataset._recompute_eval_ground_truth_alpha([annotation], [1])

    np.testing.assert_allclose(annotation["rotation_y"], [0.2])
    # The geometric center projects at the optical center in this fixture.
    np.testing.assert_allclose(annotation["alpha"], [0.2], atol=1e-5)


def test_labeled_test_split_is_complete_and_never_augmented(labeled_roots):
    dataset = CustomV2Dataset(
        "test", make_cfg(labeled_roots["testing"], random_flip=1.0)
    )
    assert dataset.idx_list == ["000001", "000002"]
    _, _, target, first_info = dataset[0]
    assert target["mask_2d"].any()
    assert not first_info["flip"]
    second_image, _, second_target, second_info = dataset[1]
    assert not second_info["flip"]
    assert int(second_info["img_id"]) == 2
    assert not second_target["mask_2d"].any()
    # Test images are deterministic, even if the train flip probability is 1.
    again = dataset[1][0]
    np.testing.assert_array_equal(second_image, again)


@pytest.mark.parametrize(
    "option,value",
    [
        ("aug_pd", True),
        ("aug_crop", True),
        ("aug_calib", True),
        ("random_mixup3d", 0.1),
    ],
)
def test_unsupported_augmentations_are_rejected(labeled_roots, option, value):
    with pytest.raises(ValueError, match="supports horizontal flip only"):
        CustomV2Dataset("train", make_cfg(labeled_roots["training"], **{option: value}))


def test_class_map_and_mean_shape_validation(labeled_roots):
    with pytest.raises(ValueError, match="contiguous indices"):
        CustomV2Dataset(
            "train", make_cfg(labeled_roots["training"], cls2id={"Car": 1})
        )

    no_means = CustomV2Dataset(
        "train", make_cfg(labeled_roots["training"], use_meanshape=False)
    )
    targets = no_means[0][2]
    np.testing.assert_array_equal(targets["size_3d"][0], targets["src_size_3d"][0])
    np.testing.assert_array_equal(no_means.cls_mean_size, np.zeros((1, 3), np.float32))
    no_means.cfg.dataset.cls_mean_size = None
    no_means_without_configured_means = CustomV2Dataset("train", no_means.cfg)
    np.testing.assert_array_equal(
        no_means_without_configured_means.cls_mean_size, np.zeros((1, 3), np.float32)
    )


def test_overflow_is_capped_and_low_quality_is_masked(labeled_roots):
    label_path = labeled_roots["training"] / "label_2" / "000001.txt"
    label_path.write_text(
        label_line(truncation=0.8) + "".join(label_line() for _ in range(7))
    )
    dataset = CustomV2Dataset(
        "train", make_cfg(labeled_roots["training"], max_objs=3)
    )
    _, _, targets, _ = dataset[0]
    assert targets["labels"].shape == (3,)
    assert targets["mask_2d"].tolist() == [False, True, True]
    assert targets["depth"].shape == (3, 1)


def test_filtered_objects_leave_a_valid_empty_training_sample(labeled_roots):
    (labeled_roots["training"] / "label_2" / "000001.txt").write_text(
        label_line(z=150.0) + label_line(x=100.0)
    )
    dataset = CustomV2Dataset("train", make_cfg(labeled_roots["training"]))
    assert len(dataset) == 2
    _, _, targets, info = dataset[0]
    assert not targets["mask_2d"].any()
    assert not targets["obj_region"].any()
    assert info["stat_gt_cars"] == 2
    assert info["stat_removed_by_depth"] == 1
    assert info["stat_rejected_proj"] == 1


def test_render_helper_draws_transformed_2d_and_3d_without_gui(labeled_roots):
    dataset = CustomV2Dataset("test", make_cfg(labeled_roots["testing"]))
    image, _, _, info = dataset[0]
    rendered = render_debug_sample(
        image,
        dataset.get_label(int(info["img_id"])),
        dataset.get_calib(int(info["img_id"])),
        info,
        class_names=dataset.class_name,
    )
    assert rendered.shape == (714, 1274, 3)
    assert rendered.dtype == np.uint8
    assert np.any(rendered != np.asarray(image).transpose(1, 2, 0).clip(0, 255))


def test_split_file_overrides_fallback_index(labeled_roots):
    image_sets = labeled_roots["testing"] / "ImageSets"
    image_sets.mkdir()
    (image_sets / "test.txt").write_text("000002\n")
    dataset = CustomV2Dataset("test", make_cfg(labeled_roots["testing"]))
    assert len(dataset) == 1
    assert dataset.idx_list == ["000002"]


def test_fallback_matches_unpadded_physical_file_stems(labeled_roots):
    root = labeled_roots["training"]
    for directory, extension in (("image_2", ".png"), ("calib", ".txt"), ("label_2", ".txt")):
        for sample_id in (1, 2):
            old_path = root / directory / f"{sample_id:06d}{extension}"
            old_path.rename(root / directory / f"{sample_id}{extension}")
    dataset = CustomV2Dataset("train", make_cfg(root))
    assert dataset.idx_list == ["000001", "000002"]
    assert dataset._record(1).image_path.name == "1.png"
    assert dataset.get_label(1)


@pytest.mark.parametrize(
    "root_name,split,expected_count",
    [("training", "train", 1658), ("testing", "test", 519)],
)
def test_current_root_indexes_only_matching_labeled_samples(root_name, split, expected_count):
    root = DATA_ROOT / root_name
    if not root.is_dir():
        pytest.skip(f"custom dataset is unavailable at {root}")
    from lib.helpers.config_helper import Config

    cfg = Config.load_from_yaml(str(CONFIG_PATH))
    configured_roots = cfg.dataset.root_dir_train if split == "train" else cfg.dataset.root_dir_test
    dataset = CustomV2Dataset(split, cfg, root_dir=configured_roots[0])
    assert len(dataset) == expected_count
    assert len(dataset.idx_list) == len(set(dataset.idx_list))
    assert all(dataset._record(int(stem)).calibration_path.is_file() for stem in dataset.idx_list)
    assert all(dataset._record(int(stem)).label_path.is_file() for stem in dataset.idx_list)
    image, calibration, targets, info = dataset[0]
    assert image.shape == (3, int(dataset.input_size[1]), int(dataset.input_size[0]))
    assert calibration.shape == (3, 4)
    assert set(targets) == TARGET_KEYS
    assert int(info["img_id"]) == int(dataset.idx_list[0])
