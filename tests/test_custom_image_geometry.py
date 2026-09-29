import sys
import types

import numpy as np
import pytest
from PIL import Image

# These tests exercise NumPy-only Calibration projection methods. OpenCV is
# needed by unrelated affine helpers in the same module, so stub it only when
# the project dependency is absent from the test environment.
try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = types.ModuleType("cv2")

from lib.datasets.custom.image_geometry import crop_image, make_image_transform, validate_model_resolution
from lib.datasets.kitti.kitti_utils import Calibration


def make_calibration():
    P2 = np.array(
        [
            [800.0, 0.0, 640.0, -32.0],
            [0.0, 700.0, 360.0, 21.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return Calibration(
        {
            "P2": P2,
            "R0": np.eye(3, dtype=np.float32),
            "Tr_velo2cam": np.array(
                [[1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            "D": np.zeros(5, dtype=np.float32),
        }
    )


def test_equal_resolution_is_identity_and_preserves_calibration():
    transform = make_image_transform((1280, 720), (1280, 720))
    assert transform.mode == "letterbox"
    assert transform.scale == 1.0
    assert (transform.offset_x, transform.offset_y) == (0, 0)
    assert not transform.is_crop
    np.testing.assert_array_equal(
        transform.transform_points([[24.0, 70.0]]), [[24.0, 70.0]]
    )

    calib = make_calibration()
    original_P2 = calib.P2.copy()
    calib.apply_image_transform(transform.scale, transform.offset_x, transform.offset_y)
    np.testing.assert_array_equal(calib.P2, original_P2)


def test_equal_configured_resolutions_do_not_crop_a_larger_source():
    transform = make_image_transform((1920, 1080), (1280, 720), allow_crop=False)
    assert transform.mode == "letterbox"
    assert transform.scale == pytest.approx(2 / 3)
    assert not transform.is_crop


def test_center_crop_dimensions_offset_and_source_pixels():
    transform = make_image_transform((1280, 720), (1274, 392))
    assert transform.is_crop
    assert (transform.crop_x0, transform.crop_y0) == (3, 164)
    assert (transform.target_width, transform.target_height) == (1274, 392)

    pixels = np.zeros((720, 1280, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(1280, dtype=np.uint16)[None, :] % 256
    pixels[..., 1] = np.arange(720, dtype=np.uint16)[:, None] % 256
    cropped = crop_image(Image.fromarray(pixels), transform)
    assert cropped.size == (1274, 392)
    np.testing.assert_array_equal(np.asarray(cropped)[0, 0], pixels[164, 3])


def test_crop_box_shift_clipping_and_empty_boundary_box():
    transform = make_image_transform((1280, 720), (1274, 392))
    clipped = transform.transform_box([1, 150, 20, 190], clip_crop=True)
    np.testing.assert_array_equal(clipped, [0, 0, 17, 26])
    assert transform.transform_box([-100, 170, -20, 200], clip_crop=True) is None

    flipped = transform.transform_box([1, 150, 20, 190], clip_crop=True, flip=True)
    np.testing.assert_array_equal(flipped, [1257, 0, 1274, 26])


def test_crop_updates_principal_point_and_keeps_3d_projection_consistent():
    calib = make_calibration()
    source_calib = make_calibration()
    transform = make_image_transform((1280, 720), (1274, 392))
    points_rect = np.array(
        [[0.5, 0.1, 8.0], [-1.2, -0.4, 14.0], [2.1, 0.7, 32.0]],
        dtype=np.float32,
    )
    source_pixels, _ = source_calib.rect_to_img(points_rect)

    calib.apply_image_transform(transform.scale, transform.offset_x, transform.offset_y)
    transformed_pixels, _ = calib.rect_to_img(points_rect)
    np.testing.assert_allclose(
        transformed_pixels,
        source_pixels - np.array([3.0, 164.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-5,
    )
    assert calib.fu == source_calib.fu
    assert calib.fv == source_calib.fv
    assert calib.cu == pytest.approx(source_calib.cu - 3)
    assert calib.cv == pytest.approx(source_calib.cv - 164)

    recovered = calib.img_to_rect(
        transformed_pixels[:, 0], transformed_pixels[:, 1], points_rect[:, 2]
    )
    np.testing.assert_allclose(recovered, points_rect, rtol=1e-6, atol=1e-5)

    source_alpha = source_calib.ry2alpha(0.3, source_pixels[0, 0])
    crop_alpha = calib.ry2alpha(0.3, transformed_pixels[0, 0])
    assert crop_alpha == pytest.approx(source_alpha)


def test_letterbox_resize_projection_and_unprojection_remain_consistent():
    transform = make_image_transform((1280, 720), (640, 728))
    assert transform.mode == "letterbox"
    assert transform.scale == pytest.approx(0.5)
    assert transform.offset_x == 0
    assert transform.offset_y == 184

    source_calib = make_calibration()
    transformed_calib = make_calibration()
    point = np.array([[0.5, -0.2, 12.0]], dtype=np.float32)
    source_pixels, _ = source_calib.rect_to_img(point)
    transformed_calib.apply_image_transform(
        transform.scale, transform.offset_x, transform.offset_y
    )
    transformed_pixels, _ = transformed_calib.rect_to_img(point)
    np.testing.assert_allclose(
        transformed_pixels,
        transform.transform_points(source_pixels),
        rtol=1e-6,
        atol=1e-5,
    )
    recovered = transformed_calib.img_to_rect(
        transformed_pixels[:, 0], transformed_pixels[:, 1], point[:, 2]
    )
    np.testing.assert_allclose(recovered, point, rtol=1e-6, atol=1e-5)


def test_dinov2_multiple_of_14_contract_is_preserved():
    validate_model_resolution((1274, 392))
    with pytest.raises(ValueError, match="divisible by the DINOv2 patch size 14"):
        validate_model_resolution((1275, 392))


def test_resolution_larger_than_source_keeps_letterbox_geometry():
    transform = make_image_transform((1280, 720), (1280, 728))
    assert transform.mode == "letterbox"
    assert transform.scale == 1.0
    assert (transform.offset_x, transform.offset_y) == (0, 4)
