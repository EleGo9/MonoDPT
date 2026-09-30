import numpy as np
import pytest
from PIL import Image, ImageDraw

from lib.datasets.custom.image_geometry import (
    crop_image,
    make_image_transform,
    model_input_size,
)


def test_scaled_crop_geometry_and_legacy_integer_crop():
    half = make_image_transform(
        (1280, 720), (1260, 700), scale_factor=0.5
    )
    assert (half.target_width, half.target_height) == (630, 350)
    assert half.crop_x0 == pytest.approx(10)
    assert half.crop_width == pytest.approx(1260)
    assert half.offset_x == pytest.approx(-5.0)

    three_quarters = make_image_transform(
        (1280, 720), (1232, 672), scale_factor=0.75
    )
    assert (three_quarters.target_width, three_quarters.target_height) == (924, 504)

    legacy = make_image_transform((1280, 720), (1274, 714), scale_factor=1.0)
    assert (legacy.target_width, legacy.target_height) == (1274, 714)
    assert legacy.scale == 1.0
    assert (legacy.offset_x, legacy.offset_y) == (-3.0, -3.0)
    image = Image.fromarray(np.arange(1280 * 720, dtype=np.uint8).reshape(720, 1280))
    np.testing.assert_array_equal(
        np.asarray(crop_image(image, legacy)),
        np.asarray(image.crop((3, 3, 1277, 717))),
    )


def test_non_multiple_crop_uses_smaller_centered_source_window():
    transform = make_image_transform(
        (1280, 720), (1274, 714), scale_factor=0.5
    )
    assert (transform.target_width, transform.target_height) == (630, 350)
    assert transform.crop_width == pytest.approx(1260)
    assert transform.crop_height == pytest.approx(700)
    assert transform.crop_width <= 1274
    assert transform.crop_height <= 714
    assert transform.crop_x0 >= 0
    assert transform.crop_y0 >= 0


@pytest.mark.parametrize("scale", [1.0, 0.75, 0.5])
def test_point_transform_round_trip(scale):
    transform = make_image_transform(
        (1280, 720), (1274, 714), scale_factor=scale
    )
    rng = np.random.default_rng(51)
    points = rng.uniform([-100, -100], [1400, 820], size=(1000, 2)).astype(np.float32)
    np.testing.assert_allclose(
        transform.inverse_points(transform.transform_points(points)),
        points,
        rtol=0,
        atol=1e-3,
    )


def test_box_clipping_and_flip_involution():
    transform = make_image_transform(
        (1280, 720), (1260, 700), scale_factor=0.5
    )
    box = transform.transform_box([-100, 100, 50, 400], clip_crop=True)
    assert box is not None
    assert 0 <= box[0] <= box[2] <= transform.target_width
    assert 0 <= box[1] <= box[3] <= transform.target_height
    assert transform.transform_box([-500, 100, -300, 400], clip_crop=True) is None

    unflipped = transform.transform_box([100, 100, 500, 400], clip_crop=True)
    flipped = transform.transform_box([100, 100, 500, 400], clip_crop=True, flip=True)
    np.testing.assert_allclose(
        flipped,
        [transform.target_width - unflipped[2], unflipped[1],
         transform.target_width - unflipped[0], unflipped[3]],
    )
    flipped_twice = [
        transform.target_width - flipped[2],
        flipped[1],
        transform.target_width - flipped[0],
        flipped[3],
    ]
    np.testing.assert_allclose(flipped_twice, unflipped)


@pytest.mark.parametrize(
    "scale,crop_size",
    [(1.0, (1274, 714)), (0.75, (1232, 672)), (0.5, (1260, 700))],
)
def test_output_grid_and_window_scale(scale, crop_size):
    out_w, out_h = model_input_size(crop_size, scale)
    transform = make_image_transform((1280, 720), crop_size, scale_factor=scale)
    assert out_w % 14 == 0
    assert out_h % 14 == 0
    assert transform.crop_width * scale == pytest.approx(out_w, abs=1e-9)
    assert transform.crop_height * scale == pytest.approx(out_h, abs=1e-9)


@pytest.mark.parametrize(
    "scale,crop_size",
    [(0.5, (1260, 700)), (0.75, (1232, 672))],
)
def test_scaled_pixel_marker_tracks_point_transform(scale, crop_size):
    image = Image.new("L", (1280, 720), 0)
    ImageDraw.Draw(image).rectangle((398, 298, 402, 302), fill=255)
    transform = make_image_transform(
        image.size, crop_size, scale_factor=scale
    )
    pixels = np.asarray(crop_image(image, transform), dtype=np.float64)
    yy, xx = np.indices(pixels.shape)
    centroid = np.array(
        [(pixels * xx).sum() / pixels.sum(), (pixels * yy).sum() / pixels.sum()]
    )
    expected = transform.transform_points([[400.0, 300.0]])[0]
    assert np.max(np.abs(centroid - expected)) < 0.6


def test_subpixel_crop_geometry_at_scale_055():
    transform = make_image_transform(
        (1274, 714), (1274, 714), scale_factor=0.55
    )
    assert (transform.target_width, transform.target_height) == (700, 392)
    assert transform.crop_x0 != pytest.approx(round(transform.crop_x0))

    source_points = np.array(
        [[100.0, 100.0], [640.0, 360.0], [1100.0, 600.0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        transform.inverse_points(transform.transform_points(source_points)),
        source_points,
        rtol=0,
        atol=1e-3,
    )

    box = transform.transform_box([-20.0, -10.0, 1300.0, 730.0], clip_crop=True)
    assert box is not None
    assert 0 <= box[0] <= box[2] <= transform.target_width
    assert 0 <= box[1] <= box[3] <= transform.target_height

    for x, y in source_points:
        image = Image.new("L", (1274, 714), 0)
        ImageDraw.Draw(image).rectangle(
            (int(x) - 2, int(y) - 2, int(x) + 2, int(y) + 2), fill=255
        )
        pixels = np.asarray(crop_image(image, transform), dtype=np.float64)
        yy, xx = np.indices(pixels.shape)
        centroid = np.array(
            [(pixels * xx).sum() / pixels.sum(), (pixels * yy).sum() / pixels.sum()]
        )
        expected = transform.transform_points([[x, y]])[0]
        assert np.max(np.abs(centroid - expected)) < 0.6


def test_invalid_scale_crop_combinations_raise():
    with pytest.raises(ValueError):
        make_image_transform((100, 100), (1260, 700), scale_factor=0.5)
    with pytest.raises(ValueError):
        make_image_transform((1280, 720), (1260, 700), scale_factor=0)
    with pytest.raises(ValueError):
        make_image_transform(
            (1280, 720), (1260, 700), scale_factor=0.5, allow_crop=False
        )
