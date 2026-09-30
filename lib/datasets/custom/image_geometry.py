"""Image-space geometry shared by the custom dataset and its tests."""

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageTransform:
    """Map source image coordinates into the model input image frame.

    Sizes and offsets use PIL's ``(width, height)`` order.  The coordinate
    convention is ``u' = scale * u + offset_x`` and
    ``v' = scale * v + offset_y``.  A crop therefore has a negative offset.
    """

    source_width: int
    source_height: int
    target_width: int
    target_height: int
    mode: str
    scale: float
    offset_x: float
    offset_y: float
    crop_x0: float = 0.0
    crop_y0: float = 0.0
    crop_width: float = 0.0
    crop_height: float = 0.0

    @property
    def is_crop(self):
        return self.mode == "crop"

    def transform_points(self, points):
        points = np.asarray(points, dtype=np.float32).copy()
        points[..., 0] = points[..., 0] * self.scale + self.offset_x
        points[..., 1] = points[..., 1] * self.scale + self.offset_y
        return points

    def inverse_points(self, points):
        points = np.asarray(points, dtype=np.float32).copy()
        points[..., 0] = (points[..., 0] - self.offset_x) / self.scale
        points[..., 1] = (points[..., 1] - self.offset_y) / self.scale
        return points

    def transform_box(self, box, *, clip_crop=False, flip=False):
        """Transform an xyxy box, optionally clipping it to a crop.

        Returns ``None`` when the box has no visible area after crop clipping.
        For uncropped/letterboxed images, clipping remains disabled to preserve
        the custom loader's existing annotation behavior.
        """
        box = np.asarray(box, dtype=np.float32).copy()
        box[[0, 2]] = box[[0, 2]] * self.scale + self.offset_x
        box[[1, 3]] = box[[1, 3]] * self.scale + self.offset_y

        if clip_crop and self.is_crop:
            box[[0, 2]] = np.clip(box[[0, 2]], 0, self.target_width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, self.target_height)
            if box[2] <= box[0] or box[3] <= box[1]:
                return None

        if flip:
            x1, x2 = box[0], box[2]
            box[0], box[2] = self.target_width - x2, self.target_width - x1
        return box


def model_input_size(crop_size, scale_factor, patch_size=14):
    """Return the scaled crop size rounded down to the model patch grid."""
    scale_factor = float(scale_factor)
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be positive and finite")
    return tuple(
        max(patch_size, int(np.floor(int(c) * scale_factor / patch_size + 1e-9)) * patch_size)
        for c in crop_size
    )


def make_image_transform(
    source_size, crop_size, *, scale_factor=1.0, allow_crop=True, patch_size=14
):
    """Choose a centered crop when it fits, otherwise the legacy letterbox.

    source_size and crop_size are (width, height) pairs. The configured crop
    window is scaled only on the crop path.
    """
    source_width, source_height = map(int, source_size)
    crop_width, crop_height = map(int, crop_size)
    scale_factor = float(scale_factor)

    if min(source_width, source_height, crop_width, crop_height) <= 0:
        raise ValueError("Image dimensions and crop size must be positive")
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be positive and finite")

    fits = crop_width <= source_width and crop_height <= source_height
    is_smaller = crop_width < source_width or crop_height < source_height
    if scale_factor != 1.0 and not (allow_crop and fits):
        raise ValueError("scale_factor != 1 needs the crop path and resolution <= source size")

    if allow_crop and fits and (is_smaller or scale_factor != 1.0):
        target_width, target_height = model_input_size(
            (crop_width, crop_height), scale_factor, patch_size
        )
        window_width, window_height = target_width / scale_factor, target_height / scale_factor
        if window_width > crop_width + 1e-9 or window_height > crop_height + 1e-9:
            raise ValueError("scale_factor is too small for a patch-sized input within the crop window")
        if scale_factor == 1.0:
            crop_x0 = (source_width - target_width) // 2
            crop_y0 = (source_height - target_height) // 2
        else:
            crop_x0 = (source_width - window_width) / 2
            crop_y0 = (source_height - window_height) / 2
        return ImageTransform(
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
            mode="crop",
            scale=scale_factor,
            offset_x=-scale_factor * crop_x0,
            offset_y=-scale_factor * crop_y0,
            crop_x0=crop_x0,
            crop_y0=crop_y0,
            crop_width=window_width,
            crop_height=window_height,
        )

    target_width, target_height = crop_width, crop_height
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = int(source_width * scale)
    resized_height = int(source_height * scale)
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    return ImageTransform(
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        mode="letterbox",
        scale=scale,
        offset_x=pad_x,
        offset_y=pad_y,
    )


def crop_image(image, transform):
    """Apply the crop selected by make_image_transform to a PIL image."""
    if not transform.is_crop:
        return image
    box = (
        transform.crop_x0,
        transform.crop_y0,
        min(transform.crop_x0 + transform.crop_width, transform.source_width),
        min(transform.crop_y0 + transform.crop_height, transform.source_height),
    )
    if transform.scale == 1.0:
        return image.crop(tuple(int(value) for value in box))
    return image.resize(
        (transform.target_width, transform.target_height),
        Image.BILINEAR,
        box=box,
    )


def validate_model_resolution(resolution, patch_size=14):
    """Fail early when a configured model input violates DINOv2's patch grid."""
    width, height = map(int, resolution)
    if width <= 0 or height <= 0:
        raise ValueError(f"Model resolution must be positive, got {(width, height)}")
    if width % patch_size or height % patch_size:
        raise ValueError(
            f"Model resolution {(width, height)} must be divisible by the "
            f"DINOv2 patch size {patch_size}; the dataset/model path does not "
            "round or pad dimensions."
        )
