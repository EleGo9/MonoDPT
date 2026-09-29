"""Image-space geometry shared by the custom dataset and its tests."""

from dataclasses import dataclass

import numpy as np


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
    offset_x: int
    offset_y: int
    crop_x0: int = 0
    crop_y0: int = 0

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


def make_image_transform(source_size, target_size, *, allow_crop=True):
    """Choose a centered crop when it fits, otherwise the legacy letterbox.

    ``source_size`` and ``target_size`` are ``(width, height)`` pairs.
    Cropping is enabled by the caller only when the configured target is
    smaller than the configured source size. Equal-size inputs intentionally
    use the letterbox path with identity geometry.
    """
    source_width, source_height = map(int, source_size)
    target_width, target_height = map(int, target_size)

    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Image dimensions must be positive")

    fits = target_width <= source_width and target_height <= source_height
    is_smaller = target_width < source_width or target_height < source_height
    if allow_crop and fits and is_smaller:
        crop_x0 = (source_width - target_width) // 2
        crop_y0 = (source_height - target_height) // 2
        return ImageTransform(
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
            mode="crop",
            scale=1.0,
            offset_x=-crop_x0,
            offset_y=-crop_y0,
            crop_x0=crop_x0,
            crop_y0=crop_y0,
        )

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
    """Apply the crop selected by ``make_image_transform`` to a PIL image."""
    if not transform.is_crop:
        return image
    x0, y0 = transform.crop_x0, transform.crop_y0
    return image.crop(
        (x0, y0, x0 + transform.target_width, y0 + transform.target_height)
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
