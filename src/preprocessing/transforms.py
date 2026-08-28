from dataclasses import dataclass
import math

import cv2
import numpy as np


# ============================================================
# Model A 공통 Image Transform 설정
# ============================================================

@dataclass(frozen=True)
class TransformConfig:
    """
    Model A에서 RGB / Segmentation / Vector가 공통으로 사용하는
    이미지 좌표계 변환 설정.
    """

    resize_scale: float = 0.4
    model_stride: int = 32

    # RGB padding 값 (BGR 기준)
    pad_value: tuple[int, int, int] = (0, 0, 0)


DEFAULT_TRANSFORM_CONFIG = TransformConfig()


# ============================================================
# 계산된 Transform 정보
# ============================================================

@dataclass(frozen=True)
class TransformInfo:
    scale_x: float
    scale_y: float

    original_width: int
    original_height: int

    resized_width: int
    resized_height: int

    padded_width: int
    padded_height: int

    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int


# ============================================================
# stride 배수로 올림
# ============================================================

def ceil_to_multiple(
    value: int,
    multiple: int,
) -> int:

    if multiple <= 0:
        raise ValueError(
            "multiple은 0보다 커야 합니다."
        )

    return int(
        math.ceil(value / multiple)
        * multiple
    )


# ============================================================
# Transform 정보 계산
# ============================================================

def compute_transform_info(
    original_width: int,
    original_height: int,
    config: TransformConfig = DEFAULT_TRANSFORM_CONFIG,
) -> TransformInfo:

    if original_width <= 0 or original_height <= 0:
        raise ValueError(
            "original_width와 original_height는 "
            "0보다 커야 합니다."
        )

    if config.resize_scale <= 0:
        raise ValueError(
            "resize_scale은 0보다 커야 합니다."
        )

    if config.model_stride <= 0:
        raise ValueError(
            "model_stride는 0보다 커야 합니다."
        )


    # --------------------------------------------------------
    # Resize 크기
    # --------------------------------------------------------

    resized_width = int(
        round(
            original_width
            * config.resize_scale
        )
    )

    resized_height = int(
        round(
            original_height
            * config.resize_scale
        )
    )


    # --------------------------------------------------------
    # stride 배수에 맞게 padding target 계산
    # --------------------------------------------------------

    padded_width = ceil_to_multiple(
        resized_width,
        config.model_stride,
    )

    padded_height = ceil_to_multiple(
        resized_height,
        config.model_stride,
    )


    # --------------------------------------------------------
    # 좌우 / 상하 대칭 padding
    # --------------------------------------------------------

    pad_width = (
        padded_width
        - resized_width
    )

    pad_height = (
        padded_height
        - resized_height
    )


    pad_left = (
        pad_width // 2
    )

    pad_right = (
        pad_width
        - pad_left
    )


    pad_top = (
        pad_height // 2
    )

    pad_bottom = (
        pad_height
        - pad_top
    )


    # 실제 resize 결과 기준 scale.
    # 이후 다른 원본 크기가 들어와도 이 값을 사용한다.
    scale_x = (
        resized_width
        / original_width
    )

    scale_y = (
        resized_height
        / original_height
    )


    return TransformInfo(
        scale_x=scale_x,
        scale_y=scale_y,

        original_width=original_width,
        original_height=original_height,

        resized_width=resized_width,
        resized_height=resized_height,

        padded_width=padded_width,
        padded_height=padded_height,

        pad_left=pad_left,
        pad_right=pad_right,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
    )


# ============================================================
# Point 좌표 변환
# ============================================================

def transform_points(
    points: np.ndarray,
    transform_info: TransformInfo,
    clip: bool = True,
) -> np.ndarray:
    """
    원본 이미지 좌표의 point를
    resize + padding 좌표계로 변환한다.

    중요:
    Vector Task에서도 사용할 수 있도록
    float32 좌표를 그대로 반환한다.

    Segmentation rasterization에서만
    필요 시 마지막에 integer pixel 좌표로 변환한다.
    """

    points = np.asarray(
        points,
        dtype=np.float32,
    )


    if (
        points.ndim != 2
        or points.shape[1] != 2
    ):
        raise ValueError(
            "points shape은 (N, 2)여야 합니다."
        )


    transformed = points.copy()


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    transformed[:, 0] *= (
        transform_info.scale_x
    )

    transformed[:, 1] *= (
        transform_info.scale_y
    )


    # --------------------------------------------------------
    # Padding offset
    # --------------------------------------------------------

    transformed[:, 0] += (
        transform_info.pad_left
    )

    transformed[:, 1] += (
        transform_info.pad_top
    )


    # --------------------------------------------------------
    # 필요 시 target boundary 안으로 제한
    # --------------------------------------------------------

    if clip:

        transformed[:, 0] = np.clip(
            transformed[:, 0],
            0,
            transform_info.padded_width - 1,
        )

        transformed[:, 1] = np.clip(
            transformed[:, 1],
            0,
            transform_info.padded_height - 1,
        )


    return transformed


# ============================================================
# RGB Image Resize + Padding
# ============================================================

def transform_image(
    image_bgr: np.ndarray,
    transform_info: TransformInfo,
    config: TransformConfig = DEFAULT_TRANSFORM_CONFIG,
) -> np.ndarray:
    """
    BGR 이미지를 TransformInfo와 동일한 좌표계로 변환한다.
    """

    if image_bgr is None:
        raise ValueError(
            "image_bgr가 None입니다."
        )


    resized = cv2.resize(
        image_bgr,
        (
            transform_info.resized_width,
            transform_info.resized_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )


    padded = cv2.copyMakeBorder(
        resized,

        transform_info.pad_top,
        transform_info.pad_bottom,
        transform_info.pad_left,
        transform_info.pad_right,

        borderType=cv2.BORDER_CONSTANT,
        value=config.pad_value,
    )


    expected_shape = (
        transform_info.padded_height,
        transform_info.padded_width,
    )


    if padded.shape[:2] != expected_shape:

        raise RuntimeError(
            "Transform 결과 shape이 예상과 다릅니다. "
            f"actual={padded.shape[:2]}, "
            f"expected={expected_shape}"
        )


    return padded