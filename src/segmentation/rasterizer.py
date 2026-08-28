from dataclasses import dataclass

import cv2
import numpy as np

from src.preprocessing.transforms import (
    TransformInfo,
    transform_points,
)


# ============================================================
# Segmentation Rasterization 설정
# ============================================================

@dataclass(frozen=True)
class RasterConfig:
    """
    Segmentation GT rasterization 설정.

    현재 768×448 baseline에서의 기본값이며,
    이후 실험 시 config 값만 변경한다.
    """

    lane_thickness: int = 3
    stop_thickness: int = 3


DEFAULT_RASTER_CONFIG = RasterConfig()


# ============================================================
# Rasterization 통계 초기화
# ============================================================

def make_raster_stats() -> dict[str, int]:
    """
    한 이미지 rasterization 과정에서 발생한
    annotation 처리 통계를 저장한다.
    """

    return {
        "traffic_lane_drawn": 0,
        "stop_line_drawn": 0,
        "crosswalk_drawn": 0,

        "empty_data_skipped": 0,
        "invalid_polyline_skipped": 0,
        "invalid_polygon_skipped": 0,

        "source_points_clipped": 0,
    }


# ============================================================
# 원본 좌표 검증 + source image boundary clip
# ============================================================

def prepare_source_points(
    data: list[dict],
    original_width: int,
    original_height: int,
    stats: dict[str, int],
) -> np.ndarray | None:
    """
    JSONL의 annotation point를 numpy array로 변환하고
    원본 이미지 boundary 밖 좌표를 clip한다.

    원본 JSONL 자체는 수정하지 않는다.
    """

    if not data:
        stats["empty_data_skipped"] += 1
        return None


    try:
        points = np.asarray(
            [
                [point["x"], point["y"]]
                for point in data
            ],
            dtype=np.float32,
        )

    except (KeyError, TypeError, ValueError):
        stats["empty_data_skipped"] += 1
        return None


    if (
        points.ndim != 2
        or points.shape[1] != 2
    ):
        stats["empty_data_skipped"] += 1
        return None


    # --------------------------------------------------------
    # clip 전 좌표 보존
    # --------------------------------------------------------

    before_clip = points.copy()


    # --------------------------------------------------------
    # 원본 이미지 boundary 안으로 제한
    # --------------------------------------------------------

    points[:, 0] = np.clip(
        points[:, 0],
        0,
        original_width - 1,
    )

    points[:, 1] = np.clip(
        points[:, 1],
        0,
        original_height - 1,
    )


    # 실제로 clip된 point 개수
    clipped = np.any(
        before_clip != points,
        axis=1,
    )

    stats["source_points_clipped"] += int(
        clipped.sum()
    )


    return points


# ============================================================
# float 좌표 → OpenCV raster 좌표
# ============================================================

def to_raster_points(
    source_points: np.ndarray,
    transform_info: TransformInfo,
) -> np.ndarray:
    """
    공통 transform을 적용한 float 좌표를
    OpenCV가 사용할 integer pixel 좌표로 변환한다.

    주의:
    transform_points() 자체는 Vector Task와 공유하기 위해
    float32 좌표를 유지한다.

    정수화는 Segmentation rasterization 단계에서만 수행한다.
    """

    transformed = transform_points(
        points=source_points,
        transform_info=transform_info,
        clip=True,
    )


    raster_points = np.rint(
        transformed
    ).astype(np.int32)


    # 반올림 후에도 안전하게 한 번 더 boundary clip
    raster_points[:, 0] = np.clip(
        raster_points[:, 0],
        0,
        transform_info.padded_width - 1,
    )

    raster_points[:, 1] = np.clip(
        raster_points[:, 1],
        0,
        transform_info.padded_height - 1,
    )


    return raster_points


# ============================================================
# 최종 Segmentation GT Rasterizer
# ============================================================

def rasterize_segmentation_targets(
    annotations: list[dict],
    original_width: int,
    original_height: int,
    transform_info: TransformInfo,
    raster_config: RasterConfig = DEFAULT_RASTER_CONFIG,
) -> dict:
    """
    한 이미지의 annotation 목록을 받아
    3개의 독립 binary segmentation mask를 생성한다.

    Returns
    -------
    {
        "lane_mask": np.ndarray,
        "stop_mask": np.ndarray,
        "crosswalk_mask": np.ndarray,
        "stats": dict,
    }
    """


    # --------------------------------------------------------
    # Config 검증
    # --------------------------------------------------------

    if raster_config.lane_thickness <= 0:
        raise ValueError(
            "lane_thickness는 0보다 커야 합니다."
        )

    if raster_config.stop_thickness <= 0:
        raise ValueError(
            "stop_thickness는 0보다 커야 합니다."
        )


    target_width = (
        transform_info.padded_width
    )

    target_height = (
        transform_info.padded_height
    )


    # ========================================================
    # 독립적인 3개 binary target
    # ========================================================

    lane_mask = np.zeros(
        (target_height, target_width),
        dtype=np.uint8,
    )

    stop_mask = np.zeros(
        (target_height, target_width),
        dtype=np.uint8,
    )

    crosswalk_mask = np.zeros(
        (target_height, target_width),
        dtype=np.uint8,
    )


    stats = make_raster_stats()


    # ========================================================
    # Annotation loop
    # ========================================================

    for ann in annotations:

        cls = ann.get("class")
        category = ann.get("category")
        data = ann.get("data", [])


        # ----------------------------------------------------
        # 원본 좌표 준비
        # ----------------------------------------------------

        source_points = prepare_source_points(
            data=data,
            original_width=original_width,
            original_height=original_height,
            stats=stats,
        )


        if source_points is None:
            continue


        # ====================================================
        # traffic_lane
        #
        # solid / dotted를 Seg GT에서 구분하지 않는다.
        # 현재 목적은 lane-flow / spatial hint이다.
        # ====================================================

        if (
            cls == "traffic_lane"
            and category == "polyline"
        ):

            if len(source_points) < 2:
                stats[
                    "invalid_polyline_skipped"
                ] += 1

                continue


            points = to_raster_points(
                source_points=source_points,
                transform_info=transform_info,
            )


            cv2.polylines(
                lane_mask,
                [
                    points.reshape(
                        -1,
                        1,
                        2,
                    )
                ],
                isClosed=False,
                color=255,
                thickness=raster_config.lane_thickness,
                lineType=cv2.LINE_8,
            )


            stats[
                "traffic_lane_drawn"
            ] += 1


        # ====================================================
        # stop_line
        # ====================================================

        elif (
            cls == "stop_line"
            and category == "polyline"
        ):

            if len(source_points) < 2:
                stats[
                    "invalid_polyline_skipped"
                ] += 1

                continue


            points = to_raster_points(
                source_points=source_points,
                transform_info=transform_info,
            )


            cv2.polylines(
                stop_mask,
                [
                    points.reshape(
                        -1,
                        1,
                        2,
                    )
                ],
                isClosed=False,
                color=255,
                thickness=raster_config.stop_thickness,
                lineType=cv2.LINE_8,
            )


            stats[
                "stop_line_drawn"
            ] += 1


        # ====================================================
        # crosswalk
        # ====================================================

        elif (
            cls == "crosswalk"
            and category == "polygon"
        ):

            if len(source_points) < 3:
                stats[
                    "invalid_polygon_skipped"
                ] += 1

                continue


            points = to_raster_points(
                source_points=source_points,
                transform_info=transform_info,
            )


            cv2.fillPoly(
                crosswalk_mask,
                [
                    points.reshape(
                        -1,
                        1,
                        2,
                    )
                ],
                color=255,
            )


            stats[
                "crosswalk_drawn"
            ] += 1


    # ========================================================
    # 결과 반환
    # ========================================================

    return {
        "lane_mask": lane_mask,
        "stop_mask": stop_mask,
        "crosswalk_mask": crosswalk_mask,
        "stats": stats,
    }