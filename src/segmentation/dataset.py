import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.transforms import (
    TransformConfig,
    compute_transform_info,
    transform_image,
)
from src.segmentation.rasterizer import (
    RasterConfig,
    rasterize_segmentation_targets,
)


class SegmentationDataset(Dataset):

    def __init__(
        self,
        annotation_path,
        image_root,
        file_names=None,
        transform_config=None,
        raster_config=None,
    ):
        super().__init__()

        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root)

        if transform_config is None:
            transform_config = TransformConfig(
                resize_scale=0.4,
                model_stride=32,
                pad_value=(0, 0, 0),
            )

        if raster_config is None:
            raster_config = RasterConfig(
                lane_thickness=3,
                stop_thickness=3,
            )

        self.transform_config = transform_config
        self.raster_config = raster_config

        # ----------------------------------------------------
        # Annotation grouping
        # JSONL은 Dataset 생성 시 한 번만 읽는다.
        # ----------------------------------------------------

        annotation_groups = defaultdict(list)

        with self.annotation_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:
                annotation = json.loads(line)

                annotation_groups[
                    annotation["file_name"]
                ].append(annotation)

        self.annotation_groups = dict(
            annotation_groups
        )

        # ----------------------------------------------------
        # 사용할 file_name 결정
        # ----------------------------------------------------

        if file_names is None:
            self.file_names = sorted(
                self.annotation_groups.keys()
            )
        else:
            self.file_names = list(
                file_names
            )

        # ----------------------------------------------------
        # Annotation 존재 여부 검증
        # ----------------------------------------------------

        missing_annotations = [
            file_name
            for file_name in self.file_names
            if file_name not in self.annotation_groups
        ]

        if missing_annotations:
            raise ValueError(
                "Annotation이 없는 file_name이 있습니다. "
                f"예: {missing_annotations[:5]}"
            )

        # ----------------------------------------------------
        # file_name → 실제 image path index
        # ----------------------------------------------------

        requested_names = set(
            self.file_names
        )

        image_candidates = defaultdict(list)

        for image_path in self.image_root.rglob("*"):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                continue

            if image_path.name not in requested_names:
                continue

            image_candidates[
                image_path.name
            ].append(image_path)

        self.image_path_map = {}

        for file_name in self.file_names:

            matches = image_candidates.get(
                file_name,
                [],
            )

            if len(matches) == 0:
                raise FileNotFoundError(
                    f"Image를 찾지 못했습니다: "
                    f"{file_name}"
                )

            if len(matches) > 1:
                raise RuntimeError(
                    f"동일한 file_name의 image가 "
                    f"여러 개 있습니다: "
                    f"{file_name}"
                )

            self.image_path_map[
                file_name
            ] = matches[0]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(
        self,
        index,
    ):
        # ----------------------------------------------------
        # Sample 정보
        # ----------------------------------------------------

        file_name = self.file_names[index]

        annotations = self.annotation_groups[
            file_name
        ]

        image_path = self.image_path_map[
            file_name
        ]

        # ----------------------------------------------------
        # RGB 읽기
        # ----------------------------------------------------

        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            raise RuntimeError(
                f"Image를 읽지 못했습니다: "
                f"{image_path}"
            )

        actual_height, actual_width = (
            image_bgr.shape[:2]
        )

        # ----------------------------------------------------
        # Annotation image_size 검증
        # JSONL: [height, width]
        # ----------------------------------------------------

        annotation_height = (
            annotations[0]["image_size"][0]
        )

        annotation_width = (
            annotations[0]["image_size"][1]
        )

        if (
            actual_height != annotation_height
            or actual_width != annotation_width
        ):
            raise ValueError(
                f"Image size mismatch: "
                f"{file_name} | "
                f"image={actual_width}x{actual_height}, "
                f"annotation="
                f"{annotation_width}x{annotation_height}"
            )

        # ----------------------------------------------------
        # RGB와 GT가 공유하는 TransformInfo
        # ----------------------------------------------------

        transform_info = compute_transform_info(
            original_width=actual_width,
            original_height=actual_height,
            config=self.transform_config,
        )

        # ----------------------------------------------------
        # RGB Transform
        # ----------------------------------------------------

        image_bgr = transform_image(
            image_bgr=image_bgr,
            transform_info=transform_info,
            config=self.transform_config,
        )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        image_tensor = (
            torch.from_numpy(
                image_rgb.copy()
            )
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        # ----------------------------------------------------
        # Segmentation GT
        # ----------------------------------------------------

        raster_result = (
            rasterize_segmentation_targets(
                annotations=annotations,
                original_width=actual_width,
                original_height=actual_height,
                transform_info=transform_info,
                raster_config=self.raster_config,
            )
        )

        # channel 0 = traffic_lane
        # channel 1 = stop_line
        # channel 2 = crosswalk

        seg_target_np = np.stack(
            [
                raster_result["lane_mask"],
                raster_result["stop_mask"],
                raster_result["crosswalk_mask"],
            ],
            axis=0,
        )

        # uint8 0/255 → float32 0/1
        seg_target_np = (
            seg_target_np.astype(np.float32)
            / 255.0
        )

        seg_target = torch.from_numpy(
            seg_target_np
        )

        # ----------------------------------------------------
        # RGB ↔ GT spatial contract
        # ----------------------------------------------------

        if (
            image_tensor.shape[-2:]
            != seg_target.shape[-2:]
        ):
            raise RuntimeError(
                f"Image / GT spatial mismatch: "
                f"{file_name}"
            )

        return {
            "image": image_tensor,
            "seg_target": seg_target,
            "file_name": file_name,
        }