from pathlib import Path
from collections import Counter
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "images"
    / "train_09"
)

JSONL_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "segmentation_annotations_v2.jsonl"
)


print("=" * 70)
print("PROJECT_ROOT :", PROJECT_ROOT)
print("IMAGE_DIR    :", IMAGE_DIR)
print("JSONL_PATH   :", JSONL_PATH)
print("=" * 70)


# ============================================================
# 2. 경로 확인
# ============================================================

assert IMAGE_DIR.is_dir(), (
    f"이미지 폴더가 없습니다.\n"
    f"{IMAGE_DIR}"
)

assert JSONL_PATH.is_file(), (
    f"JSONL 파일이 없습니다.\n"
    f"{JSONL_PATH}"
)


jpg_files = sorted(IMAGE_DIR.glob("*.jpg"))

print("JPG 개수 :", len(jpg_files))

if jpg_files:
    print("JPG 예시 :", jpg_files[0].name)


# ============================================================
# 3. traffic_lane의 lane_type 전체 분포 확인
# ============================================================

lane_type_counter = Counter()

with JSONL_PATH.open("r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        ann = json.loads(line)

        if ann.get("class") == "traffic_lane":

            lane_type = ann.get(
                "lane_type",
                "missing"
            )

            lane_type_counter[lane_type] += 1


print("\n" + "=" * 70)
print("[traffic_lane lane_type 분포]")
print("=" * 70)

for lane_type, count in lane_type_counter.items():

    print(
        f"{lane_type:15s}: {count:,}"
    )


# ============================================================
# 4. dotted polyline rasterization 함수
# ============================================================

def draw_dotted_polyline(
    mask,
    points,
    thickness=8,
    dash_length=24,
    gap_length=18,
):
    """
    하나의 polyline geometry를 따라
    dotted 형태로 끊어서 rasterize한다.

    주의:
    JSONL은 dotted라는 속성과 lane geometry를 제공하지만
    실제 각 dash의 시작/끝 좌표는 제공하지 않는다.

    따라서 dash_length / gap_length는
    현재 테스트용 rasterization 파라미터이다.
    """

    if len(points) < 2:
        return

    if dash_length <= 0:
        raise ValueError(
            "dash_length는 0보다 커야 합니다."
        )

    if gap_length < 0:
        raise ValueError(
            "gap_length는 0 이상이어야 합니다."
        )

    pattern_length = (
        dash_length
        + gap_length
    )

    # 이전 segment까지 누적된 길이
    # → polyline vertex를 지나도 점선 패턴 유지
    accumulated_length = 0.0


    for i in range(
        len(points) - 1
    ):

        p1 = points[i].astype(
            np.float32
        )

        p2 = points[i + 1].astype(
            np.float32
        )


        vector = p2 - p1

        segment_length = float(
            np.linalg.norm(vector)
        )


        # 같은 좌표가 연속으로 들어온 경우
        if segment_length <= 1e-6:
            continue


        direction = (
            vector
            / segment_length
        )


        distance = 0.0


        while (
            distance
            < segment_length - 1e-6
        ):

            global_distance = (
                accumulated_length
                + distance
            )

            phase = (
                global_distance
                % pattern_length
            )


            # =================================================
            # dash 구간
            # =================================================

            if phase < dash_length:

                remaining_dash = (
                    dash_length
                    - phase
                )

                draw_length = min(
                    remaining_dash,
                    segment_length - distance
                )


                if draw_length <= 1e-6:
                    break


                start = (
                    p1
                    + direction * distance
                )

                end = (
                    p1
                    + direction
                    * (
                        distance
                        + draw_length
                    )
                )


                cv2.line(
                    mask,
                    tuple(
                        np.round(start)
                        .astype(int)
                    ),
                    tuple(
                        np.round(end)
                        .astype(int)
                    ),
                    color=255,
                    thickness=thickness,
                    lineType=cv2.LINE_8
                )


                distance += draw_length


            # =================================================
            # gap 구간
            # =================================================

            else:

                remaining_gap = (
                    pattern_length
                    - phase
                )

                skip_length = min(
                    remaining_gap,
                    segment_length - distance
                )


                if skip_length <= 1e-6:
                    break


                distance += skip_length


        accumulated_length += (
            segment_length
        )


# ============================================================
# 5. 테스트할 이미지
# ============================================================

TEST_FILE = "13883084.jpg"

IMAGE_PATH = (
    IMAGE_DIR
    / TEST_FILE
)


assert IMAGE_PATH.is_file(), (
    f"테스트 이미지를 찾을 수 없습니다.\n"
    f"{IMAGE_PATH}"
)


# ============================================================
# 6. 테스트 이미지의 annotation만 가져오기
# ============================================================

annotations = []


with JSONL_PATH.open(
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        ann = json.loads(line)


        if (
            ann.get("file_name")
            == TEST_FILE
        ):

            annotations.append(ann)


print("\n" + "=" * 70)
print("[테스트 이미지 annotation]")
print("=" * 70)

print(
    "파일명 :",
    TEST_FILE
)

print(
    "annotation 수 :",
    len(annotations)
)


for ann in annotations:

    print(
        f'id={ann.get("annotation_id", -1):>3} | '
        f'class={ann.get("class", "unknown"):<15} | '
        f'lane_type={str(ann.get("lane_type")):<10} | '
        f'category={ann.get("category", "unknown"):<10} | '
        f'points={len(ann.get("data", []))}'
    )


# ============================================================
# 7. 원본 이미지 읽기
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)


assert image is not None, (
    "이미지를 읽지 못했습니다."
)


H, W = image.shape[:2]


print(
    "\n실제 이미지 크기 :",
    H,
    W
)


# ============================================================
# 8. 클래스별 binary mask 생성
# ============================================================

lane_mask = np.zeros(
    (H, W),
    dtype=np.uint8
)

stop_mask = np.zeros(
    (H, W),
    dtype=np.uint8
)

crosswalk_mask = np.zeros(
    (H, W),
    dtype=np.uint8
)


# ============================================================
# 9. Rasterization 설정
# ============================================================

# 아직 테스트용 임시값
LANE_THICKNESS = 8

STOP_THICKNESS = 12


# dotted lane 테스트 파라미터
DOTTED_DASH_LENGTH = 24

DOTTED_GAP_LENGTH = 18


print("\n" + "=" * 70)
print("[Rasterization 설정]")
print("=" * 70)

print(
    "LANE_THICKNESS       :",
    LANE_THICKNESS
)

print(
    "STOP_THICKNESS       :",
    STOP_THICKNESS
)

print(
    "DOTTED_DASH_LENGTH   :",
    DOTTED_DASH_LENGTH
)

print(
    "DOTTED_GAP_LENGTH    :",
    DOTTED_GAP_LENGTH
)


# ============================================================
# 10. Rasterization
# ============================================================

for ann in annotations:

    cls = ann.get(
        "class"
    )

    category = ann.get(
        "category"
    )

    data = ann.get(
        "data",
        []
    )


    # 좌표가 없는 annotation
    if not data:
        continue


    points = np.array(
        [
            [
                point["x"],
                point["y"]
            ]
            for point in data
        ],
        dtype=np.int32
    )


    # ========================================================
    # 이미지 영역 밖 좌표 clip
    # ========================================================

    points[:, 0] = np.clip(
        points[:, 0],
        0,
        W - 1
    )

    points[:, 1] = np.clip(
        points[:, 1],
        0,
        H - 1
    )


    points_cv = points.reshape(
        -1,
        1,
        2
    )


    # ========================================================
    # traffic_lane
    # ========================================================

    if (
        cls == "traffic_lane"
        and category == "polyline"
    ):

        if len(points) < 2:
            continue


        lane_type = ann.get(
            "lane_type",
            "unknown"
        )


        # ----------------------------------------------------
        # solid
        # ----------------------------------------------------

        if lane_type == "solid":

            cv2.polylines(
                lane_mask,
                [points_cv],
                isClosed=False,
                color=255,
                thickness=LANE_THICKNESS,
                lineType=cv2.LINE_8
            )


        # ----------------------------------------------------
        # dotted
        # ----------------------------------------------------

        elif lane_type == "dotted":

            draw_dotted_polyline(
                lane_mask,
                points,
                thickness=LANE_THICKNESS,
                dash_length=DOTTED_DASH_LENGTH,
                gap_length=DOTTED_GAP_LENGTH
            )


        # ----------------------------------------------------
        # 예상하지 못한 lane_type
        # ----------------------------------------------------

        else:

            print(
                "[WARNING] "
                f"unknown lane_type={lane_type} | "
                f"file={ann.get('file_name')} | "
                f"id={ann.get('annotation_id')}"
            )


            # geometry 자체를 잃지 않기 위해
            # 일단 연속선으로 처리
            cv2.polylines(
                lane_mask,
                [points_cv],
                isClosed=False,
                color=255,
                thickness=LANE_THICKNESS,
                lineType=cv2.LINE_8
            )


    # ========================================================
    # stop_line
    # ========================================================

    elif (
        cls == "stop_line"
        and category == "polyline"
    ):

        if len(points) < 2:
            continue


        cv2.polylines(
            stop_mask,
            [points_cv],
            isClosed=False,
            color=255,
            thickness=STOP_THICKNESS,
            lineType=cv2.LINE_8
        )


    # ========================================================
    # crosswalk
    # ========================================================

    elif (
        cls == "crosswalk"
        and category == "polygon"
    ):

        # polygon 최소 3점 필요
        if len(points) < 3:
            continue


        cv2.fillPoly(
            crosswalk_mask,
            [points_cv],
            color=255
        )


# ============================================================
# 11. Rasterization 결과 통계
# ============================================================

print("\n" + "=" * 70)
print("[Rasterization 결과]")
print("=" * 70)


print(
    "traffic_lane pixels :",
    np.count_nonzero(
        lane_mask
    )
)

print(
    "stop_line pixels    :",
    np.count_nonzero(
        stop_mask
    )
)

print(
    "crosswalk pixels    :",
    np.count_nonzero(
        crosswalk_mask
    )
)


# ============================================================
# 12. 원본 이미지 RGB 변환
# ============================================================

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 13. Segmentation 컬러 시각화
#
# traffic_lane = 빨강
# stop_line    = 초록
# crosswalk    = 파랑
#
# background는 아무것도 칠하지 않음
# ============================================================

seg_color = np.zeros(
    (H, W, 3),
    dtype=np.uint8
)


seg_color[
    lane_mask > 0
] = [
    255,
    0,
    0
]


seg_color[
    stop_mask > 0
] = [
    0,
    255,
    0
]


seg_color[
    crosswalk_mask > 0
] = [
    0,
    0,
    255
]


# ============================================================
# 14. 원본 + segmentation overlay
# ============================================================

overlay = cv2.addWeighted(
    image_rgb,
    0.7,
    seg_color,
    0.6,
    0
)


# ============================================================
# 15. Overlay 출력
# ============================================================

plt.figure(
    figsize=(18, 10)
)


plt.imshow(
    overlay
)


plt.title(
    "Red = traffic_lane | "
    "Green = stop_line | "
    "Blue = crosswalk"
)


plt.axis(
    "off"
)


plt.tight_layout()

plt.show()


# ============================================================
# 16. 순수 segmentation mask도 출력
# ============================================================

plt.figure(
    figsize=(18, 10)
)


plt.imshow(
    seg_color
)


plt.title(
    "Segmentation Mask | "
    "Red = traffic_lane | "
    "Green = stop_line | "
    "Blue = crosswalk | "
    "Black = background"
)


plt.axis(
    "off"
)


plt.tight_layout()

plt.show()