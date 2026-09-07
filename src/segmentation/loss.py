import torch
import torch.nn as nn


# ============================================================
# Soft Dice Loss
# - sample별 × channel별 계산
# - GT가 존재하는 channel만 Dice 평균에 포함
# ============================================================

def soft_dice_loss(
    logits,
    targets,
    smooth=1.0,
):
    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if logits.shape != targets.shape:
        raise ValueError(
            f"logits shape {logits.shape} and "
            f"targets shape {targets.shape} must match."
        )

    if smooth <= 0:
        raise ValueError(
            f"smooth must be greater than 0, got {smooth}."
        )


    # --------------------------------------------------------
    # logits → probability
    # [B, C, H, W]
    # --------------------------------------------------------

    probabilities = torch.sigmoid(
        logits
    )


    # --------------------------------------------------------
    # sample × channel 단위 Dice 계산
    #
    # H, W만 합산
    #
    # 결과 shape:
    # [B, C]
    # --------------------------------------------------------

    dims = (
        2,
        3,
    )

    intersection = (
        probabilities * targets
    ).sum(
        dim=dims
    )

    prediction_sum = probabilities.sum(
        dim=dims
    )

    target_sum = targets.sum(
        dim=dims
    )


    # --------------------------------------------------------
    # Dice Score
    # [B, C]
    # --------------------------------------------------------

    dice_score = (
        2.0 * intersection + smooth
    ) / (
        prediction_sum
        + target_sum
        + smooth
    )


    # --------------------------------------------------------
    # GT가 실제 존재하는 sample/channel만 선택
    # --------------------------------------------------------

    valid_mask = (
        target_sum > 0
    )


    # --------------------------------------------------------
    # Dice Loss
    # --------------------------------------------------------

    if valid_mask.any():

        valid_dice_score = dice_score[
            valid_mask
        ]

        dice_loss = (
            1.0
            - valid_dice_score
        ).mean()

    else:

        # Batch 전체에서 모든 GT channel이 비어 있는 경우
        #
        # Dice는 0으로 두고 BCE가 학습을 담당한다.
        # logits와 연결된 0을 만들어 gradient graph를 유지한다.
        dice_loss = (
            logits.sum() * 0.0
        )


    return dice_loss


# ============================================================
# Segmentation Loss
# BCEWithLogitsLoss + Soft Dice Loss
# ============================================================

class SegmentationLoss(nn.Module):

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        smooth=1.0,
    ):
        super().__init__()

        if bce_weight < 0:
            raise ValueError(
                f"bce_weight must be >= 0, got {bce_weight}."
            )

        if dice_weight < 0:
            raise ValueError(
                f"dice_weight must be >= 0, got {dice_weight}."
            )

        if smooth <= 0:
            raise ValueError(
                f"smooth must be greater than 0, got {smooth}."
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        self.bce = nn.BCEWithLogitsLoss()


    def forward(
        self,
        logits,
        targets,
    ):
        # ----------------------------------------------------
        # Shape validation
        # ----------------------------------------------------

        if logits.shape != targets.shape:
            raise ValueError(
                f"logits shape {logits.shape} and "
                f"targets shape {targets.shape} must match."
            )


        # ----------------------------------------------------
        # BCE Loss
        # - 모든 pixel / 모든 channel 감독
        # ----------------------------------------------------

        bce_loss = self.bce(
            logits,
            targets,
        )


        # ----------------------------------------------------
        # Soft Dice Loss
        # - sample × channel 단위
        # - non-empty GT channel만 평균
        # ----------------------------------------------------

        dice_loss = soft_dice_loss(
            logits,
            targets,
            smooth=self.smooth,
        )


        # ----------------------------------------------------
        # Total Loss
        # ----------------------------------------------------

        total_loss = (
            self.bce_weight * bce_loss
            +
            self.dice_weight * dice_loss
        )


        # ----------------------------------------------------
        # 학습 / logging에서 각각 확인할 수 있도록 반환
        # ----------------------------------------------------

        return {
            "loss": total_loss,
            "bce_loss": bce_loss,
            "dice_loss": dice_loss,
        }
        