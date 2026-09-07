import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Decoder Block
# Conv → GroupNorm → GELU
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_groups=32,
    ):
        super().__init__()

        if out_channels % num_groups != 0:
            raise ValueError(
                f"out_channels({out_channels}) must be divisible "
                f"by num_groups({num_groups})."
            )

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=out_channels,
            ),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Progressive Segmentation Head
# ============================================================

class SegmentationHead(nn.Module):

    def __init__(
        self,
        encoder_channels=(96, 192, 384, 768),
        decoder_channels=256,
        num_classes=3,
        num_groups=32,
    ):
        super().__init__()

        if len(encoder_channels) != 4:
            raise ValueError(
                "encoder_channels must contain exactly 4 values "
                "for c1, c2, c3, c4."
            )

        c1_channels, c2_channels, c3_channels, c4_channels = (
            encoder_channels
        )

        # ----------------------------------------------------
        # Encoder feature projections
        # ----------------------------------------------------

        self.c4_projection = nn.Conv2d(
            c4_channels,
            decoder_channels,
            kernel_size=1,
        )

        self.c3_projection = nn.Conv2d(
            c3_channels,
            decoder_channels,
            kernel_size=1,
        )

        self.c2_projection = nn.Conv2d(
            c2_channels,
            decoder_channels,
            kernel_size=1,
        )

        self.c1_projection = nn.Conv2d(
            c1_channels,
            decoder_channels,
            kernel_size=1,
        )

        # ----------------------------------------------------
        # Progressive Decoder Blocks
        # ----------------------------------------------------

        fusion_channels = decoder_channels * 2

        self.c43_fusion = DecoderBlock(
            in_channels=fusion_channels,
            out_channels=decoder_channels,
            num_groups=num_groups,
        )

        self.d3c2_fusion = DecoderBlock(
            in_channels=fusion_channels,
            out_channels=decoder_channels,
            num_groups=num_groups,
        )

        self.d2c1_fusion = DecoderBlock(
            in_channels=fusion_channels,
            out_channels=decoder_channels,
            num_groups=num_groups,
        )

        # ----------------------------------------------------
        # Final classifier
        # ----------------------------------------------------

        self.classifier = nn.Conv2d(
            decoder_channels,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        features,
        output_size,
    ):
        required_keys = (
            "c1",
            "c2",
            "c3",
            "c4",
        )

        missing_keys = [
            key
            for key in required_keys
            if key not in features
        ]

        if missing_keys:
            raise KeyError(
                f"Missing encoder feature keys: {missing_keys}"
            )

        c1 = features["c1"]
        c2 = features["c2"]
        c3 = features["c3"]
        c4 = features["c4"]

        # ====================================================
        # C4 + C3 → D3
        # ====================================================

        c4 = self.c4_projection(c4)
        c3 = self.c3_projection(c3)

        c4 = F.interpolate(
            c4,
            size=c3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        d3 = torch.cat(
            [c4, c3],
            dim=1,
        )

        d3 = self.c43_fusion(
            d3
        )

        # ====================================================
        # D3 + C2 → D2
        # ====================================================

        c2 = self.c2_projection(c2)

        d3 = F.interpolate(
            d3,
            size=c2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        d2 = torch.cat(
            [d3, c2],
            dim=1,
        )

        d2 = self.d3c2_fusion(
            d2
        )

        # ====================================================
        # D2 + C1 → D1
        # ====================================================

        c1 = self.c1_projection(c1)

        d2 = F.interpolate(
            d2,
            size=c1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        d1 = torch.cat(
            [d2, c1],
            dim=1,
        )

        d1 = self.d2c1_fusion(
            d1
        )

        # ====================================================
        # Final Segmentation Logits
        # ====================================================

        logits = self.classifier(
            d1
        )

        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits
        