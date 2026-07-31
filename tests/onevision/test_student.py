# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from onevision_zap.student import VisualUtilityStudentLayer


def test_student_layer_scores_every_image_token() -> None:
    # Given: hidden states with three image positions and two question positions.
    layer = VisualUtilityStudentLayer(
        hidden_dim=8,
        conv_dim=4,
        proj_dim=4,
        mlp_dim=8,
        num_conv_blocks=1,
        kernel_size=3,
    ).eval()
    hidden_states = torch.arange(48, dtype=torch.float32).reshape(1, 6, 8)

    # When: the student predicts visual utility.
    scores = layer(
        hidden_states,
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5]),
    )

    # Then: one score is produced for each image token.
    assert scores.shape == (1, 3)
