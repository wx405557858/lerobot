import math
from dataclasses import asdict
from typing import Callable, Literal

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.distributions import MultivariateNormal, TanhTransform, Transform, TransformedDistribution

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.normalize import NormalizeBuffer
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.sac.configuration_sac import SACConfig, is_image_feature
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.utils import get_device_from_parameters


class SACSmolVLAObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: SACConfig, input_normalizer: nn.Module, ds_meta: LeRobotDatasetMetadata) -> None:
        super().__init__()
        self.config = config
        self.input_normalization = input_normalizer
        self.ds_meta = ds_meta
        self._init_smolvla_layers()
        self._compute_output_dim()

    def _init_smolvla_layers(self) -> None:
        self.image_keys = [k for k in self.config.input_features if is_image_feature(k)]
        self.has_images = bool(self.image_keys)
        if not self.has_images:
            return
        self.policy_config = SmolVLAConfig(
            device="cuda",
            chunk_size=1,
            n_action_steps=1,
            max_action_dim=32,
            num_vlm_layers=16,
        )
        policy_path = "lerobot/smolvla_base"
        self.policy_config.pretrained_path = policy_path
        self.smolvla : SmolVLAPolicy = make_policy(cfg=self.policy_config, ds_meta=self.ds_meta)

    def get_cached_embeddings(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        # print("task in get_cached_embeddings:", obs["task"][:1])
        obs = self.input_normalization(obs)
        embeddings = self.smolvla.forward_embeddings(obs)
        return {"embeddings": embeddings}

    def _compute_output_dim(self) -> None:
        self._out_dim = self.policy_config.max_action_dim

    def forward(
        self, obs: dict[str, Tensor], cache: dict[str, Tensor] | None = None, detach: bool = False
    ) -> Tensor:
        if cache is not None and "embeddings" in cache:
            return cache["embeddings"]
        
        obs = self.input_normalization(obs)
        embeddings = self.smolvla.forward_embeddings(obs)

        if embeddings is not None:
            return embeddings

        raise ValueError(
            "No parts to concatenate, you should have at least one image or environment state or state or text"
        )

    @property
    def output_dim(self) -> int:
        return self._out_dim