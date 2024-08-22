# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
from copy import copy
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch

from tensordict import NonTensorData, TensorDictBase

from tensordict.utils import NestedKey

from torchrl.envs.transforms import ObservationTransform
from torchrl.record.recorder import VideoRecorder
from torchrl.record.loggers import Logger

_has_tv = importlib.util.find_spec("torchvision", None) is not None


class IsaacVideoRecorder(VideoRecorder):
  

    def __init__(
        self,
        logger: Logger,
        tag: str,
        in_keys: Optional[Sequence[NestedKey]] = None,
        skip: int | None = None,
        center_crop: Optional[int] = None,
        make_grid: bool | None = None,
        out_keys: Optional[Sequence[NestedKey]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            logger, 
            tag, 
            in_keys, 
            skip,
            center_crop,
            make_grid,
            out_keys, 
            **kwargs
        )

    def dump(self, suffix: Optional[str] = None) -> None:
        """Writes the video to the ``self.logger`` attribute.

        Calling ``dump`` when no image has been stored in a no-op.

        Args:
            suffix (str, optional): a suffix for the video to be recorded
        """
        if self.obs:
            obs = torch.stack(self.obs, 0).unsqueeze(0).cpu()
        else:
            obs = None
        self.obs = []
        if obs is not None:
            if suffix is None:
                tag = self.tag
            else:
                tag = "_".join([self.tag, suffix])
            if self.logger is not None:
                self.logger.log_video(
                    name=tag,
                    video=obs,
                    step=self.iter,
                    **self.video_kwargs,
                )
        self.iter += 1
        self.count = 0
        self.obs = []

