# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from copy import deepcopy
from typing import Dict, Optional

import numpy as np

import torch
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

from training.dataset.vos_raw_dataset import VOSRawDataset
from training.dataset.vos_sampler import VOSSampler
from training.dataset.vos_segment_loader import JSONSegmentLoader

from training.utils.data_utils import Frame, Object, VideoDatapoint

MAX_RETRIES = 100


class VOSDataset(VisionDataset):
    def __init__(
        self,
        transforms,
        training: bool,
        video_dataset: VOSRawDataset,
        sampler: VOSSampler,
        multiplier: int,
        always_target=True,
        target_segments_available=True,
    ):
        self._transforms = transforms
        self.training = training
        self.video_dataset = video_dataset
        self.sampler = sampler

        self.repeat_factors = torch.ones(len(self.video_dataset), dtype=torch.float32)
        self.repeat_factors *= multiplier
        print(f"Raw dataset length = {len(self.video_dataset)}")

        self.curr_epoch = 0  # Used in case data loader behavior changes across epochs
        self.always_target = always_target
        self.target_segments_available = target_segments_available

    def _get_datapoint(self, idx):
        video = None
        segment_loader = None
        sampled_frms_and_objs = None
        last_error = None

        for retry in range(MAX_RETRIES):
            try:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                # sample a video
                video, segment_loader = self.video_dataset.get_video(idx)
                # sample frames and object indices to be used in a datapoint
                sampled_frms_and_objs = self.sampler.sample(
                    video, segment_loader, epoch=self.curr_epoch
                )
                break  # Succesfully loaded video
            except Exception as e:
                last_error = e
                if self.training:
                    logging.warning(
                        f"Loading failed (id={idx}); Retry {retry} with exception: {e}"
                    )
                    idx = random.randrange(0, len(self.video_dataset))
                else:
                    # Shouldn't fail to load a val video
                    raise e

        if video is None or segment_loader is None or sampled_frms_and_objs is None:
            raise last_error if last_error is not None else RuntimeError(
                f"Failed to load datapoint {idx}"
            )

        datapoint = self.construct(video, sampled_frms_and_objs, segment_loader)
        for transform in self._transforms:
            datapoint = transform(datapoint, epoch=self.curr_epoch)
        return datapoint

    def construct(self, video, sampled_frms_and_objs, segment_loader):
        """
        Constructs a VideoDatapoint sample to pass to transforms
        """
        sampled_frames = sampled_frms_and_objs.frames
        sampled_object_ids = sampled_frms_and_objs.object_ids

        images = []
        rgb_images = load_images(sampled_frames)
        # Iterate over the sampled frames and store their rgb data and object data (bbox, segment)
        for frame_idx, frame in enumerate(sampled_frames):
            w, h = rgb_images[frame_idx].size
            images.append(
                Frame(
                    data=rgb_images[frame_idx],
                    objects=[],
                )
            )
            # We load the gt segments associated with the current frame
            if isinstance(segment_loader, JSONSegmentLoader):
                segments = segment_loader.load(
                    frame.frame_idx, obj_ids=sampled_object_ids
                )
            else:
                segments = segment_loader.load(frame.frame_idx)
            for obj_id in sampled_object_ids:
                # Extract the segment
                if obj_id in segments:
                    assert (
                        segments[obj_id] is not None
                    ), "None targets are not supported"
                    # segment is uint8 and remains uint8 throughout the transforms
                    segment = segments[obj_id].to(torch.uint8)
                else:
                    # There is no target, we either use a zero mask target or drop this object
                    if not self.always_target:
                        continue
                    segment = torch.zeros(h, w, dtype=torch.uint8)

                images[frame_idx].objects.append(
                    Object(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=segment,
                    )
                )
        return VideoDatapoint(
            frames=images,
            video_id=video.video_id,
            size=(h, w),
        )

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
        return len(self.video_dataset)


def load_images(frames):
    all_images = []
    cache = {}
    for frame in frames:
        if frame.data is None:
            # Load the frame rgb data from file
            path = frame.image_path
            if path in cache:
                all_images.append(deepcopy(all_images[cache[path]]))
                continue
            with g_pathmgr.open(path, "rb") as fopen:
                all_images.append(PILImage.open(fopen).convert("RGB"))
            cache[path] = len(all_images) - 1
        else:
            # The frame rgb data has already been loaded
            # Convert it to a PILImage
            all_images.append(tensor_2_PIL(frame.data))

    return all_images


def tensor_2_PIL(data: torch.Tensor) -> PILImage.Image:
    data = data.cpu().numpy().transpose((1, 2, 0)) * 255.0
    data = data.astype(np.uint8)
    return PILImage.fromarray(data)



class MultiAxisVOSDataset(torch.utils.data.Dataset):
    """Dataset wrapper that samples uniformly across multiple axis-specific sub-datasets.
    
    This implements Strategy 2: during training, each batch item is randomly
    sampled from Z, Y, or X pseudo-volumes with equal probability.
    
    The underlying sub-datasets are standard VOSDataset instances, one per axis.
    
    Args:
        z_dataset: VOSDataset for Z-axis pseudo-volumes
        y_dataset: VOSDataset for Y-axis pseudo-volumes  
        x_dataset: VOSDataset for X-axis pseudo-volumes
        axis_weights: Sampling weights for each axis (defaults to uniform)
        transforms: Shared transforms (usually identity — applied by sub-datasets)
        training: Whether in training mode
        multiplier: Epoch multiplier
    """
    
    def __init__(
        self,
        z_dataset: Optional[torch.utils.data.Dataset] = None,
        y_dataset: Optional[torch.utils.data.Dataset] = None,
        x_dataset: Optional[torch.utils.data.Dataset] = None,
        axis_weights: Optional[Dict[str, float]] = None,
        transforms=None,
        training: bool = True,
        multiplier: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        
        self.datasets = {}
        if z_dataset is not None:
            self.datasets["z"] = z_dataset
        if y_dataset is not None:
            self.datasets["y"] = y_dataset
        if x_dataset is not None:
            self.datasets["x"] = x_dataset
        
        if not self.datasets:
            raise ValueError("At least one axis dataset must be provided")
        
        # Setup sampling weights
        if axis_weights is None:
            axis_weights = {ax: 1.0 / len(self.datasets) for ax in self.datasets}
        else:
            total = sum(axis_weights.get(ax, 0) for ax in self.datasets)
            if total <= 0:
                raise ValueError("axis_weights must contain a positive weight for at least one axis")
            axis_weights = {ax: axis_weights.get(ax, 0) / total for ax in self.datasets}
        
        self.axis_weights = axis_weights
        self.axes = list(self.datasets.keys())
        self.weights = [axis_weights[ax] for ax in self.axes]
        
        sizes = [len(ds) for ds in self.datasets.values()]
        self._dataset_lens = dict(zip(self.axes, sizes))
        self._effective_len = int(
            sum(s * w for s, w in zip(sizes, self.weights)) * multiplier
        )
        self._effective_len = max(1, self._effective_len)
        self.repeat_factors = torch.ones(self._effective_len, dtype=torch.float32)
        
        self.transforms = transforms
        self.training = training
    
    def __len__(self) -> int:
        return self._effective_len
    
    def __getitem__(self, index: int) -> dict:
        """Randomly sample an axis, then get an item from that sub-dataset."""
        axis = random.choices(self.axes, weights=self.weights, k=1)[0]

        axis_dataset = self.datasets[axis]
        axis_idx = random.randint(0, len(axis_dataset) - 1)

        sample = axis_dataset[axis_idx]

        if isinstance(sample, dict):
            sample["_axis"] = axis
        else:
            setattr(sample, "axis", axis)
        
        return sample
    
    def get_axis_statistics(self) -> Dict[str, int]:
        """Return number of samples in each axis dataset."""
        return self._dataset_lens
