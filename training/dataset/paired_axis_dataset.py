from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def natural_sort_key(value: str | Path):
    value = str(value)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", value)]


def group_files_by_case(files: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in files:
        match = re.match(r"^(.+?)[_-](\d+)$", path.stem)
        if match:
            case_id = match.group(1)
            index = int(match.group(2))
        else:
            case_id = path.stem
            index = 0
        groups[case_id].append((index, path))
    return {
        case_id: [path for _, path in sorted(items, key=lambda item: item[0])]
        for case_id, items in sorted(groups.items(), key=lambda item: natural_sort_key(item[0]))
    }


def load_case_volume(
    case_files: list[Path],
    image_channel_index: int = 0,
    max_slices: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(case_files[0], allow_pickle=True) as data:
        first_imgs = np.asarray(data["imgs"])
        first_gts = np.asarray(data["gts"])

    if first_gts.ndim == 3:
        imgs = first_imgs.astype(np.float32)
        gts = first_gts.astype(np.float32)
        if imgs.ndim == 4:
            imgs = imgs[..., min(image_channel_index, imgs.shape[-1] - 1)]
        if max_slices is not None:
            imgs = imgs[:max_slices]
            gts = gts[:max_slices]
        return imgs, gts

    imgs_list = []
    gts_list = []
    for index, path in enumerate(case_files):
        if max_slices is not None and index >= max_slices:
            break
        with np.load(path, allow_pickle=True) as data:
            img = np.asarray(data["imgs"], dtype=np.float32)
            gt = np.asarray(data["gts"], dtype=np.float32)
        if img.ndim == 3:
            img = img[..., min(image_channel_index, img.shape[-1] - 1)]
        imgs_list.append(img)
        gts_list.append(gt)
    return np.stack(imgs_list, axis=0), np.stack(gts_list, axis=0)


def extract_axis_window(
    imgs: np.ndarray,
    gts: np.ndarray,
    axis: str,
    start: int,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if axis == "z":
        return imgs[start:start + num_frames], gts[start:start + num_frames]
    if axis == "y":
        return imgs[:, start:start + num_frames, :].transpose(1, 0, 2), gts[:, start:start + num_frames, :].transpose(1, 0, 2)
    if axis == "x":
        return imgs[:, :, start:start + num_frames].transpose(2, 0, 1), gts[:, :, start:start + num_frames].transpose(2, 0, 1)
    raise ValueError(f"Unknown axis: {axis}")


class PairedAxisDataset(Dataset):
    """Return Z/Y/X clips from the same 3D case with volume-coordinate metadata.

    This is intentionally raw-tensor oriented. A paired trainer or collate layer
    should convert each axis clip into the SAM2 BatchedVideoDatapoint format.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        file_list_txt: str | Path | None = None,
        num_frames: int = 8,
        image_channel_index: int = 0,
        max_slices_per_case: int | None = None,
        require_mask: bool = True,
        multiplier: int = 1,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.num_frames = int(num_frames)
        self.image_channel_index = int(image_channel_index)
        self.max_slices_per_case = max_slices_per_case
        self.require_mask = require_mask
        self.multiplier = int(multiplier)

        files = sorted(self.dataset_dir.rglob("*.npz"), key=natural_sort_key)
        if file_list_txt is not None:
            allowed = {
                line.strip().removesuffix(".npz")
                for line in Path(file_list_txt).read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            files = [path for path in files if path.relative_to(self.dataset_dir).with_suffix("").as_posix() in allowed]

        self.case_groups = group_files_by_case(files)
        self.case_ids = list(self.case_groups.keys())
        if not self.case_ids:
            raise ValueError(f"No NPZ cases found in {self.dataset_dir}")
        self.repeat_factors = torch.ones(len(self), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.case_ids) * self.multiplier

    def _sample_start(self, axis_len: int, mask_any: np.ndarray | None) -> int:
        max_start = max(0, axis_len - self.num_frames)
        if mask_any is not None and mask_any.any():
            candidates = np.where(mask_any)[0]
            center = int(random.choice(candidates.tolist()))
            return max(0, min(center - self.num_frames // 2, max_start))
        return random.randint(0, max_start)

    def __getitem__(self, index: int) -> dict:
        case_id = self.case_ids[index % len(self.case_ids)]
        imgs, gts = load_case_volume(
            self.case_groups[case_id],
            image_channel_index=self.image_channel_index,
            max_slices=self.max_slices_per_case,
        )
        if self.require_mask and not (gts > 0).any():
            raise ValueError(f"Case {case_id} has no foreground mask")

        d, h, w = imgs.shape
        starts = {
            "z": self._sample_start(d, (gts > 0).any(axis=(1, 2))),
            "y": self._sample_start(h, (gts > 0).any(axis=(0, 2))),
            "x": self._sample_start(w, (gts > 0).any(axis=(0, 1))),
        }

        sample = {
            "case_id": case_id,
            "volume_shape": torch.tensor([d, h, w], dtype=torch.long),
        }
        for axis in ("z", "y", "x"):
            clip_img, clip_gt = extract_axis_window(imgs, gts, axis, starts[axis], self.num_frames)
            sample[axis] = {
                "imgs": torch.from_numpy(clip_img.astype(np.float32)),
                "gts": torch.from_numpy(clip_gt.astype(np.float32)),
                "start": torch.tensor(starts[axis], dtype=torch.long),
                "axis": axis,
            }
        return sample

