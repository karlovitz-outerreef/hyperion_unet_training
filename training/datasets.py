from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class Dir2Dataset(Dataset):
    """
    Dir2Dataset(root, images_path, labels_path=None, img_transform=None, label_transform=None, ...)

    - Builds its index from all *.npy under {root}/{images_path}
    - If labels_path is provided, it pairs by filename stem (without extension)
    - Uses lazy, memory-mapped np.load for fast first-touch on SageMaker
    """

    def __init__(
        self,
        root: str | Path,
        images_path: str | Path,
        labels_path: Optional[str | Path] = None,
        augment: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        drop_missing_labels: bool = True,
        return_paths: bool = False,
    ):
        self.root = Path(root)
        self.images_dir = (self.root / images_path).resolve()
        self.labels_dir = (self.root / labels_path).resolve() if labels_path is not None else None

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        self.augment = augment
        self.return_paths = return_paths

        # Build image file list (sorted for determinism)
        img_files = sorted(self.images_dir.rglob("*.npy"))
        if not img_files:
            raise RuntimeError(f"No .npy files found under {self.images_dir}")

        # Optional: map label stems -> path
        label_map: Dict[str, Path] = {}
        if self.labels_dir is not None:
            if not self.labels_dir.exists():
                raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
            for p in self.labels_dir.rglob("*.npy"):
                label_map[p.stem] = p.resolve()

        # Build paired index
        self._records: List[Tuple[Path, Optional[Path]]] = []
        for img_p in img_files:
            lbl_p = None
            if label_map:
                lbl_p = label_map.get(img_p.stem)
                if lbl_p is None and drop_missing_labels:
                    continue
            self._records.append((img_p.resolve(), lbl_p.resolve() if lbl_p else None))

        if not self._records:
            raise RuntimeError("No records after pairing/filtering—check paths and filenames.")

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _load_npy_lazy(path: Path) -> np.ndarray:
        # Lazy, memory-mapped load (doesn't pull entire file immediately)
        return np.load(path, mmap_mode="r")

    def __getitem__(self, idx: int):
        img_path, lbl_path = self._records[idx]

        # Lazy load
        img = self._load_npy_lazy(img_path)
        lbl = self._load_npy_lazy(lbl_path) if lbl_path is not None else None

        # Transforms (e.g. numpy -> tensor)
        if self.augment is not None:
            if lbl is not None:
                out = self.augment(image=img, label=lbl)
                img, lbl = out['image'], out['label']
            else:
                out = self.augment(image=img)
                img = out['image']

        if self.return_paths:
            return img, lbl, str(img_path), (str(lbl_path) if lbl_path else None)
        return img, lbl
