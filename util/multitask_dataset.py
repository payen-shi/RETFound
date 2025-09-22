"""Dataset utilities for multi-task fine-tuning on RETFound."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


_SUPPORTED_EXTENSIONS: List[str] = [
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".PNG",
    ".JPG",
    ".JPEG",
    ".BMP",
    ".TIF",
    ".TIFF",
]


@dataclass(frozen=True)
class SampleRecord:
    """Container holding metadata for a single training sample."""

    image_id: str
    image_path: Path
    labels: Mapping[str, int]


class RetFoundMultiTaskDataset(Dataset):
    """Dataset that reads images from a single folder with labels in an Excel file.

    Parameters
    ----------
    dataframe:
        Pandas DataFrame containing at least an ``image_id`` column and the label
        columns defined in ``label_columns``.
    image_root:
        Directory that stores all images referenced in ``dataframe``.
    label_columns:
        Mapping from task name to the column name in ``dataframe``. The values
        inside those columns must be encoded as integers starting from zero.
    transform:
        Optional torchvision-compatible transform applied to every image.
    image_extension:
        Optional default extension appended to ``image_id`` when resolving the
        file path. When ``None`` the loader will try a list of common
        ophthalmic image extensions.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_root: str | Path,
        label_columns: Mapping[str, str],
        transform=None,
        image_extension: Optional[str] = None,
    ) -> None:
        super().__init__()
        if "image_id" not in dataframe.columns:
            raise ValueError("The annotation file must contain an 'image_id' column.")

        self.transform = transform
        self.image_root = Path(image_root)
        self.image_extension = image_extension
        self.label_columns = dict(label_columns)

        missing_cols = [col for col in self.label_columns.values() if col not in dataframe.columns]
        if missing_cols:
            raise ValueError(
                "The annotation file is missing label columns: "
                + ", ".join(sorted(missing_cols))
            )

        self.samples: List[SampleRecord] = []
        for row in dataframe.itertuples(index=False):
            image_id = str(getattr(row, "image_id"))
            image_path = self._resolve_path(image_id)

            labels: Dict[str, int] = {}
            for task, column in self.label_columns.items():
                value = getattr(row, column)
                if pd.isna(value):
                    raise ValueError(
                        f"Found missing value for task '{task}' in image '{image_id}'. "
                        "Please drop rows with missing labels before creating the dataset."
                    )
                labels[task] = int(value)

            self.samples.append(SampleRecord(image_id=image_id, image_path=image_path, labels=labels))

        if len(dataframe) > 0 and not self.samples:
            raise ValueError("No samples were loaded. Please check the annotation file and image directory.")

    def _resolve_path(self, image_id: str) -> Path:
        """Resolve the absolute path for a given ``image_id``."""
        # Direct match when an extension is already provided.
        candidate = Path(image_id)
        if candidate.suffix:
            full_path = self.image_root / candidate
            if full_path.exists():
                return full_path
            raise FileNotFoundError(f"Image '{candidate}' referenced in the annotation file was not found.")

        if self.image_extension:
            full_path = self.image_root / f"{image_id}{self.image_extension}"
            if full_path.exists():
                return full_path

        for suffix in _SUPPORTED_EXTENSIONS:
            full_path = self.image_root / f"{image_id}{suffix}"
            if full_path.exists():
                return full_path

        raise FileNotFoundError(
            f"Unable to locate an image file for id '{image_id}'. "
            "Specify --image-ext when running the training script if your images use a non-standard extension."
        )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, index: int):
        record = self.samples[index]
        image = Image.open(record.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = {task: torch.tensor(label, dtype=torch.long) for task, label in record.labels.items()}
        return image, target, record.image_id

    @property
    def task_names(self) -> Iterable[str]:
        return self.label_columns.keys()

    @property
    def image_ids(self) -> List[str]:  # pragma: no cover - convenience wrapper
        return [record.image_id for record in self.samples]

