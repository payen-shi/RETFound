#!/usr/bin/env python3
"""End-to-end multi-task fine-tuning script for RETFound."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms

import models_vit as models
from util.multitask_dataset import RetFoundMultiTaskDataset
from util.pos_embed import interpolate_pos_embed


TaskColumns = Mapping[str, str]
LabelMappings = Dict[str, Dict[str, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-task fine-tuning for RETFound")
    parser.add_argument("--images-dir", type=str, required=True, help="Directory containing the raw images.")
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to the Excel file (.xlsx) storing the labels. Must include an 'image_id' column.",
    )
    parser.add_argument("--output-dir", type=str, default="multitask_runs", help="Directory used to store checkpoints.")
    parser.add_argument("--image-ext", type=str, default=None, help="Optional default file extension, e.g. '.jpg'.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay for AdamW.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers.")
    parser.add_argument("--model", type=str, default="RETFound_mae", choices=["RETFound_mae"], help="Backbone to fine-tune.")
    parser.add_argument("--finetune", type=str, default="", help="Path to the pre-trained RETFound checkpoint (.pth file).")
    parser.add_argument("--drop-path", type=float, default=0.2, help="Drop path rate for the backbone.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Only train the task-specific heads.")
    parser.add_argument("--input-size", type=int, default=256, help="Input resolution.")
    parser.add_argument("--device", type=str, default="cuda", help="Device identifier (e.g. 'cuda' or 'cpu').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for reproducibility.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision training.")
    parser.add_argument(
        "--label-columns",
        type=str,
        default="diagnosis,md_stage,vfi_grade,ght_class",
        help="Comma-separated list describing the label column names in the Excel file.",
    )
    parser.add_argument(
        "--task-names",
        type=str,
        default="diagnosis,md_stage,vfi_grade,ght_class",
        help="Comma-separated list of task identifiers matching --label-columns order.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_annotations(path: str) -> pd.DataFrame:
    extension = Path(path).suffix.lower()
    if extension not in {".xls", ".xlsx"}:
        raise ValueError("Annotation file must be an Excel file (.xls or .xlsx).")
    dataframe = pd.read_excel(path)
    if dataframe.empty:
        raise ValueError("Annotation file is empty.")
    return dataframe


def encode_labels(dataframe: pd.DataFrame, label_columns: TaskColumns) -> Tuple[pd.DataFrame, LabelMappings]:
    df = dataframe.copy()
    mappings: LabelMappings = {}
    for task, column in label_columns.items():
        values = df[column].dropna().astype(int)
        unique = sorted(values.unique().tolist())
        if not unique:
            raise ValueError(f"Column '{column}' does not contain any labels.")
        mapping = {int(original): idx for idx, original in enumerate(unique)}
        df[column] = df[column].map(mapping)
        if df[column].isna().any():
            raise ValueError(
                f"Column '{column}' contains values outside of the supported set {list(mapping.keys())}."
            )
        mappings[task] = {"to_index": mapping, "from_index": {idx: original for original, idx in mapping.items()}}
    return df, mappings


def split_dataframe(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= val_ratio < 1.0 or not 0.0 <= test_ratio < 1.0:
        raise ValueError("val_ratio and test_ratio must lie within [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    stratify_values = df[stratify_column] if stratify_column else None
    remaining_ratio = val_ratio + test_ratio
    if remaining_ratio == 0:
        raise ValueError("At least one of val_ratio or test_ratio must be > 0.")

    train_df, holdout_df = train_test_split(
        df,
        test_size=remaining_ratio,
        stratify=stratify_values,
        random_state=seed,
    )

    if val_ratio == 0:
        return train_df, pd.DataFrame(columns=df.columns), holdout_df
    if test_ratio == 0:
        return train_df, holdout_df, pd.DataFrame(columns=df.columns)

    val_size = val_ratio / remaining_ratio
    stratify_holdout = holdout_df[stratify_column] if stratify_column else None
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=1 - val_size,
        stratify=stratify_holdout,
        random_state=seed,
    )
    return train_df, val_df, test_df


def create_dataloaders(
    args: argparse.Namespace,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    label_columns: TaskColumns,
) -> Tuple[DataLoader, DataLoader, DataLoader, Iterable[str]]:
    train_transform = create_transform(
        input_size=args.input_size,
        is_training=True,
        color_jitter=0.3,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )

    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    resize_size = int(args.input_size / crop_pct)
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )

    dataset_train = RetFoundMultiTaskDataset(
        df_train,
        image_root=args.images_dir,
        label_columns=label_columns,
        transform=train_transform,
        image_extension=args.image_ext,
    )
    dataset_val = RetFoundMultiTaskDataset(
        df_val,
        image_root=args.images_dir,
        label_columns=label_columns,
        transform=eval_transform,
        image_extension=args.image_ext,
    )
    dataset_test = RetFoundMultiTaskDataset(
        df_test,
        image_root=args.images_dir,
        label_columns=label_columns,
        transform=eval_transform,
        image_extension=args.image_ext,
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader_train, loader_val, loader_test, dataset_train.task_names


class MultiTaskRetFound(nn.Module):
    """RETFound backbone with independent classification heads for each task."""

    def __init__(self, backbone: nn.Module, task_classes: Mapping[str, int], dropout: float = 0.0) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        hidden_dim = getattr(backbone, "num_features", None)
        if hidden_dim is None and hasattr(backbone, "head") and hasattr(backbone.head, "in_features"):
            hidden_dim = backbone.head.in_features
        if hidden_dim is None:
            raise ValueError("Unable to infer the embedding dimension from the backbone.")

        self.heads = nn.ModuleDict({task: nn.Linear(hidden_dim, num_classes) for task, num_classes in task_classes.items()})
        for head in self.heads.values():
            trunc_normal_(head.weight, std=2e-5)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone.forward_features(x) if hasattr(self.backbone, "forward_features") else self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 3:
            features = features.mean(dim=1)
        features = self.dropout(features)
        return {task: head(features) for task, head in self.heads.items()}


class MultiTaskLoss(nn.Module):
    """Sum of cross-entropy losses, optionally weighted per task."""

    def __init__(self, task_weights: Mapping[str, float] | None = None) -> None:
        super().__init__()
        self.task_weights = dict(task_weights) if task_weights is not None else {}
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses: Dict[str, torch.Tensor] = {}
        total = torch.zeros(1, device=next(iter(outputs.values())).device)
        for task, logits in outputs.items():
            loss = self.criterion(logits, targets[task])
            losses[task] = loss.detach()
            weight = self.task_weights.get(task, 1.0)
            total = total + weight * loss
        return total.squeeze(0), losses


def build_model(args: argparse.Namespace, task_classes: Mapping[str, int]) -> nn.Module:
    if args.model != "RETFound_mae":
        raise ValueError("Currently only the RETFound_mae backbone is supported for multi-task fine-tuning.")

    backbone = models.RETFound_mae(
        img_size=args.input_size,
        num_classes=0,
        drop_path_rate=args.drop_path,
        global_pool=True,
    )

    if args.finetune:
        checkpoint_path = args.finetune
        if Path(checkpoint_path).is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_path}' was not found. Provide a local path to the pre-trained RETFound weights."
            )

        checkpoint_model = checkpoint.get("model", checkpoint)
        state_dict = backbone.state_dict()
        for key in ["head.weight", "head.bias"]:
            if key in checkpoint_model and checkpoint_model[key].shape != state_dict.get(key, torch.empty(0)).shape:
                del checkpoint_model[key]
        interpolate_pos_embed(backbone, checkpoint_model)
        missing, unexpected = backbone.load_state_dict(checkpoint_model, strict=False)
        print(f"Loaded pre-trained weights with missing keys: {missing}, unexpected keys: {unexpected}")

    model = MultiTaskRetFound(backbone, task_classes)
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    return model


def train_one_epoch(
    model: nn.Module,
    criterion: MultiTaskLoss,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    sample_count = 0
    correct: Dict[str, int] = {task: 0 for task in model.heads.keys()}
    total: Dict[str, int] = {task: 0 for task in model.heads.keys()}

    for images, targets, _ in data_loader:
        images = images.to(device, non_blocking=True)
        targets = {task: label.to(device, non_blocking=True) for task, label in targets.items()}

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss, _ = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

        for task, logits in outputs.items():
            preds = torch.argmax(logits, dim=1)
            correct[task] += (preds == targets[task]).sum().item()
            total[task] += batch_size

    metrics = {"loss": running_loss / max(sample_count, 1)}
    for task in correct:
        metrics[f"{task}_acc"] = correct[task] / max(total[task], 1)
    metrics["epoch"] = epoch
    return metrics


def evaluate(
    model: nn.Module,
    criterion: MultiTaskLoss,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    return_predictions: bool = False,
) -> Dict[str, object]:
    model.eval()
    running_loss = 0.0
    sample_count = 0
    targets_all: Dict[str, list] = {task: [] for task in model.heads.keys()}
    preds_all: Dict[str, list] = {task: [] for task in model.heads.keys()}
    image_ids: list = []

    with torch.no_grad():
        for images, targets, ids in data_loader:
            images = images.to(device, non_blocking=True)
            batch_size = images.size(0)
            targets_gpu = {task: label.to(device, non_blocking=True) for task, label in targets.items()}

            with autocast(enabled=use_amp):
                outputs = model(images)
                loss, _ = criterion(outputs, targets_gpu)

            running_loss += loss.item() * batch_size
            sample_count += batch_size

            for task, logits in outputs.items():
                preds = torch.argmax(logits, dim=1)
                preds_all[task].extend(preds.cpu().tolist())
                targets_all[task].extend(targets_gpu[task].cpu().tolist())
            if return_predictions:
                image_ids.extend(ids)

    metrics: Dict[str, object] = {
        "loss": running_loss / max(sample_count, 1),
        "tasks": {},
    }

    accuracies = []
    macro_f1_scores = []
    for task in model.heads.keys():
        task_targets = np.array(targets_all[task])
        task_preds = np.array(preds_all[task])
        acc = accuracy_score(task_targets, task_preds) if len(task_targets) else math.nan
        f1 = f1_score(task_targets, task_preds, average="macro") if len(task_targets) else math.nan
        metrics["tasks"][task] = {"accuracy": float(acc), "macro_f1": float(f1)}
        if not math.isnan(acc):
            accuracies.append(acc)
        if not math.isnan(f1):
            macro_f1_scores.append(f1)

    metrics["mean_accuracy"] = float(np.mean(accuracies)) if accuracies else math.nan
    metrics["mean_macro_f1"] = float(np.mean(macro_f1_scores)) if macro_f1_scores else math.nan

    if return_predictions:
        metrics["predictions"] = {
            "image_id": image_ids,
            **{f"{task}_target": targets_all[task] for task in model.heads.keys()},
            **{f"{task}_pred": preds_all[task] for task in model.heads.keys()},
        }

    return metrics


def save_json(data: Dict[str, object], path: Path) -> None:
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, float):
            return None if math.isnan(obj) else obj
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with path.open("w", encoding="utf-8") as handle:
        json.dump(convert(data), handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    use_amp = not args.no_amp and device.type == "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_names_list = [name.strip() for name in args.task_names.split(",") if name.strip()]
    label_cols = [col.strip() for col in args.label_columns.split(",") if col.strip()]
    if not task_names_list or not label_cols:
        raise ValueError("No label columns were provided.")
    if len(task_names_list) != len(label_cols):
        raise ValueError("--task-names and --label-columns must have the same number of entries.")
    label_columns = dict(zip(task_names_list, label_cols))

    df_raw = read_annotations(args.annotations)
    df_encoded, label_mappings = encode_labels(df_raw, label_columns)
    df_encoded = df_encoded.dropna(subset=["image_id"])

    df_train, df_val, df_test = split_dataframe(
        df_encoded,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_column=next(iter(label_columns.values())),
    )

    loaders = create_dataloaders(args, df_train, df_val, df_test, label_columns)
    train_loader, val_loader, test_loader, dataset_task_names = loaders

    task_classes = {task: int(df_encoded[column].max() + 1) for task, column in label_columns.items()}
    model = build_model(args, task_classes).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    criterion = MultiTaskLoss()

    history = []
    best_metric = -float("inf")
    best_checkpoint = output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, criterion, train_loader, optimizer, scaler, device, epoch, use_amp)
        val_stats = evaluate(model, criterion, val_loader, device, use_amp)
        scheduler.step()

        summary = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "val_mean_accuracy": val_stats.get("mean_accuracy", float("nan")),
            "val_mean_macro_f1": val_stats.get("mean_macro_f1", float("nan")),
        }
        for task in dataset_task_names:
            summary[f"train_{task}_acc"] = train_stats.get(f"{task}_acc", float("nan"))
            task_metrics = val_stats["tasks"].get(task, {})
            summary[f"val_{task}_acc"] = task_metrics.get("accuracy", float("nan"))
            summary[f"val_{task}_macro_f1"] = task_metrics.get("macro_f1", float("nan"))

        history.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
        save_json({"history": history}, output_dir / "training_history.json")

        current_metric = val_stats.get("mean_accuracy", float("nan"))
        if not math.isnan(current_metric) and current_metric > best_metric:
            best_metric = current_metric
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "label_mappings": label_mappings,
                    "args": vars(args),
                },
                best_checkpoint,
            )

    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded best checkpoint from epoch {checkpoint.get('epoch', 'N/A')} for testing.")

    test_stats = evaluate(model, criterion, test_loader, device, use_amp, return_predictions=True)
    save_json(test_stats, output_dir / "test_metrics.json")

    predictions = test_stats.pop("predictions", {})
    if predictions:
        df_predictions = pd.DataFrame(predictions)
        df_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
        print(f"Saved test predictions to {output_dir / 'test_predictions.csv'}")

    print("Test metrics:")
    print(json.dumps(test_stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

