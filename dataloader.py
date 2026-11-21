"""Leakage-safe loading and subject-level cross-validation splits."""

import json
import os
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader, Dataset, Subset


MODALITIES = ("ts", "ngc", "granger", "pearson", "spearman")


class StandardScaler:
    """Scalar z-score fitted only on the training partition."""

    def __init__(self, mean, std, eps=1e-8):
        self.mean = float(mean)
        self.std = max(float(std), eps)

    @classmethod
    def fit(cls, data):
        return cls(np.mean(data), np.std(data))

    def transform(self, data):
        return (data - self.mean) / self.std


class IdentityScaler:
    def transform(self, data):
        return data


class MultiViewDataset(Dataset):
    def __init__(self, arrays, labels, subject_ids, scalers):
        self.arrays = arrays
        self.labels = labels
        self.subject_ids = subject_ids
        self.scalers = scalers

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        views = [
            torch.as_tensor(
                self.scalers[name].transform(self.arrays[name][index]),
                dtype=torch.float32,
            )
            for name in MODALITIES
        ]
        return (
            *views,
            torch.tensor(self.labels[index], dtype=torch.long),
            str(self.subject_ids[index]),
        )


def _resolve_path(data_root, value):
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    if "$" in expanded:
        raise ValueError(
            f"Unresolved environment variable in path {value!r}. "
            "Set HCA_GCN_DATA_ROOT or edit the local configuration."
        )
    path = Path(expanded)
    if not path.is_absolute():
        path = data_root / path
    return path.resolve()


def _collect_files(directory, id_pattern, modality):
    if not directory.is_dir():
        raise FileNotFoundError(f"{modality} directory does not exist: {directory}")
    records = {}
    for path in sorted(directory.glob("*.mat")):
        match = id_pattern.search(path.stem)
        if match is None:
            continue
        subject_id = match.group(1)
        if subject_id in records:
            raise ValueError(
                f"Duplicate {modality} files for subject {subject_id}: "
                f"{records[subject_id]} and {path}"
            )
        records[subject_id] = path
    if not records:
        raise FileNotFoundError(
            f"No MAT files with matching subject IDs found in {directory}"
        )
    return records


def _read_mat(path, key):
    contents = loadmat(path)
    if key not in contents:
        raise KeyError(f"Key {key!r} is missing from {path}")
    return np.asarray(contents[key], dtype=np.float32)


def _load_subject_group(label, directories, keys, id_pattern, time_points, roi_num):
    file_maps = {
        name: _collect_files(directories[name], id_pattern, f"{name}/label={label}")
        for name in MODALITIES
    }
    common_ids = set.intersection(*(set(mapping) for mapping in file_maps.values()))
    if not common_ids:
        raise ValueError(f"No aligned subjects were found for label {label}.")

    rows = []
    for subject_id in sorted(common_ids):
        ts = _read_mat(file_maps["ts"][subject_id], keys["ts"])
        if ts.shape == (time_points, roi_num):
            ts = ts.T
        elif ts.shape != (roi_num, time_points):
            raise ValueError(
                f"Time-series shape for {subject_id} is {ts.shape}; expected "
                f"({time_points}, {roi_num}) or ({roi_num}, {time_points})."
            )

        matrices = {}
        for name in MODALITIES[1:]:
            matrix = _read_mat(file_maps[name][subject_id], keys[name])
            if matrix.shape != (roi_num, roi_num):
                raise ValueError(
                    f"{name} shape for {subject_id} is {matrix.shape}; expected "
                    f"({roi_num}, {roi_num})."
                )
            matrices[name] = matrix

        rows.append(
            {
                "subject_id": subject_id,
                "label": label,
                "ts": ts,
                **matrices,
            }
        )
    return rows


def _validate_fold(fold, subject_ids, fold_index):
    sets = {
        name: set(np.asarray(subject_ids)[indices].tolist())
        for name, indices in fold.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise ValueError(
                f"Fold {fold_index}: {left}/{right} subject overlap: "
                f"{sorted(overlap)}"
            )


def _counts_for_fold_sizes(total_positive, fold_sizes):
    total = int(sum(fold_sizes))
    ideal = np.asarray(fold_sizes, dtype=float) * total_positive / total
    counts = np.floor(ideal).astype(int)
    remainder = total_positive - int(counts.sum())
    order = np.argsort(-(ideal - counts))
    counts[order[:remainder]] += 1
    return counts.tolist()


def _generate_folds(labels, fold_sizes, validation_size, seed):
    labels = np.asarray(labels, dtype=int)
    positive = np.where(labels == 1)[0]
    negative = np.where(labels == 0)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)

    if fold_sizes is None:
        base, remainder = divmod(labels.size, 5)
        fold_sizes = [base + (index < remainder) for index in range(5)]
    fold_sizes = [int(value) for value in fold_sizes]
    if sum(fold_sizes) != labels.size:
        raise ValueError(
            f"test_fold_sizes sum to {sum(fold_sizes)}, but the cohort has "
            f"{labels.size} subjects."
        )

    positive_sizes = _counts_for_fold_sizes(len(positive), fold_sizes)
    negative_sizes = [
        size - positive_count
        for size, positive_count in zip(fold_sizes, positive_sizes)
    ]
    if sum(negative_sizes) != len(negative) or min(negative_sizes) < 0:
        raise ValueError("Unable to stratify the requested test fold sizes.")

    folds = []
    positive_start = negative_start = 0
    for fold_index, (test_size, pos_size, neg_size) in enumerate(
        zip(fold_sizes, positive_sizes, negative_sizes)
    ):
        test_indices = np.concatenate(
            [
                positive[positive_start : positive_start + pos_size],
                negative[negative_start : negative_start + neg_size],
            ]
        )
        positive_start += pos_size
        negative_start += neg_size

        remaining = np.setdiff1d(
            np.arange(labels.size), test_indices, assume_unique=False
        )
        remaining_positive = remaining[labels[remaining] == 1].copy()
        remaining_negative = remaining[labels[remaining] == 0].copy()
        fold_rng = np.random.default_rng(seed + fold_index + 1)
        fold_rng.shuffle(remaining_positive)
        fold_rng.shuffle(remaining_negative)

        val_sizes = _counts_for_fold_sizes(
            len(remaining_positive), [validation_size, len(remaining) - validation_size]
        )
        val_positive = val_sizes[0]
        val_negative = validation_size - val_positive
        val_indices = np.concatenate(
            [
                remaining_positive[:val_positive],
                remaining_negative[:val_negative],
            ]
        )
        train_indices = np.setdiff1d(
            remaining, val_indices, assume_unique=False
        )

        fold = {
            "train": np.sort(train_indices),
            "val": np.sort(val_indices),
            "test": np.sort(test_indices),
        }
        if len(fold["test"]) != test_size:
            raise RuntimeError("Internal error while constructing test folds.")
        folds.append(fold)

    all_test = np.concatenate([fold["test"] for fold in folds])
    if len(np.unique(all_test)) != labels.size:
        raise ValueError("Each subject must occur in exactly one held-out test fold.")
    return folds


def _load_folds(path, subject_ids):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    id_to_index = {
        str(subject_id): index for index, subject_id in enumerate(subject_ids)
    }
    folds = []
    for fold_index in range(len(payload)):
        record = payload[f"fold_{fold_index}"]
        fold = {}
        for split in ("train", "val", "test"):
            missing = [
                subject_id
                for subject_id in record[f"{split}_ids"]
                if str(subject_id) not in id_to_index
            ]
            if missing:
                raise ValueError(
                    f"Fold {fold_index} contains unknown {split} IDs: {missing}"
                )
            fold[split] = np.asarray(
                [id_to_index[str(value)] for value in record[f"{split}_ids"]],
                dtype=int,
            )
        folds.append(fold)
    return folds


def _save_folds(path, folds, subject_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    ids = np.asarray(subject_ids)
    for fold_index, fold in enumerate(folds):
        payload[f"fold_{fold_index}"] = {
            f"{split}_ids": ids[fold[split]].tolist()
            for split in ("train", "val", "test")
        }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def init_dataloader(dataset_config, seed=123):
    """Return leakage-safe train/validation/test loaders for all folds."""

    roi_num = int(dataset_config["node_size"])
    time_points = int(dataset_config["time_points"])
    data_root = Path(
        os.path.expanduser(
            os.path.expandvars(dataset_config.get("data_root", "."))
        )
    )
    if "$" in str(data_root):
        raise ValueError(
            "Set the HCA_GCN_DATA_ROOT environment variable before training."
        )
    data_root = data_root.resolve()

    path_keys = {
        "ts": "ts",
        "ngc": "ngc",
        "granger": "granger",
        "pearson": "pearson",
        "spearman": "spearman",
    }
    group_directories = {}
    for group in ("control", "patient"):
        group_directories[group] = {
            name: _resolve_path(
                data_root, dataset_config[f"{prefix}_{group}_path"]
            )
            for name, prefix in path_keys.items()
        }

    keys = {
        "ts": dataset_config.get("ts_key", "ROISignals"),
        "ngc": dataset_config.get("ngc_key", "GC_est"),
        "granger": dataset_config.get("granger_key", "Granger"),
        "pearson": dataset_config.get("pearson_key", "Pearson"),
        "spearman": dataset_config.get("spearman_key", "Spearman"),
    }
    id_pattern = re.compile(
        dataset_config.get("subject_id_regex", r"(\d{7})")
    )

    rows = _load_subject_group(
        0,
        group_directories["control"],
        keys,
        id_pattern,
        time_points,
        roi_num,
    )
    rows.extend(
        _load_subject_group(
            1,
            group_directories["patient"],
            keys,
            id_pattern,
            time_points,
            roi_num,
        )
    )

    subject_ids = np.asarray([row["subject_id"] for row in rows], dtype=str)
    if np.unique(subject_ids).size != subject_ids.size:
        raise ValueError("Subject IDs must be globally unique across classes.")
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    arrays = {
        name: np.stack([row[name] for row in rows]).astype(np.float32)
        for name in MODALITIES
    }

    split_file = dataset_config.get("split_file")
    if split_file:
        split_path = _resolve_path(Path(dataset_config["project_root"]), split_file)
        if not split_path.is_file():
            raise FileNotFoundError(f"Configured split file not found: {split_path}")
        folds = _load_folds(split_path, subject_ids)
    else:
        folds = _generate_folds(
            labels,
            dataset_config.get("test_fold_sizes"),
            int(dataset_config.get("validation_size", 15)),
            int(seed),
        )
        output_path = dataset_config.get("split_output_path")
        if output_path:
            _save_folds(
                _resolve_path(Path(dataset_config["project_root"]), output_path),
                folds,
                subject_ids,
            )

    expected_splits = int(dataset_config.get("n_splits", 5))
    if len(folds) != expected_splits:
        raise ValueError(
            f"Expected {expected_splits} folds, received {len(folds)}."
        )

    all_test_indices = np.concatenate([fold["test"] for fold in folds])
    if (
        all_test_indices.size != labels.size
        or np.unique(all_test_indices).size != labels.size
    ):
        raise ValueError(
            "Across folds, every subject must appear exactly once in a test set."
        )
    configured_sizes = dataset_config.get("test_fold_sizes")
    if configured_sizes is not None:
        observed_sizes = [len(fold["test"]) for fold in folds]
        if observed_sizes != [int(value) for value in configured_sizes]:
            raise ValueError(
                f"Test fold sizes {observed_sizes} do not match the configured "
                f"sizes {configured_sizes}."
            )
    expected_indices = set(range(labels.size))
    for fold_index, fold in enumerate(folds):
        union = set(np.concatenate([fold["train"], fold["val"], fold["test"]]))
        if union != expected_indices:
            raise ValueError(
                f"Fold {fold_index} does not partition the complete cohort."
            )

    all_train, all_val, all_test = [], [], []
    normalization = dataset_config.get("normalization", "train_zscore")
    num_workers = int(dataset_config.get("num_workers", 0))
    pin_memory = bool(dataset_config.get("pin_memory", False))

    for fold_index, fold in enumerate(folds):
        _validate_fold(fold, subject_ids, fold_index)
        train_indices = fold["train"]
        if normalization == "train_zscore":
            scalers = {
                name: StandardScaler.fit(arrays[name][train_indices])
                for name in MODALITIES
            }
        elif normalization == "none":
            scalers = {name: IdentityScaler() for name in MODALITIES}
        else:
            raise ValueError(
                "normalization must be 'train_zscore' or 'none'."
            )

        dataset = MultiViewDataset(arrays, labels, subject_ids, scalers)
        generator = torch.Generator().manual_seed(seed + fold_index)
        common = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": False,
        }
        all_train.append(
            DataLoader(
                Subset(dataset, train_indices),
                batch_size=int(dataset_config["batch_size"]),
                shuffle=True,
                generator=generator,
                **common,
            )
        )
        all_val.append(
            DataLoader(
                Subset(dataset, fold["val"]),
                batch_size=int(dataset_config["val_batch_size"]),
                shuffle=False,
                **common,
            )
        )
        all_test.append(
            DataLoader(
                Subset(dataset, fold["test"]),
                batch_size=int(dataset_config["test_batch_size"]),
                shuffle=False,
                **common,
            )
        )

        print(
            f"Fold {fold_index}: train={len(fold['train'])}, "
            f"val={len(fold['val'])}, test={len(fold['test'])}; "
            f"test labels={np.bincount(labels[fold['test']], minlength=2).tolist()}"
        )

    return (all_train, all_val, all_test), roi_num, time_points
