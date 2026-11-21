"""Command-line entry point for leakage-safe HCA-GCN experiments."""

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

from dataloader import init_dataloader
from model.model import HCAGCN
from train import BasicTrain


PROJECT_ROOT = Path(__file__).resolve().parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(requested):
    if requested and requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(model, config):
    name = config.get("optimizer", "adam").lower()
    kwargs = {
        "lr": float(config["lr"]),
        "weight_decay": float(config.get("weight_decay", 0.0)),
    }
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError("optimizer must be 'adam' or 'adamw'.")


def save_oof_predictions(path, fold_results):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["subject_id", "fold", "y_true", "probability", "prediction"]
        )
        for fold_index, result in enumerate(fold_results):
            for subject_id, y_true, probability, prediction in zip(
                result["subject_ids"],
                result["y_true"],
                result["y_prob"],
                result["y_pred"],
            ):
                writer.writerow(
                    [
                        subject_id,
                        fold_index,
                        int(y_true),
                        float(probability),
                        int(prediction),
                    ]
                )


def run_experiment(config, repeat_index=0, device_override=None):
    base_seed = int(config.get("seed", 123))
    seed = base_seed + repeat_index
    set_seed(seed)

    config["data"]["project_root"] = str(PROJECT_ROOT)
    loaders, roi_num, time_points = init_dataloader(
        config["data"], seed=seed
    )
    train_loaders, val_loaders, test_loaders = loaders

    if roi_num != int(config["model"]["roi_num"]):
        raise ValueError("Loaded ROI count does not match model.roi_num.")
    if time_points != int(config["model"]["time_series"]):
        raise ValueError(
            "Loaded time-point count does not match model.time_series."
        )

    device = choose_device(device_override or config.get("device", "auto"))
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = PROJECT_ROOT / config["train"].get(
        "output_root", "outputs"
    )
    run_dir = (
        output_root
        / config["model"].get("type", "HCA_GCN")
        / (
            f"{config['data']['dataset']}_{config['data']['atlas']}_"
            f"{config['model'].get('variant', 'full')}_"
            f"repeat{repeat_index}_{timestamp}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    fold_results = []
    for fold_index, (train_loader, val_loader, test_loader) in enumerate(
        zip(train_loaders, val_loaders, test_loaders)
    ):
        # A fresh model and optimizer are created for every held-out fold.
        model = HCAGCN(
            config["model"],
            roi_num=roi_num,
            node_feature_dim=int(config["model"]["node_feature_dim"]),
            time_series=time_points,
        ).to(device)
        optimizer = build_optimizer(model, config["train"])
        trainer = BasicTrain(
            config["train"],
            model,
            (optimizer,),
            (train_loader, val_loader, test_loader),
            run_dir / f"fold_{fold_index}",
            device,
            fold_index=fold_index,
        )
        metrics, test_result = trainer.train()
        fold_metrics.append(metrics)
        fold_results.append(test_result)

    metric_names = ("accuracy", "auc", "sensitivity", "specificity")
    summary = {
        name: {
            "mean": float(np.mean([fold[name] for fold in fold_metrics])),
            "std": float(np.std([fold[name] for fold in fold_metrics], ddof=1)),
        }
        for name in metric_names
    }
    summary.update(
        {
            "checkpoint_criterion": "minimum validation loss",
            "decision_threshold": float(
                config["train"].get("decision_threshold", 0.5)
            ),
            "fold_metrics": fold_metrics,
        }
    )
    with (run_dir / "cross_validation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    save_oof_predictions(run_dir / "oof_predictions.csv", fold_results)

    print("Five-fold summary:")
    for name in metric_names:
        print(
            f"  {name}: {summary[name]['mean']:.4f} "
            f"+/- {summary[name]['std']:.4f}"
        )
    print(f"Results saved to {run_dir}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate HCA-GCN using subject-level CV."
    )
    parser.add_argument(
        "--config",
        default="setting/abide_HCAGCN.yaml",
        help="YAML configuration path (relative paths use the project root).",
    )
    parser.add_argument("--repeat-time", type=int, default=1)
    parser.add_argument(
        "--device",
        default=None,
        help="Optional override such as cpu, cuda, or cuda:0.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for repeat_index in range(args.repeat_time):
        run_experiment(config, repeat_index, args.device)


if __name__ == "__main__":
    main()
