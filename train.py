"""Training and held-out evaluation for HCA-GCN.

A checkpoint is selected using validation loss only.  The held-out test fold is
evaluated exactly once with a fixed decision threshold.
"""

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


class BasicTrain:
    """Train one cross-validation fold without consulting the test fold."""

    def __init__(
        self,
        train_config,
        model,
        optimizers,
        dataloaders,
        log_folder,
        device,
        fold_index=0,
    ):
        self.config = train_config
        self.model = model.to(device)
        self.optimizer = optimizers[0]
        self.train_loader, self.val_loader, self.test_loader = dataloaders
        self.output_dir = Path(log_folder)
        self.device = device
        self.fold_index = fold_index

        self.epochs = int(train_config["epochs"])
        self.threshold = float(train_config.get("decision_threshold", 0.5))
        if self.threshold != 0.5:
            raise ValueError(
                "The manuscript protocol uses a fixed 0.5 decision threshold."
            )
        self.patience = int(train_config.get("early_stopping_patience", 0))
        self.min_delta = float(train_config.get("min_delta", 0.0))
        self.sparsity_weight = float(
            train_config.get("sparsity_loss_weight", 0.0)
        )
        self.use_sparsity = bool(train_config.get("sparsity_loss", True))
        self.accumulation_steps = int(
            train_config.get("gradient_accumulation_steps", 1)
        )

        weights = train_config.get("class_weights")
        weight_tensor = (
            torch.as_tensor(weights, dtype=torch.float32, device=device)
            if weights is not None
            else None
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    @staticmethod
    def _unpack(batch):
        if len(batch) != 7:
            raise ValueError(
                "Expected five views, label, and subject_id in each batch."
            )
        return batch

    def _forward(self, batch):
        fc, ngc, granger, pearson, spearman, labels, subject_ids = self._unpack(
            batch
        )
        views = [
            fc.to(self.device),
            ngc.to(self.device),
            granger.to(self.device),
            pearson.to(self.device),
            spearman.to(self.device),
        ]
        labels = labels.long().to(self.device)
        outputs = self.model(*views)
        logits, graph_fc, graph_ec = outputs[:3]
        return logits, graph_fc, graph_ec, labels, list(subject_ids)

    def _train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        y_true, y_pred = [], []

        for step, batch in enumerate(self.train_loader, start=1):
            logits, graph_fc, graph_ec, labels, _ = self._forward(batch)
            loss = self.loss_fn(logits, labels)
            if self.use_sparsity and self.sparsity_weight > 0:
                loss = loss + self.sparsity_weight * (
                    graph_fc.abs().mean() + graph_ec.abs().mean()
                )

            (loss / self.accumulation_steps).backward()
            if (
                step % self.accumulation_steps == 0
                or step == len(self.train_loader)
            ):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * labels.numel()
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

        return {
            "loss": running_loss / max(len(y_true), 1),
            "accuracy": accuracy_score(y_true, y_pred),
        }

    def _evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        y_true, y_prob, subject_ids = [], [], []

        with torch.no_grad():
            for batch in loader:
                logits, _, _, labels, batch_ids = self._forward(batch)
                running_loss += self.loss_fn(logits, labels).item() * labels.numel()
                y_true.extend(labels.cpu().tolist())
                y_prob.extend(F.softmax(logits, dim=1)[:, 1].cpu().tolist())
                subject_ids.extend(batch_ids)

        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        y_pred = (y_prob >= self.threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        auc = (
            roc_auc_score(y_true, y_prob)
            if np.unique(y_true).size == 2
            else float("nan")
        )

        return {
            "loss": running_loss / max(y_true.size, 1),
            "accuracy": accuracy_score(y_true, y_pred),
            "auc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "subject_ids": np.asarray(subject_ids, dtype=str),
        }

    @staticmethod
    def _public_metrics(result):
        excluded = {"y_true", "y_prob", "y_pred", "subject_ids"}
        return {
            key: (float(value) if isinstance(value, np.floating) else value)
            for key, value in result.items()
            if key not in excluded
        }

    def _save_predictions(self, result):
        with (self.output_dir / "test_predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["subject_id", "fold", "y_true", "probability", "prediction"]
            )
            for subject_id, y_true, probability, prediction in zip(
                result["subject_ids"],
                result["y_true"],
                result["y_prob"],
                result["y_pred"],
            ):
                writer.writerow(
                    [
                        subject_id,
                        self.fold_index,
                        int(y_true),
                        float(probability),
                        int(prediction),
                    ]
                )

    def _save_learnable_graphs(self):
        matrices_fc, matrices_ec, labels, subject_ids = [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                _, graph_fc, graph_ec, batch_labels, batch_ids = self._forward(
                    batch
                )
                matrices_fc.append(graph_fc.cpu().numpy())
                matrices_ec.append(graph_ec.cpu().numpy())
                labels.extend(batch_labels.cpu().tolist())
                subject_ids.extend(batch_ids)

        np.savez_compressed(
            self.output_dir / "held_out_graphs.npz",
            graph_fc=np.concatenate(matrices_fc, axis=0),
            graph_ec=np.concatenate(matrices_ec, axis=0),
            labels=np.asarray(labels, dtype=int),
            subject_ids=np.asarray(subject_ids, dtype=str),
        )

    def train(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        best_state = None
        best_epoch = -1
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, self.epochs + 1):
            train_result = self._train_epoch()
            val_result = self._evaluate(self.val_loader)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_result["loss"],
                    "train_accuracy": train_result["accuracy"],
                    "val_loss": val_result["loss"],
                    "val_accuracy": val_result["accuracy"],
                    "val_auc": val_result["auc"],
                }
            )

            improved = val_result["loss"] < best_val_loss - self.min_delta
            if improved:
                best_val_loss = val_result["loss"]
                best_epoch = epoch
                best_state = deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"Fold {self.fold_index} | Epoch {epoch:04d} | "
                f"train loss {train_result['loss']:.6f} | "
                f"val loss {val_result['loss']:.6f} | "
                f"val AUC {val_result['auc']:.4f}"
            )
            if self.patience and epochs_without_improvement >= self.patience:
                break

        if best_state is None:
            raise RuntimeError("No validation checkpoint was selected.")

        self.model.load_state_dict(best_state)
        checkpoint_path = self.output_dir / "best_by_validation_loss.pt"
        torch.save(
            {
                "model_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_validation_loss": best_val_loss,
                "decision_threshold": self.threshold,
            },
            checkpoint_path,
        )

        # The test fold is first accessed here, after checkpoint selection.
        test_result = self._evaluate(self.test_loader)
        metrics = self._public_metrics(test_result)
        metrics.update(
            {
                "fold": self.fold_index,
                "best_epoch": best_epoch,
                "checkpoint_criterion": "minimum validation loss",
                "decision_threshold": self.threshold,
                "number_of_test_subjects": int(test_result["y_true"].size),
            }
        )

        with (self.output_dir / "metrics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(metrics, handle, indent=2, allow_nan=False)
        with (self.output_dir / "history.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(history, handle, indent=2, allow_nan=False)

        self._save_predictions(test_result)
        if self.config.get("save_held_out_graphs", True):
            self._save_learnable_graphs()

        print(
            f"Fold {self.fold_index} held-out result | "
            f"ACC {metrics['accuracy']:.4f} | AUC {metrics['auc']:.4f} | "
            f"SEN {metrics['sensitivity']:.4f} | "
            f"SPE {metrics['specificity']:.4f} | "
            f"TP/FN/TN/FP {metrics['tp']}/{metrics['fn']}/"
            f"{metrics['tn']}/{metrics['fp']}"
        )
        return metrics, test_result
