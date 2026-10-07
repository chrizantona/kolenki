"""Small fixed-epoch heads; Gold is only evaluated after all gradient updates."""
import json
import hashlib
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .data import LABELS, ids_hash
from .encoder import file_sha
from .heads import StudyHead


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch(value, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_features(ids, directory, identity):
    arrays = {"features": [], "mask": [], "positions": []}
    for uid in ids:
        with np.load(Path(directory) / f"{uid}.npz", allow_pickle=False) as row:
            if row["study"].item() != uid or row["identity"].item() != identity:
                raise RuntimeError("Feature UID or extraction identity mismatch")
            f, m, p = row["features"], row["mask"], row["positions"]
            if f.ndim != 3 or f.shape[0] != 6 or f.shape[-1] != 1024 or m.shape != (6,) or p.shape != f.shape[:2]:
                raise RuntimeError(f"Malformed feature geometry: {uid}")
            if m.dtype != np.bool_ or not m.any() or not np.isfinite(f).all() or not np.isfinite(p).all():
                raise RuntimeError(f"Invalid feature values: {uid}")
            if not np.all(f[~m] == 0) or not ((p >= 0) & (p <= 1)).all():
                raise RuntimeError(f"Invalid absent-slot features / positions: {uid}")
            arrays["features"].append(torch.from_numpy(f.astype(np.float32)))
            arrays["mask"].append(torch.from_numpy(m.copy()))
            arrays["positions"].append(torch.from_numpy(p.copy()))
    return tuple(torch.stack(arrays[key]) for key in ("features", "mask", "positions"))


def score_auc(targets, predictions, target_kind):
    if not np.isfinite(predictions).all() or predictions.shape != targets.shape:
        raise RuntimeError("Invalid prediction array")
    per = [float(roc_auc_score(targets[:, i], predictions[:, i])) if np.unique(targets[:, i]).size == 2 else None
           for i in range(len(LABELS))]
    valid = [value for value in per if value is not None]
    return {"n": len(targets), "macro_auc": float(np.mean(valid)) if valid else None,
            "per_class_auc": dict(zip(LABELS, per)), "target_kind": target_kind,
            "classes_with_both_labels": len(valid)}


@torch.inference_mode()
def predict(head, tensors, device, batch_size):
    head.eval()
    outputs = []
    for start in range(0, len(tensors[0]), batch_size):
        batch = [value[start:start + batch_size].to(device) for value in tensors]
        result = head(*batch).sigmoid()
        if not torch.isfinite(result).all():
            raise RuntimeError("Non-finite head predictions")
        outputs.append(result.cpu())
    return torch.cat(outputs).numpy()


def fit(ids, labels, feature_dir, identity, config, device, path, epochs=None, max_seconds=1200):
    seed_all(config["seed"])
    if any(labels.loc[ids, "is_gold"]):
        raise RuntimeError("Gold studies cannot enter gradient training")
    tensors = load_features(ids, feature_dir, identity)
    targets = torch.from_numpy(labels.loc[ids, LABELS].to_numpy(dtype=np.float32, copy=True))
    head = StudyHead(dim=config["head_dim"], kind=config["head_kind"], dropout=config["head_dropout"]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["head_lr"], weight_decay=config["head_weight_decay"])
    epochs = epochs or config["head_epochs"]
    steps_per_epoch = math.ceil(len(ids) / config["head_batch_size"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
    history, updates, start = [], 0, time.monotonic()
    for epoch in range(epochs):
        head.train()
        order = torch.randperm(len(ids))
        loss_sum = 0.0
        for selected in order.split(config["head_batch_size"]):
            if time.monotonic() - start > max_seconds:
                raise RuntimeError("Head budget exhausted before fixed epochs completed")
            batch = [value[selected].to(device) for value in tensors]
            target = targets[selected].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(*batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite head training loss")
            loss.backward()
            for parameter in head.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("Non-finite head gradient")
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
            optimizer.step()
            scheduler.step()
            updates += 1
            loss_sum += float(loss.detach()) * len(selected)
        history.append({"epoch": epoch + 1, "loss": loss_sum / len(ids), "updates": updates})
        print(json.dumps({"stage": "head", "output": Path(path).name, **history[-1]}), flush=True)
    report = {"epochs": epochs, "optimizer_updates": updates, "n_gradient_studies": len(ids),
              "gradient_ids_sha256": ids_hash(ids), "gold_used_for_gradients": 0,
              "fixed_final_epoch": True, "gold_checkpoint_selection": False,
              "seconds": time.monotonic() - start, "history": history,
              "head_code_sha256": hashlib.sha256(Path(__file__).with_name("heads.py").read_bytes()).hexdigest(),
              "training_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_torch({"model": {key: value.cpu() for key, value in head.state_dict().items()},
                  "config": config, "labels": LABELS, "feature_identity": identity,
                  "training_report": report}, path)
    # Hash after save: a checkpoint cannot recursively contain its own hash.
    # The external report records the actual exported file, including smoke runs.
    report["checkpoint_receipt"] = {"filename": Path(path).name,
                                    "bytes": Path(path).stat().st_size,
                                    "sha256": file_sha(path)}
    return head, report


def run_training(labels, official, feature_dir, identity, config, output, device="cuda"):
    output = Path(output)
    weak = labels.index[labels.is_gold.eq(0)].tolist()
    gold = labels.index[labels.is_gold.eq(1)].tolist()
    holdout = labels.index[labels.is_gold.eq(0) & labels.fold.eq(config["weak_holdout_fold"])].tolist()
    holdout_set = set(holdout)
    gradient = [uid for uid in weak if uid not in holdout_set]
    if not holdout or set(gradient) & holdout_set or set(weak) & set(gold):
        raise RuntimeError("Invalid gradient/holdout/Gold separation")
    training_start = time.monotonic()
    cv_head, cv_report = fit(gradient, labels, feature_dir, identity, config, device, output / "head_weak_holdout.pt", max_seconds=600)
    predictions = predict(cv_head, load_features(holdout, feature_dir, identity), device, config["head_batch_size"])
    holdout_score = score_auc((labels.loc[holdout, LABELS].to_numpy() >= .5).astype(int), predictions,
                             "thresholded_public_teacher_consensus; not expert ground truth")
    pd.DataFrame(predictions, index=pd.Index(holdout, name="StudyInstanceUID"), columns=LABELS).to_csv(output / "weak_holdout_predictions.csv")
    pd.Series(gradient, name="StudyInstanceUID").to_csv(output / "weak_holdout_gradient_ids.csv", index=False)
    pd.Series(holdout, name="StudyInstanceUID").to_csv(output / "weak_holdout_ids.csv", index=False)
    del cv_head
    production, production_report = fit(weak, labels, feature_dir, identity, config, device, output / "head.pt",
                                        max_seconds=max(1, 1200 - (time.monotonic() - training_start)))
    predictions = predict(production, load_features(gold, feature_dir, identity), device, config["head_batch_size"])
    gold_score = score_auc(official.loc[gold, LABELS].to_numpy(dtype=int), predictions, "official_expert_labels; diagnostic only")
    pd.DataFrame(predictions, index=pd.Index(gold, name="StudyInstanceUID"), columns=LABELS).to_csv(output / "gold_predictions.csv")
    pd.Series(weak, name="StudyInstanceUID").to_csv(output / "gradient_train_ids.csv", index=False)
    pd.Series(gold, name="StudyInstanceUID").to_csv(output / "excluded_gold_ids.csv", index=False)
    return {"weak_holdout_head": cv_report, "weak_holdout_metrics": holdout_score,
            "production_head": production_report, "gold_metrics": gold_score}
