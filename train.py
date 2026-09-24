"""Tres experimentos reales de fine-tuning GPU sobre variedades de Fruits-360."""

from __future__ import annotations

import json
import os
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import boto3
import torch
from botocore.exceptions import ClientError
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from fruit_classes import FRUIT_FAMILIES, choose_categories, family_of

SCRATCH = Path("/scratch")
TRIALS = (
    ("efficientnet_b0", 0.0003),
    ("mobilenet_v3_large", 0.0004),
    ("resnet18", 0.0002),
)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.68, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.22, 0.22, 0.22, 0.05)], p=0.7),
    transforms.RandomPerspective(distortion_scale=0.12, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class FruitDataset(Dataset):
    def __init__(self, root: Path, categories: list[str], families: list[str], transform, limit: int, extra_root: Path | None = None):
        self.items: list[tuple[Path, int]] = []
        self.transform = transform
        family_indexes = {family: index for index, family in enumerate(families)}
        for category in categories:
            index = family_indexes[family_of(category)]
            photos = sorted((root / category).glob("*.jpg"))
            if not photos:
                raise RuntimeError(f"Sin fotos para {category} en {root}")
            if limit and len(photos) > limit:
                # Muestras distribuidas a lo largo de la rotación, no solo primeros frames.
                picks = [round(i * (len(photos) - 1) / (limit - 1)) for i in range(limit)]
                photos = [photos[i] for i in picks]
            self.items.extend((photo, index) for photo in photos)
        if extra_root and extra_root.is_dir():
            for family, index in family_indexes.items():
                folder = extra_root / family
                if folder.is_dir():
                    self.items.extend((p, index) for p in sorted(folder.iterdir())
                                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        with Image.open(path) as photo:
            return self.transform(photo.convert("RGB")), label


def download_data(s3, bucket: str) -> tuple[Path, Path]:
    archive = SCRATCH / "fruits360.zip"
    print("Bajando dataset completo desde S3...", flush=True)
    s3.download_file(bucket, "dataset/fruits360.zip", str(archive))
    destination = SCRATCH / "data"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        root = destination.resolve()
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"Ruta inválida en el ZIP: {member.filename}")
        print(f"Descomprimiendo {len(zf.infolist()):,} archivos...", flush=True)
        zf.extractall(destination)
    archive.unlink()
    training = next((p for p in destination.rglob("Training") if p.is_dir()), None)
    if training is None or not (training.parent / "Test").is_dir():
        raise RuntimeError("No encontré las carpetas Training y Test de Fruits-360")
    return training, training.parent / "Test"


def make_model(name: str, num_classes: int, pretrained: bool = True):
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        head = nn.Linear(model.classifier[1].in_features, num_classes)
        model.classifier[1] = head
        last_block = model.features[-2:]
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        head = nn.Linear(model.classifier[3].in_features, num_classes)
        model.classifier[3] = head
        last_block = model.features[-3:]
    else:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        head = nn.Linear(model.fc.in_features, num_classes)
        model.fc = head
        last_block = model.layer4
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head.parameters():
        parameter.requires_grad = True
    return model, head, last_block


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, top3_correct, total = 0, 0, 0
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        indices = logits.topk(3, dim=1).indices.cpu()
        for prediction, label in zip(indices, labels):
            actual = int(label)
            guess = int(prediction[0])
            correct += guess == actual
            top3_correct += actual in prediction.tolist()
            total += 1
    return {
        "images": total,
        "fruit_top1": round(correct / total, 4),
        "fruit_top3": round(top3_correct / total, 4),
    }


def download_labeled_photos(s3, bucket: str, prefix: str) -> Path:
    root = SCRATCH / prefix
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            parts = key.split("/")
            if len(parts) != 3 or parts[1] not in FRUIT_FAMILIES or parts[2] in {"", ".", ".."}:
                continue
            if parts[2].lower().endswith((".jpg", ".jpeg", ".png")):
                path = root / parts[1] / parts[2]
                path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(path))
    return root


def external_test(s3, bucket: str, model, device, classes):
    root = SCRATCH / "real-world-test"
    download_labeled_photos(s3, bucket, "real-world-test")
    if not root.is_dir():
        return {"status": "not_evaluated", "reason": "Faltan fotos reales etiquetadas"}
    files = []
    for family in classes:
        folder = root / family
        if folder.is_dir():
            files.extend((p, family) for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    correct = 0
    by_family = defaultdict(lambda: {"images": 0, "correct": 0})
    model.eval()
    with torch.no_grad():
        for path, family in files:
            with Image.open(path) as photo:
                image = EVAL_TRANSFORM(photo.convert("RGB")).unsqueeze(0).to(device)
            predicted = int(model(image).argmax(dim=1).item())
            hit = classes[predicted] == family
            correct += hit
            by_family[family]["images"] += 1
            by_family[family]["correct"] += hit
    families = len({family for _, family in files})
    return {
        "status": "evaluated" if len(files) >= 100 and families >= 10 else "insufficient_sample",
        "images": len(files),
        "families": families,
        "fruit_family_top1": round(correct / len(files), 4) if files else None,
        "macro_fruit_family_top1": round(sum(row["correct"] / row["images"] for row in by_family.values()) / families, 4) if families else None,
        "by_family": dict(sorted(by_family.items())),
        "note": "Este test usa fotos externas y no participa en la selección del modelo.",
    }


def restore_progress(s3, bucket, trial, model_name, classes, model, best_path, artifact_prefix=""):
    """Retoma desde la última época completa después de una interrupción Spot."""
    key = f"{artifact_prefix}checkpoints/trial-{trial}/resume.pt"
    path = SCRATCH / "resume.pt"
    try:
        s3.download_file(bucket, key, str(path))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return 1, -1.0, [], 0.0
        raise
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["trial"] != trial or saved["model_name"] != model_name or saved["classes"] != classes:
        raise RuntimeError("El checkpoint no coincide con este ensayo o dataset")
    epoch = int(saved["epoch"])
    if not 1 <= epoch <= 4:
        raise RuntimeError("Época inválida en el checkpoint")
    model.load_state_dict(saved["current_state_dict"])
    torch.save({"model_name": model_name, "classes": classes,
                "state_dict": saved["best_state_dict"], "epoch": saved["best_epoch"]}, best_path)
    print(f"Ensayo {trial}: retomando desde la época {epoch + 1} (checkpoint en S3)", flush=True)
    return epoch + 1, float(saved["best_score"]), saved["history"], float(saved.get("elapsed_sec", 0))


def main():
    trial = int(os.environ["JOB_COMPLETION_INDEX"])
    bucket = os.environ["S3_BUCKET"]
    train_limit = int(os.getenv("TRAIN_IMAGES_PER_VARIETY", "500"))
    test_limit = int(os.getenv("TEST_IMAGES_PER_VARIETY", "100"))
    final_epoch = int(os.getenv("TRAIN_FINAL_EPOCH", "4"))
    prefix_name = os.getenv("ARTIFACT_PREFIX", "").strip("/")
    artifact_prefix = f"{prefix_name}/" if prefix_name else ""
    if train_limit < 2 or test_limit < 2 or not 1 <= final_epoch <= 4:
        raise ValueError("Límites de fotos o épocas inválidos")
    model_name, learning_rate = TRIALS[trial]
    random.seed(42 + trial)
    torch.manual_seed(42 + trial)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("No hay GPU NVIDIA disponible; revisá el device plugin y el nodo Spot")
    device = torch.device("cuda")
    print(f"Ensayo {trial}: {model_name} en {torch.cuda.get_device_name(0)}", flush=True)
    started = time.monotonic()
    s3 = boto3.client("s3")
    training_root, test_root = download_data(s3, bucket)
    categories = choose_categories(training_root, test_root)
    classes = sorted({family_of(category) for category in categories})
    if len(categories) < 150 or len(classes) < 50:
        raise RuntimeError(f"Dataset incompleto: {len(categories)} variedades y {len(classes)} frutas")
    real_train_root = download_labeled_photos(s3, bucket, "real-world-train")
    train_data = FruitDataset(training_root, categories, classes, TRAIN_TRANSFORM, limit=train_limit, extra_root=real_train_root)
    test_data = FruitDataset(test_root, categories, classes, EVAL_TRANSFORM, limit=test_limit)
    # Cada fruta tiene el mismo peso total; las fotos naturales aportadas pesan
    # 8 veces más que un frame de estudio de la misma fruta.
    raw_weights = [8.0 if real_train_root.is_dir() and path.is_relative_to(real_train_root) else 1.0
                   for path, _ in train_data.items]
    class_totals = defaultdict(float)
    for (_, label), weight in zip(train_data.items, raw_weights):
        class_totals[label] += weight
    sample_weights = [weight / class_totals[label]
                      for (_, label), weight in zip(train_data.items, raw_weights)]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_data), replacement=True)
    train_loader = DataLoader(train_data, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=96, num_workers=4, pin_memory=True)
    real_train_images = sum(1 for p in real_train_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if real_train_root.is_dir() else 0
    print(f"{len(classes)} frutas, {len(categories)} variedades, {len(train_data):,} fotos de entrenamiento ({real_train_images} externas), {len(test_data):,} de evaluación interna", flush=True)

    model, head, last_block = make_model(model_name, len(classes))
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_path = SCRATCH / f"trial-{trial}.pt"
    start_epoch, best, history, previous_elapsed = restore_progress(s3, bucket, trial, model_name, classes, model, best_path, artifact_prefix)
    for epoch in range(start_epoch, final_epoch + 1):
        # Época 1: aprender la nueva cabeza. Épocas siguientes: afinar bloque visual final.
        if epoch == 2:
            for parameter in last_block.parameters():
                parameter.requires_grad = True
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate * (0.3 if epoch > 1 else 1.0),
            weight_decay=0.01,
        )
        model.train()
        total_loss = 0.0
        epoch_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        score = evaluate(model, test_loader, device)
        score.update({"epoch": epoch, "train_loss_sum": round(total_loss, 3),
                      "epoch_wall_sec": round(time.monotonic() - epoch_started, 2),
                      "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 2),
                      "elapsed_sec": round(previous_elapsed + time.monotonic() - started)})
        history.append(score)
        print(f"Época {epoch}/{final_epoch} | top-1 fruta={score['fruit_top1']:.1%} | top-3={score['fruit_top3']:.1%} | {score['elapsed_sec']}s", flush=True)
        if score["fruit_top1"] > best:
            best = score["fruit_top1"]
            cpu_weights = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            torch.save({"model_name": model_name, "classes": classes, "state_dict": cpu_weights, "epoch": epoch}, best_path)
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        progress_path = SCRATCH / "resume-upload.pt"
        torch.save({"trial": trial, "model_name": model_name, "classes": classes,
                    "epoch": epoch, "best_epoch": best_checkpoint["epoch"], "best_score": best,
                    "history": history, "elapsed_sec": previous_elapsed + time.monotonic() - started,
                    "best_state_dict": best_checkpoint["state_dict"],
                    "current_state_dict": {name: tensor.detach().cpu().clone()
                                           for name, tensor in model.state_dict().items()}}, progress_path)
        s3.upload_file(str(progress_path), bucket, f"{artifact_prefix}checkpoints/trial-{trial}/resume.pt")
        print(f"Época {epoch}: avance persistido en S3; pico de memoria GPU {score['peak_gpu_memory_gib']} GiB", flush=True)

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    external = external_test(s3, bucket, model, device, classes)
    report = {
        "trial": trial, "model": model_name, "fruit_classes": len(classes), "source_varieties": len(categories),
        "train_images": len(train_data), "internal_test_images": len(test_data),
        "train_images_per_variety_limit": train_limit, "test_images_per_variety_limit": test_limit,
        "final_epoch": final_epoch, "artifact_prefix": prefix_name,
        "real_world_train_images": real_train_images,
        "best_internal_fruit_top1": best, "history": history,
        "real_world_test": external, "elapsed_sec": round(previous_elapsed + time.monotonic() - started),
        "gpu_pod_seconds_approx": round(previous_elapsed + time.monotonic() - started),
        "caveat": ("Prueba corta del circuito; no representa la calidad del entrenamiento completo. " if final_epoch <= 2 else "")
                  + "Fruits-360 usa fondo blanco y frames correlacionados; la precisión interna no mide desempeño con fotos de celular.",
    }
    s3.upload_file(str(best_path), bucket, f"{artifact_prefix}results/trial-{trial}/model.pt")
    s3.put_object(Bucket=bucket, Key=f"{artifact_prefix}results/trial-{trial}/metrics.json", Body=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"), ContentType="application/json")
    print(f"Modelo y métricas guardados en S3: ensayo {trial}", flush=True)


if __name__ == "__main__":
    main()
