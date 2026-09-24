"""Inferencia local con el checkpoint descargado al finalizar el lab."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch
from PIL import Image

from fruit_classes import SPANISH_NAMES
from train import EVAL_TRANSFORM, make_model


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model, _, _ = make_model(checkpoint["model_name"], len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["classes"]


@torch.no_grad()
def predict(model, classes: list[str], image_bytes: bytes):
    with Image.open(io.BytesIO(image_bytes)) as photo:
        tensor = EVAL_TRANSFORM(photo.convert("RGB")).unsqueeze(0)
    probabilities = model(tensor).softmax(dim=1)[0]
    values, indices = probabilities.topk(5)
    return [
        {"fruta": SPANISH_NAMES.get(classes[int(index)], classes[int(index)]),
         "etiqueta_dataset": classes[int(index)],
         "puntaje_relativo": round(float(value), 4)}
        for value, index in zip(values, indices)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    model, classes = load_checkpoint(args.model)
    print(json.dumps(predict(model, classes, args.image.read_bytes()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
