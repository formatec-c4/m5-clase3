"""Selecciona el mejor ensayo solo con el test interno; el externo no se usa para elegir."""

import json
import shutil
import sys
from pathlib import Path


def main(root: Path):
    reports = []
    for trial in range(3):
        path = root / f"trial-{trial}" / "metrics.json"
        if path.exists():
            reports.append(json.loads(path.read_text()))
    if not reports:
        raise SystemExit("No hay métricas descargadas")
    winner = max(reports, key=lambda report: report["best_internal_fruit_top1"])
    shutil.copy2(root / f"trial-{winner['trial']}" / "model.pt", root / "best.pt")
    summary = {
        "winner_trial": winner["trial"],
        "model": winner["model"],
        "internal_fruit_top1": winner["best_internal_fruit_top1"],
        "external_test": winner["real_world_test"],
        "training_gpu_pod_hours_approx": round(sum(r.get("gpu_pod_seconds_approx", r["elapsed_sec"]) for r in reports) / 3600, 3),
        "winner_peak_gpu_memory_gib": max((epoch.get("peak_gpu_memory_gib", 0) for epoch in winner["history"]), default=0),
        "all_trials": [{"trial": r["trial"], "model": r["model"],
                        "internal_fruit_top1": r["best_internal_fruit_top1"],
                        "elapsed_sec": r["elapsed_sec"],
                        "external_test": r["real_world_test"]} for r in reports],
        "warning": "GPU-pod-horas es tiempo aproximado reservado por los ensayos, no medición de uso activo ni costo AWS. Sin suficientes fotos externas etiquetadas, no se puede afirmar robustez con fotos reales.",
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
