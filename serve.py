"""Interfaz local para probar fotos una vez descargado best.pt."""

import argparse
import json
import os
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from infer import load_checkpoint, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    bucket = os.environ.get("S3_BUCKET")
    if bucket:
        import boto3

        args.model.parent.mkdir(parents=True, exist_ok=True)
        s3 = boto3.client("s3")
        s3.download_file(bucket, "serving/best.pt", str(args.model))
        s3.download_file(bucket, "serving/summary.json", str(args.model.parent / "summary.json"))
    model, classes = load_checkpoint(args.model)
    summary_path = args.model.parent / "summary.json"
    index = Path(__file__).with_name("index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, data: bytes, mime: str, status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/healthz":
                self.respond(b"ok", "text/plain")
            elif self.path == "/":
                self.respond(index, "text/html; charset=utf-8")
            elif self.path == "/api/summary":
                body = summary_path.read_bytes() if summary_path.exists() else b"{}"
                self.respond(body, "application/json; charset=utf-8")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if self.path != "/api/predict":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 10_000_000:
                self.send_error(HTTPStatus.BAD_REQUEST, "La foto debe pesar menos de 10 MB")
                return
            try:
                started = time.perf_counter()
                result = predict(model, classes, self.rfile.read(length))
                body = json.dumps({"top5": result,
                                   "inference_ms": round((time.perf_counter() - started) * 1000, 1)},
                                  ensure_ascii=False).encode("utf-8")
                self.respond(body, "application/json; charset=utf-8")
            except Exception:
                self.send_error(HTTPStatus.BAD_REQUEST, "No pude procesar esa foto")

    print(f"Probá el modelo en http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
