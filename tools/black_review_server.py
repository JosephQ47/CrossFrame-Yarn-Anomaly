"""Local, prediction-free image review UI for the July development batch.

Run: python tools/black_review_server.py
Then open http://127.0.0.1:8765/?reviewer=A (or B) in a browser.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "data/black_july_blind_review_2026-10-31"
UI = Path(__file__).with_name("black_review_ui.html")
DATASET_DIR = Path("D:/dataset/black_fabric").resolve()
LOCK = threading.Lock()
STATUSES = {"", "normal", "black_yarn", "other_defect", "uncertain"}
FIELDS = ["frame_id", "source_path", "sha256", "cam", "defect_status",
          "defect_type", "box_or_centerline_json", "run_state", "hand_or_occlusion",
          "event_id", "reviewer_id", "reviewed_at", "notes"]


def sheet_path(reviewer):
    if reviewer not in ("A", "B"):
        raise ValueError("reviewer must be A or B")
    return REVIEW_DIR / f"july_reviewer_{reviewer}.csv"


def read_sheet(reviewer):
    with sheet_path(reviewer).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 69 or len({r["frame_id"] for r in rows}) != 69:
        raise ValueError("review sheet must contain 69 unique frames")
    return rows


def verified_images():
    a, b = read_sheet("A"), read_sheet("B")
    images = {}
    for left, right in zip(a, b):
        if any(left[k] != right[k] for k in ("frame_id", "source_path", "sha256", "cam")):
            raise ValueError("A/B image inventories differ")
        path = Path(left["source_path"]).resolve()
        if path.parent != DATASET_DIR or not path.is_file():
            raise ValueError(f"image outside verified dataset: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != left["sha256"]:
            raise ValueError(f"source image SHA mismatch: {path}")
        images[left["frame_id"]] = path
    return images


def validate_update(payload):
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    status = payload.get("defect_status", "")
    if status not in STATUSES:
        raise ValueError("invalid defect status")
    updates = {"defect_status": status}
    for key in ("defect_type", "run_state", "hand_or_occlusion", "event_id", "notes"):
        value = payload.get(key, "")
        if not isinstance(value, str) or len(value) > 2000:
            raise ValueError(f"invalid {key}")
        updates[key] = value
    boxes = payload.get("boxes", [])
    if not isinstance(boxes, list) or len(boxes) > 30:
        raise ValueError("invalid boxes")
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4 or
                any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                    for v in box) or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("invalid box coordinates")
    if status == "normal" and boxes:
        raise ValueError("normal images cannot retain defect boxes")
    updates["box_or_centerline_json"] = json.dumps(boxes, ensure_ascii=False)
    return updates


class Handler(BaseHTTPRequestHandler):
    images = {}

    def reply(self, code, body, content_type="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/":
                return self.reply(200, UI.read_bytes(), "text/html; charset=utf-8")
            if path.startswith("/api/frames/"):
                reviewer = path.removeprefix("/api/frames/")
                rows = read_sheet(reviewer)
                return self.reply(200, [{k: v for k, v in r.items() if k != "source_path"}
                                        for r in rows])
            if path.startswith("/image/"):
                frame_id = path.removeprefix("/image/")
                image = self.images.get(frame_id)
                if image is None:
                    return self.reply(404, {"error": "unknown image"})
                data = image.read_bytes()
                return self.reply(200, data, "image/jpeg")
            return self.reply(404, {"error": "not found"})
        except (ValueError, OSError) as exc:
            return self.reply(400, {"error": str(exc)})

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        parts = path.strip("/").split("/")
        if len(parts) != 4 or parts[:2] != ["api", "frames"]:
            return self.reply(404, {"error": "not found"})
        reviewer, frame_id = parts[2], parts[3]
        if frame_id not in self.images:
            return self.reply(404, {"error": "unknown image"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 10000:
                raise ValueError("invalid request length")
            updates = validate_update(json.loads(self.rfile.read(length)))
            with LOCK:
                rows = read_sheet(reviewer)
                row = next(r for r in rows if r["frame_id"] == frame_id)
                row.update(updates)
                row["reviewer_id"] = reviewer
                from datetime import datetime, timezone
                row["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                target = sheet_path(reviewer)
                temp = target.with_suffix(".tmp")
                with temp.open("w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                os.replace(temp, target)
            return self.reply(200, {"ok": True, "reviewed_at": row["reviewed_at"]})
        except (ValueError, KeyError, StopIteration, json.JSONDecodeError) as exc:
            return self.reply(400, {"error": str(exc)})


def main():
    global REVIEW_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    args = parser.parse_args()
    REVIEW_DIR = args.review_dir.resolve()
    Handler.images = verified_images()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Blind review: http://127.0.0.1:{args.port}/?reviewer=A", flush=True)
    print(f"Second reviewer: http://127.0.0.1:{args.port}/?reviewer=B", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
