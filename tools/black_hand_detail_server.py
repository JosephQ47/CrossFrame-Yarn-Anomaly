"""Local image-based visible-yarn centerline review.

Runs separately from the locked A blind review. Drafts are append-only in the
gitignored outputs directory and are never consumed as training truth here.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
LOCKED = ROOT / "outputs/black_july_blind_review/july_reviewer_A_locked.csv"
LOCK_MANIFEST = ROOT / "outputs/black_july_blind_review/july_reviewer_A_lock.json"
OUT = ROOT / "outputs/black_hand_detail_review"
JOURNAL = OUT / "yarn_detail.reviews.jsonl"
UI = Path(__file__).with_name("black_hand_detail_ui.html")
SCRIPT = Path(__file__).with_name("black_yarn_detail.js")
MUTEX = threading.Lock()


def roster():
    manifest = json.loads(LOCK_MANIFEST.read_text(encoding="utf-8"))
    if hashlib.sha256(LOCKED.read_bytes()).hexdigest() != manifest["csv_sha256"]:
        raise RuntimeError("A locked review hash mismatch")
    with LOCKED.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    hand = [r for r in rows if r["hand_or_occlusion"] == "hand"]
    if len(hand) != 7 or len({r["frame_id"] for r in hand}) != 7:
        raise RuntimeError("expected seven unique A-labeled hand frames")
    return {r["frame_id"]: r for r in hand}


def current_reviews(items):
    reviews = {}
    if JOURNAL.exists():
        with JOURNAL.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    if item["frame_id"] not in items:
                        raise RuntimeError("hand review journal frame mismatch")
                    reviews[item["frame_id"]] = item
    return reviews


def valid_points(value, minimum, maximum, size):
    h, w = size
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError("invalid point count")
    output = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("invalid point")
        x, y = point
        if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point)
                or not (0 <= x <= w and 0 <= y <= h)):
            raise ValueError("point outside source image")
        output.append([round(float(x), 1), round(float(y), 1)])
    return output


def validate(payload, source):
    if not isinstance(payload, dict):
        raise ValueError("request must be JSON object")
    import cv2
    import numpy as np
    path = Path(source["source_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("source image hash mismatch")
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("source image unreadable")
    size = img.shape[:2]
    line = payload.get("yarn_centerline", [])
    centerline = valid_points(line, 0, 100, size)
    if centerline and len(centerline) < 2:
        raise ValueError("centerline needs at least two points")
    if source["defect_status"] != "black_yarn" and centerline:
        raise ValueError("uncertain frame cannot receive a yarn centerline")
    note = payload.get("notes", "")
    if not isinstance(note, str) or len(note) > 2000:
        raise ValueError("invalid note")
    note = note.strip()
    if source["defect_status"] == "black_yarn" and not centerline and not note:
        raise ValueError("draw the visible yarn centerline or explain why it cannot be drawn")
    if source["defect_status"] != "black_yarn" and not note:
        raise ValueError("explain that the yarn cannot be determined")
    return {"yarn_centerline": centerline, "notes": note}


class Handler(BaseHTTPRequestHandler):
    items = {}

    def reply(self, status, body, mime="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", mime)
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
            if path == "/detail.js":
                return self.reply(200, SCRIPT.read_bytes(), "text/javascript; charset=utf-8")
            if path == "/api/items":
                reviews = current_reviews(self.items)
                return self.reply(200, [{"frame_id": key, "cam": row["cam"],
                                         "status": row["defect_status"],
                                         "sha256": row["sha256"],
                                         "a_boxes": json.loads(row["box_or_centerline_json"] or "[]"),
                                         "detail": reviews.get(key)}
                                        for key, row in self.items.items()])
            if path == "/api/export":
                reviews = current_reviews(self.items)
                return self.reply(200, {"scope": "draft visible-yarn detail; A review immutable",
                                        "items": list(reviews.values())})
            if path.startswith("/image/"):
                key = path.removeprefix("/image/")
                row = self.items.get(key)
                if row is None:
                    return self.reply(404, {"error": "unknown frame"})
                source = Path(row["source_path"])
                data = source.read_bytes()
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise RuntimeError("source image hash mismatch")
                return self.reply(200, data, "image/jpeg")
            return self.reply(404, {"error": "not found"})
        except (OSError, ValueError, RuntimeError) as exc:
            return self.reply(400, {"error": str(exc)})

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        if not path.startswith("/api/items/"):
            return self.reply(404, {"error": "not found"})
        key = path.removeprefix("/api/items/")
        source = self.items.get(key)
        if source is None:
            return self.reply(404, {"error": "unknown frame"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 30000:
                raise ValueError("invalid body size")
            payload = json.loads(self.rfile.read(length))
            detail = validate(payload, source)
            entry = {"frame_id": key, "source_sha256": source["sha256"],
                     "reviewed_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                     **detail}
            with MUTEX:
                OUT.mkdir(parents=True, exist_ok=True)
                with JOURNAL.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            return self.reply(200, {"ok": True, "reviewed_at_utc": entry["reviewed_at_utc"]})
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            return self.reply(400, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    Handler.items = roster()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Hand detail review: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
