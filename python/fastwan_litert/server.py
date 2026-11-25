"""`fastwan-litert serve`: a local studio. One engine thread owns the graphs
and runs jobs one at a time; the HTTP thread only reads a job table, so the
page's polling never waits on the model (fastwan-rs's `serve` design). Binds
127.0.0.1 and sends no CORS headers: another site cannot queue clips here."""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .geometry import Geometry

PAGE = (Path(__file__).parent / "studio.html").read_text()


class Jobs:
    def __init__(self, pipe, gallery: Path | None, log):
        self.pipe, self.gallery, self.log = pipe, gallery, log
        self.lock = threading.Lock()
        self.table: dict[str, dict] = {}
        self.queue: list[str] = []
        self.cv = threading.Condition(self.lock)
        threading.Thread(target=self._engine, daemon=True).start()

    def prompts(self) -> list[str]:
        if not self.gallery:
            return []
        return sorted(p.stem for p in self.gallery.glob("*.safetensors"))

    def submit(self, prompt: str, size: str, seed: int) -> str:
        jid = uuid.uuid4().hex[:12]
        with self.cv:
            self.table[jid] = {"id": jid, "prompt": prompt, "size": size, "seed": seed, "state": "queued", "events": [], "frames": []}
            self.queue.append(jid)
            self.cv.notify()
        return jid

    def status(self, jid: str, since: int) -> dict | None:
        with self.lock:
            j = self.table.get(jid)
            if not j:
                return None
            return {k: v for k, v in j.items() if k != "frames"} | {"events": j["events"][since:], "next": len(j["events"])}

    def frame(self, jid: str, i: int) -> bytes | None:
        with self.lock:
            j = self.table.get(jid)
            if not j or i >= len(j["frames"]):
                return None
            return j["frames"][i]

    def _engine(self):
        from safetensors.numpy import load_file

        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                jid = self.queue.pop(0)
                j = self.table[jid]
                j["state"] = "running"
            try:
                g = Geometry.parse(j["size"])
                if self.gallery is None:
                    raise RuntimeError("no --gallery: the studio needs precomputed prompt embeddings")
                path = self.gallery / f"{j['prompt']}.safetensors"
                e = load_file(str(path))
                embeds = e.get("prompt_embeds", next(iter(e.values()))).astype(np.float32)

                def on(ev):
                    ev = dict(ev)
                    if ev["type"] == "frames":
                        pngs = [_png_bytes(f) for f in ev.pop("rgb")]
                        with self.lock:
                            j["frames"].extend(pngs)
                    with self.lock:
                        j["events"].append(ev)

                t0 = time.time()
                video, tm = self.pipe.generate(embeds, g, int(j["seed"]), on)
                with self.lock:
                    j["state"] = "done"
                    j["events"].append({"type": "done", "frames": video.frames, "width": video.width, "height": video.height, "secs": time.time() - t0})
            except Exception as ex:  # noqa: BLE001
                self.log(f"job {jid} failed: {ex!r}")
                with self.lock:
                    j["state"] = "failed"
                    j["events"].append({"type": "error", "message": str(ex)})


def _png_bytes(rgb) -> bytes:
    import io
    import struct
    import zlib

    h, w, _ = rgb.shape

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def serve(pipe, host: str, port: int, gallery: Path | None, log=print) -> None:
    jobs = Jobs(pipe, gallery, log)
    m = pipe.m

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path, _, query = self.path.partition("?")
            q = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            if path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/info":
                self._json(200, {"model": m.raw["model"], "weights": m.weights, "sizes": m.raw["sizes"], "prompts": jobs.prompts(), "accelerator": pipe.store.accelerator})
            elif path == "/api/job":
                st = jobs.status(q.get("id", ""), int(q.get("since", "0")))
                self._json(200 if st else 404, st or {"error": "no such job"})
            elif path == "/api/frame":
                png = jobs.frame(q.get("id", ""), int(q.get("i", "0")))
                if png is None:
                    self._json(404, {"error": "no such frame"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/generate":
                self._json(404, {"error": "not found"})
                return
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": "JSON only"})
                return
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            try:
                Geometry.parse(req.get("size", "")).validate(pipe.cfg)
            except ValueError as e:
                self._json(400, {"error": str(e)})
                return
            jid = jobs.submit(str(req.get("prompt", "")), req["size"], int(req.get("seed", 7)))
            self._json(200, {"id": jid})

    srv = ThreadingHTTPServer((host, port), H)
    log(f"fastwan-litert studio on http://{host}:{port}/ ({m.raw['model']}, {m.weights}, {pipe.store.accelerator})")
    srv.serve_forever()
