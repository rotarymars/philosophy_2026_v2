#!/usr/bin/env python3
"""Live viewer for polypack episodes.

Run from the repository root:
    python3 viz/server.py            # then open http://localhost:8000
    python3 viz/server.py --port 9000

It serves viz/index.html and, at /stream, runs build/polypack_visualize and
relays its JSON lines to the browser as Server-Sent Events.
Uses only the Python standard library.
"""

import argparse
import glob
import json
import os
import subprocess
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(ROOT, "build", "polypack_visualize")
PAGE = os.path.join(ROOT, "viz", "index.html")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_body(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            with open(PAGE, "rb") as f:
                self.send_body(200, "text/html; charset=utf-8", f.read())
        elif url.path == "/models":
            models = sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "models", "*.pt")))
            self.send_body(200, "application/json", json.dumps(models).encode())
        elif url.path == "/stream":
            self.stream(parse_qs(url.query))
        else:
            self.send_body(404, "text/plain", b"not found")

    def stream(self, q):
        get = lambda k, d: q.get(k, [d])[0]
        player = get("player", "greedy")
        if player not in ("policy", "firstfit", "greedy"):
            return self.send_body(400, "text/plain", b"bad player")
        cmd = [BINARY, "--player", player, "--seed", str(int(get("seed", "1"))),
               "--min-items", str(int(get("minItems", "501"))),
               "--items", str(int(get("maxItems", "1000")))]
        if player == "policy":
            model = os.path.realpath(os.path.join(ROOT, get("model", "models/policy.pt")))
            if not model.startswith(os.path.join(ROOT, "models") + os.sep):
                return self.send_body(400, "text/plain", b"model must be under models/")
            cmd += ["--model", model]
            if get("sample", "0") == "1":
                cmd.append("--sample")
        else:
            cmd += ["--settle", "0" if get("settle", "1") == "0" else "1"]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            for line in proc.stdout:
                self.wfile.write(f"data: {line.strip()}\n\n".encode())
                self.wfile.flush()
            # If the visualizer failed, show why in the page instead of nothing.
            if proc.wait() != 0:
                err = proc.stderr.read().strip()
                print(f"visualizer failed: {' '.join(cmd)}\n{err}", flush=True)
                first = err.splitlines()[0] if err else f"polypack_visualize exited with code {proc.returncode}"
                msg = json.dumps({"type": "error", "message": first[:300]})
                self.wfile.write(f"data: {msg}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            proc.kill()
            proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    # Rebuild first so a `git pull` is picked up (a stale binary cannot read new
    # checkpoints and has no live offline view). Quick if nothing changed.
    if os.path.isdir(os.path.join(ROOT, "build")):
        print("building polypack_visualize ...", flush=True)
        subprocess.run(["cmake", "--build", os.path.join(ROOT, "build"), "--target", "polypack_visualize"],
                       cwd=ROOT, check=True)
    if not os.path.exists(BINARY):
        raise SystemExit(f"{BINARY} not found; run scripts/setup.sh first")
    print(f"polypack viewer on http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
