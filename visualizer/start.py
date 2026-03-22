#!/usr/bin/env python3
"""
Robotics-APCC Live Visualizer — Startup Script

Launches all three servers and streams their logs with colored prefixes:
  1. FastAPI backend  (port 8080)
  2. Viser 3D server  (port 8081)
  3. Vite frontend    (port 5173)

Usage:
    python visualizer/start.py          # from project root
    python start.py                     # from visualizer/
"""

import sys
import os
import signal
import subprocess
import threading
import multiprocessing
from multiprocessing import Queue
from pathlib import Path

# ── Resolve project root ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FRONTEND_DIR = SCRIPT_DIR / "frontend"


# ── ANSI colors for log prefixes ──
COLORS = {
    "api":   "\033[36m",   # cyan
    "viser": "\033[35m",   # magenta
    "vite":  "\033[33m",   # yellow
    "reset": "\033[0m",
    "bold":  "\033[1m",
    "dim":   "\033[2m",
}


def _prefix(tag: str) -> str:
    color = COLORS.get(tag, COLORS["reset"])
    return f"{color}{COLORS['bold']}[{tag:>5}]{COLORS['reset']} "


# ── Backend launchers (run in child processes) ──

def start_viser(scene_queue: Queue, port: int = 8081):
    sys.path.insert(0, str(PROJECT_ROOT))
    from visualizer.backend.viser_server import run_viser_server
    run_viser_server(scene_queue=scene_queue, port=port)


def start_fastapi(scene_queue: Queue, port: int = 8080):
    sys.path.insert(0, str(PROJECT_ROOT))
    from visualizer.backend.server import run_server
    run_server(scene_queue=scene_queue, port=port)


# ── Stream subprocess stdout/stderr with colored prefix ──

def _stream_pipe(pipe, tag: str):
    """Read lines from a pipe and print with a colored prefix."""
    prefix = _prefix(tag)
    try:
        for raw_line in iter(pipe.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
            print(f"{prefix}{line}", flush=True)
    except Exception:
        pass
    finally:
        pipe.close()


def main():
    print(f"""
{COLORS['bold']}{'=' * 60}
  Robotics-APCC Live Visualizer
{'=' * 60}{COLORS['reset']}
""")

    # ── Shared queue for FastAPI → Viser communication ──
    scene_queue = Queue()

    # ── Start Viser server (Python process) ──
    viser_proc = multiprocessing.Process(
        target=start_viser, args=(scene_queue,),
        name="ViserServer", daemon=True,
    )
    viser_proc.start()
    print(f"  {_prefix('viser')}3D server starting → http://localhost:8081")

    # ── Start FastAPI server (Python process) ──
    api_proc = multiprocessing.Process(
        target=start_fastapi, args=(scene_queue,),
        name="FastAPIServer", daemon=True,
    )
    api_proc.start()
    print(f"  {_prefix('api')}REST server starting → http://localhost:8080")

    # ── Start Vite frontend dev server (subprocess with log streaming) ──
    vite_proc = None
    if not FRONTEND_DIR.exists():
        print(f"  {COLORS['dim']}⚠ frontend/ not found — skipping Vite{COLORS['reset']}")
    elif not (FRONTEND_DIR / "node_modules").exists():
        print(f"  {COLORS['dim']}⚠ node_modules missing — run 'cd visualizer/frontend && npm install' first{COLORS['reset']}")
    else:
        vite_proc = subprocess.Popen(
            ["npx", "vite", "--host", "0.0.0.0", "--port", "5173"],
            cwd=str(FRONTEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "FORCE_COLOR": "0"},  # disable Vite ANSI so our prefix is clean
        )
        # Stream stdout and stderr in background threads
        threading.Thread(target=_stream_pipe, args=(vite_proc.stdout, "vite"), daemon=True).start()
        threading.Thread(target=_stream_pipe, args=(vite_proc.stderr, "vite"), daemon=True).start()
        print(f"  {_prefix('vite')}Frontend dev server → http://localhost:5173")

    print(f"""
{COLORS['bold']}  → Open http://localhost:5173 in your browser
  → Press Ctrl+C to stop all servers
{'=' * 60}{COLORS['reset']}
""")

    # ── Graceful shutdown ──
    def shutdown(signum=None, frame=None):
        print(f"\n{COLORS['dim']}  Shutting down...{COLORS['reset']}")
        if vite_proc and vite_proc.poll() is None:
            vite_proc.terminate()
            try:
                vite_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                vite_proc.kill()
        viser_proc.terminate()
        api_proc.terminate()
        viser_proc.join(timeout=3)
        api_proc.join(timeout=3)
        print(f"  {COLORS['dim']}All servers stopped.{COLORS['reset']}")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Wait for processes ──
    try:
        while True:
            # Check if any critical process died unexpectedly
            if not viser_proc.is_alive():
                print(f"\n  {_prefix('viser')}Process exited unexpectedly!")
                shutdown()
            if not api_proc.is_alive():
                print(f"\n  {_prefix('api')}Process exited unexpectedly!")
                shutdown()
            if vite_proc and vite_proc.poll() is not None:
                print(f"\n  {_prefix('vite')}Process exited unexpectedly!")
                shutdown()
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
