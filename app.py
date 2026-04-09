"""
Simple launcher for FinSense.

Usage:
  python app.py                # run dashboard
  python app.py --mode backend # run backend only
  python app.py --mode both    # run backend + dashboard
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_dashboard(port: int = 8501, host: str = "0.0.0.0") -> int:
    """Run Streamlit dashboard in foreground."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "dashboard" / "app.py"),
        "--server.port",
        str(port),
        "--server.address",
        host,
    ]
    return subprocess.call(cmd, cwd=str(ROOT))


def run_backend(port: int = 8001, host: str = "0.0.0.0") -> int:
    """Run backend in foreground."""
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.call(cmd, cwd=str(ROOT))


def run_both(backend_port: int = 8001, dashboard_port: int = 8501, host: str = "0.0.0.0") -> int:
    """Run backend in background and dashboard in foreground."""
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        host,
        "--port",
        str(backend_port),
    ]
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(ROOT))
    try:
        return run_dashboard(port=dashboard_port, host=host)
    finally:
        if backend_proc.poll() is None:
            # Graceful shutdown first
            try:
                backend_proc.send_signal(signal.SIGINT)
            except Exception:
                pass
            try:
                backend_proc.wait(timeout=5)
            except Exception:
                backend_proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="FinSense launcher")
    parser.add_argument("--mode", choices=["dashboard", "backend", "both"], default="dashboard")
    parser.add_argument("--backend-port", type=int, default=int(os.getenv("BACKEND_PORT", "8001")))
    parser.add_argument("--dashboard-port", type=int, default=int(os.getenv("STREAMLIT_PORT", "8501")))
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    args = parser.parse_args()

    if args.mode == "backend":
        return run_backend(port=args.backend_port, host=args.host)
    if args.mode == "both":
        return run_both(
            backend_port=args.backend_port,
            dashboard_port=args.dashboard_port,
            host=args.host,
        )
    return run_dashboard(port=args.dashboard_port, host=args.host)


if __name__ == "__main__":
    raise SystemExit(main())

