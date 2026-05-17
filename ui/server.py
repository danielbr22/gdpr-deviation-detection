"""FastAPI backend for the pipeline dashboard UI."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
RESULTS_FILE = ROOT / "data" / "evaluation" / "results.json"
FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunConfig(BaseModel):
    smoke_test: bool = False
    force: bool = False
    provider: str = "ollama"  # "ollama" | "openai"
    api_key: str = ""


# ── Run state (singleton) ──────────────────────────────────────────────────────
_proc: asyncio.subprocess.Process | None = None
_run_log_file: str | None = None
_run_lines: list[str] = []
_run_done: asyncio.Event = asyncio.Event()
_run_done.set()  # no run active at startup
_run_lock = asyncio.Lock()


def _parse_timestamp(filename: str) -> str | None:
    try:
        stem = filename.removesuffix(".log")
        parts = stem.split("_")
        dt = datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S")
        return dt.isoformat()
    except Exception:
        return None


# ── API: logs ──────────────────────────────────────────────────────────────────
@app.get("/api/logs")
def get_logs():
    LOGS_DIR.mkdir(exist_ok=True)
    logs = []
    for f in sorted(LOGS_DIR.glob("pipeline_*.log"), reverse=True):
        stem = f.name.removesuffix(".log")
        results_snap = LOGS_DIR / f"{stem}_results.json"
        logs.append({
            "filename": f.name,
            "timestamp": _parse_timestamp(f.name),
            "has_results": results_snap.exists(),
            "size_bytes": f.stat().st_size,
        })
    return logs


@app.get("/api/logs/{filename}/content")
def get_log_content(filename: str):
    path = LOGS_DIR / filename
    if not path.exists() or not filename.endswith(".log"):
        raise HTTPException(404)
    return {"content": path.read_text(errors="replace")}


@app.get("/api/logs/{filename}/results")
def get_log_results(filename: str):
    stem = filename.removesuffix(".log")
    snap = LOGS_DIR / f"{stem}_results.json"
    if not snap.exists():
        raise HTTPException(404, detail="No results snapshot for this run")
    return json.loads(snap.read_text())


# ── API: current results ───────────────────────────────────────────────────────
@app.get("/api/results")
def get_current_results():
    if not RESULTS_FILE.exists():
        raise HTTPException(404, detail="results.json not found")
    return json.loads(RESULTS_FILE.read_text())


# ── API: pipeline config ───────────────────────────────────────────────────────
def _read_dotenv() -> dict[str, str]:
    """Parse ROOT/.env without executing it — returns key→value pairs."""
    result: dict[str, str] = {}
    env_file = ROOT / ".env"
    if not env_file.exists():
        return result
    for line in env_file.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


@app.get("/api/config")
def get_config():
    dotenv = _read_dotenv()
    # Prefer process env, fall back to .env file values
    env_key = os.environ.get("OPENAI_API_KEY") or dotenv.get("OPENAI_API_KEY", "")
    env_provider = os.environ.get("LLM_PROVIDER") or dotenv.get("LLM_PROVIDER", "")
    default_provider = env_provider or ("openai" if env_key else "ollama")
    return {"provider": default_provider, "has_api_key": bool(env_key)}


# ── API: pipeline run ──────────────────────────────────────────────────────────
@app.get("/api/run/status")
def run_status():
    running = _proc is not None and _proc.returncode is None
    return {"running": running, "log_file": _run_log_file}


@app.post("/api/run/start")
async def start_run(config: RunConfig = RunConfig()):
    global _proc, _run_log_file, _run_lines, _run_done

    async with _run_lock:
        if _proc is not None and _proc.returncode is None:
            raise HTTPException(409, detail="A pipeline run is already in progress")

        _run_lines = []
        _run_done = asyncio.Event()
        _run_log_file = None

        cmd = ["bash", str(ROOT / "run.sh")]
        if config.smoke_test:
            cmd.append("--test")
        elif config.force:
            cmd.append("--force")

        env = {**os.environ, "CAFFEINATED": "1", "LLM_PROVIDER": config.provider}
        if config.api_key:
            env["OPENAI_API_KEY"] = config.api_key

        _proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(ROOT),
            env=env,
        )
        asyncio.create_task(_drain_proc())
        return {"status": "started"}


async def _drain_proc():
    global _run_log_file, _run_done

    async for raw in _proc.stdout:
        line = raw.decode(errors="replace")
        _run_lines.append(line)
        # Extract log filename from the pipeline's startup banner
        if _run_log_file is None and "log:" in line:
            for token in line.split():
                if "pipeline_" in token and ".log" in token:
                    _run_log_file = Path(token).name
                    break

    await _proc.wait()

    if _proc.returncode == 0 and RESULTS_FILE.exists() and _run_log_file:
        stem = _run_log_file.removesuffix(".log")
        shutil.copy2(RESULTS_FILE, LOGS_DIR / f"{stem}_results.json")

    _run_done.set()


@app.post("/api/run/stop")
async def stop_run():
    global _proc, _run_done
    if _proc is None or _proc.returncode is not None:
        raise HTTPException(400, detail="No active run to stop")
    _proc.terminate()
    try:
        await asyncio.wait_for(_proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        _proc.kill()
        await _proc.wait()
    _run_done.set()
    return {"status": "stopped"}


@app.get("/api/run/stream")
async def stream_run(request: Request):
    async def generator():
        sent = 0
        while True:
            if await request.is_disconnected():
                break
            # Flush all buffered lines
            while sent < len(_run_lines):
                yield {"data": _run_lines[sent].rstrip("\n")}
                sent += 1
            if _run_done.is_set():
                yield {"data": "__DONE__"}
                break
            await asyncio.sleep(0.15)

    return EventSourceResponse(generator())


# ── Serve built frontend (production mode) ─────────────────────────────────────
if FRONTEND_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="assets",
    )

    @app.get("/{path:path}")
    def serve_spa(path: str):
        return FileResponse(str(FRONTEND_DIST / "index.html"))
