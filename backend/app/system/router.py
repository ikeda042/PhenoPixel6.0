import logging
import os
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException

router_system = APIRouter(tags=["system"])
logger = logging.getLogger("uvicorn.error")
REPO_ROOT = Path(__file__).resolve().parents[2].parent


@router_system.post("/system/git-pull")
def git_pull():
    if not (REPO_ROOT / ".git").exists():
        raise HTTPException(status_code=500, detail="Repository root not found")

    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        ["git", "pull", "--ff-only"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        logger.info("git pull stdout:\n%s", stdout)
    if stderr:
        logger.warning("git pull stderr:\n%s", stderr)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=stderr or stdout or "git pull failed")
    return {"status": "ok", "output": stdout}
