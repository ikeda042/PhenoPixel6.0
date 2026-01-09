import json
import logging
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)
_ENV_LOADED = False


def _load_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        if key not in os.environ:
            os.environ[key] = value


def _get_webhook_url() -> str | None:
    _load_env()
    return (
        os.getenv("SLACK_WEHBOOK_URL")
        or os.getenv("SLACK_WEBHOOK_URL")
        or os.getenv("slack_webhook_url")
    )


def _get_base_path() -> str | None:
    _load_env()
    return (
        os.getenv("BASE_PATH")
        or os.getenv("SERVER_ORIGIN")
        or os.getenv("server_origin")
    )


def notify_slack_database_created(
    db_name: str,
    *,
    message: str | None = None,
    contour_count: int | None = None,
    param1: int | None = None,
    image_size: int | None = None,
) -> None:
    webhook_url = _get_webhook_url()
    if not webhook_url:
        return

    base_path = _get_base_path()
    db_url = None
    cells_url = None
    if base_path:
        base_path = base_path.rstrip("/")
        db_url = f"{base_path}/databases/?{urlencode({'db_name': db_name})}"
        cells_url = f"{base_path}/cells?{urlencode({'db': db_name})}"

    slack_message = (
        message
        or f"nd2extract\u304c\u5b8c\u4e86\u3057\u307e\u3057\u305f\u3002"
        f"database `{db_name}` \u3092\u4f5c\u6210\u3057\u307e\u3057\u305f\u3002"
    )
    details = []
    if contour_count is not None:
        details.append(f"{contour_count}\u500b\u306e\u8f2a\u90ed\u3092\u53d6\u5f97\u3057\u307e\u3057\u305f\u3002")
    param_parts = []
    if param1 is not None:
        param_parts.append(f"param1 = {param1}")
    if image_size is not None:
        param_parts.append(f"image_size = {image_size} x {image_size} px^2")
    if param_parts:
        details.append(", ".join(param_parts))
    if cells_url:
        details.append(f"URL : {cells_url}")
    if details:
        slack_message = f"{slack_message}\n" + "\n".join(details)
    if db_url and cells_url is None:
        slack_message = f"{slack_message}\n{db_url}"

    payload = json.dumps({"text": slack_message}).encode("utf-8")
    request = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            response.read()
        logger.info("Slack notified for database creation: %s", db_name)
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "ignore")
        except Exception:
            body = ""
        logger.warning(
            "Slack notification failed %s: %s",
            exc.code,
            body or exc.reason,
        )
    except URLError as exc:
        logger.warning("Slack notification failed: %s", exc.reason)
    except Exception as exc:
        logger.warning("Slack notification failed: %s", exc)


def notify_slack_bulk_engine_completed(
    db_name: str,
    *,
    task: str,
    label: str | None = None,
    channel: str | None = None,
    degree: int | None = None,
    center_ratio: float | None = None,
    max_to_min_ratio: float | None = None,
) -> None:
    webhook_url = _get_webhook_url()
    if not webhook_url:
        return

    base_path = _get_base_path()
    bulk_url = None
    if base_path:
        base_path = base_path.rstrip("/")
        bulk_url = f"{base_path}/bulk-engine?{urlencode({'dbname': db_name})}"

    task_text = task.strip() if task else "bulkengine task"
    slack_message = f"bulkengine\u306e{task_text}\u304c\u5b8c\u4e86\u3057\u307e\u3057\u305f\u3002database `{db_name}`"
    details: list[str] = []
    label_text = str(label).strip() if label is not None else ""
    if label_text:
        details.append(f"label = {label_text}")
    if channel:
        details.append(f"channel = {channel}")
    if degree is not None:
        details.append(f"degree = {degree}")
    if center_ratio is not None:
        details.append(f"center_ratio = {center_ratio}")
    if max_to_min_ratio is not None:
        details.append(f"max_to_min_ratio = {max_to_min_ratio}")
    if bulk_url:
        details.append(f"URL : {bulk_url}")
    if details:
        slack_message = f"{slack_message}\n" + "\n".join(details)

    payload = json.dumps({"text": slack_message}).encode("utf-8")
    request = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            response.read()
        logger.info("Slack notified for bulk engine: %s (%s)", db_name, task_text)
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "ignore")
        except Exception:
            body = ""
        logger.warning(
            "Slack notification failed %s: %s",
            exc.code,
            body or exc.reason,
        )
    except URLError as exc:
        logger.warning("Slack notification failed: %s", exc.reason)
    except Exception as exc:
        logger.warning("Slack notification failed: %s", exc)
