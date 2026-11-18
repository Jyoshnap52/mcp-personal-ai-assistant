import json
import logging
from datetime import UTC, datetime
from pathlib import Path

AUDIT_LOG_PATH = Path(__file__).resolve().parent / "audit.log"

# Configure security audit logger
logger = logging.getLogger("security_audit")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(AUDIT_LOG_PATH, encoding="utf-8")
    formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "event": %(message)s}'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_security_event(
    event_type: str,
    tool_name: str | None = None,
    user_prompt: str | None = None,
    details: dict | None = None,
    status: str = "SUCCESS",
):
    """Writes a structured security event to security/audit.log for compliance."""
    payload = {
        "event_type": event_type,
        "tool_name": tool_name,
        "status": status,
        "user_prompt_preview": (user_prompt[:100] + "...") if user_prompt else None,
        "details": details or {},
        "utc_time": datetime.now(UTC).isoformat(),
    }
    logger.info(json.dumps(payload))
