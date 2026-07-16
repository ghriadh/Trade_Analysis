"""
ETL status writer.

Called by each pipeline entrypoint (run_bronze, run_silver, run_gold)
to record execution results in data/etl_status.json.

Streamlit reads this file to display pipeline metrics without
re-running the ETL.

Usage in a pipeline entrypoint:
    from src.etl_status import ETLStatusWriter

    writer = ETLStatusWriter()
    writer.start_pipeline("bronze")
    # ... run pipeline steps ...
    writer.record_stage("bronze", "countries",    rows=249,    duration_s=1.2,  status="done")
    writer.record_stage("bronze", "indicators",   rows=17,     duration_s=0.4,  status="done")
    writer.record_stage("bronze", "indicators_values", rows=31500, duration_s=120.0, status="done")
    writer.record_stage("bronze", "trade_data",   rows=142000, duration_s=847.0, status="done",
                        checkpoints={"2015": 187, "2016": 185, ...})
    writer.end_pipeline("bronze")
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STATUS_PATH = Path("data/etl_status.json")

PIPELINES = ("bronze", "silver", "gold")


class ETLStatusWriter:
    """
    Read-modify-write wrapper around etl_status.json.

    Thread-safety: not required — pipelines run sequentially.
    The file is read once at init, modified in memory, and written
    at each record_stage / end_pipeline call.
    """

    def __init__(self, path: Path = DEFAULT_STATUS_PATH) -> None:
        self.path = path
        self._data = self._load()

    # ============================================================
    # PUBLIC API
    # ============================================================

    def start_pipeline(self, pipeline: str) -> None:
        """Mark a pipeline as running and record its start time."""
        self._ensure_pipeline(pipeline)
        self._data["pipelines"][pipeline]["status"]     = "running"
        self._data["pipelines"][pipeline]["started_at"] = _now()
        self._data["pipelines"][pipeline]["ended_at"]   = None
        self._save()
        logger.info("ETL status | pipeline=%s started", pipeline)

    def record_stage(
        self,
        pipeline: str,
        stage: str,
        rows: int,
        duration_s: float,
        status: str = "done",
        checkpoints: dict | None = None,
        error: str | None = None,
    ) -> None:
        """
        Record the result of one pipeline stage.

        Args:
            pipeline:    "bronze", "silver", or "gold"
            stage:       module name (e.g. "trade_data", "countries")
            rows:        number of rows produced
            duration_s:  wall-clock seconds for this stage
            status:      "done" | "failed" | "skipped"
            checkpoints: optional dict for partitioned extractions
                         e.g. {"2015": 187, "2016": 185, ...} (year → reporters done)
            error:       optional error message if status == "failed"
        """
        self._ensure_pipeline(pipeline)
        self._data["pipelines"][pipeline]["stages"][stage] = {
            "status":      status,
            "rows":        rows,
            "duration_s":  round(duration_s, 1),
            "updated_at":  _now(),
            **({"checkpoints": checkpoints} if checkpoints else {}),
            **({"error": error}             if error       else {}),
        }
        self._save()

    def end_pipeline(self, pipeline: str, status: str = "done") -> None:
        """Mark a pipeline as complete and record its end time."""
        self._ensure_pipeline(pipeline)
        self._data["pipelines"][pipeline]["status"]   = status
        self._data["pipelines"][pipeline]["ended_at"] = _now()
        self._data["last_updated"] = _now()
        self._save()
        logger.info("ETL status | pipeline=%s %s", pipeline, status)

    # ============================================================
    # INTERNAL
    # ============================================================

    def _ensure_pipeline(self, pipeline: str) -> None:
        if pipeline not in PIPELINES:
            raise ValueError(f"Unknown pipeline '{pipeline}'. Valid: {PIPELINES}")
        self._data["pipelines"].setdefault(pipeline, {"status": "pending", "stages": {}})

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not read etl_status.json — starting fresh")
        return _empty_status()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)


# ============================================================
# HELPERS
# ============================================================

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _empty_status() -> dict:
    return {
        "last_updated": None,
        "pipelines": {
            pipeline: {
                "status":     "pending",
                "started_at": None,
                "ended_at":   None,
                "stages":     {},
            }
            for pipeline in PIPELINES
        },
    }
