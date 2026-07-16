"""
ETL status reader — Streamlit side.

Reads data/etl_status.json produced by the pipeline writer (src/etl_status.py)
and returns typed, display-ready structures for the ETL pages.

Usage:
    from streamlit.utils.etl_status import load_etl_status, get_pipeline_summary

    status = load_etl_status()
    summary = get_pipeline_summary(status, "bronze")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

STATUS_PATH = Path("data/etl_status.json")

PIPELINE_LABELS = {
    "bronze": "Bronze",
    "silver": "Silver",
    "gold":   "Gold",
}

STAGE_LABELS = {
    "countries":         "Countries",
    "products":          "Products",
    "trade_flow":        "Trade flows",
    "indicators":        "Indicators",
    "indicators_values": "Indicator values",
    "trade_data":        "Trade data",
}

STATUS_COLORS = {
    "done":    "success",
    "running": "warning",
    "failed":  "error",
    "skipped": "secondary",
    "pending": "secondary",
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class StageStatus:
    name:        str
    label:       str
    status:      str
    rows:        int
    duration_s:  float
    updated_at:  str | None
    checkpoints: dict | None = None
    error:       str | None  = None

    @property
    def duration_fmt(self) -> str:
        """Human-readable duration."""
        s = self.duration_s
        if s < 60:
            return f"{s:.0f}s"
        m, sec = divmod(s, 60)
        return f"{int(m)}m {int(sec):02d}s"

    @property
    def rows_fmt(self) -> str:
        """Human-readable row count."""
        if self.rows >= 1_000_000:
            return f"{self.rows / 1_000_000:.1f}M"
        if self.rows >= 1_000:
            return f"{self.rows / 1_000:.0f}K"
        return str(self.rows)

    @property
    def color(self) -> str:
        return STATUS_COLORS.get(self.status, "secondary")


@dataclass
class PipelineSummary:
    name:       str
    label:      str
    status:     str
    started_at: str | None
    ended_at:   str | None
    stages:     list[StageStatus] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(s.rows for s in self.stages)

    @property
    def total_duration_s(self) -> float:
        return sum(s.duration_s for s in self.stages)

    @property
    def total_duration_fmt(self) -> str:
        s = self.total_duration_s
        if s < 60:
            return f"{s:.0f}s"
        m, sec = divmod(s, 60)
        if m < 60:
            return f"{int(m)}m {int(sec):02d}s"
        h, min_ = divmod(m, 60)
        return f"{int(h)}h {int(min_):02d}m"

    @property
    def color(self) -> str:
        return STATUS_COLORS.get(self.status, "secondary")

    @property
    def is_done(self) -> bool:
        return self.status == "done"

    @property
    def has_run(self) -> bool:
        return self.status not in ("pending", None)


@dataclass
class ETLStatus:
    last_updated: str | None
    pipelines:    dict[str, PipelineSummary]
    raw:          dict

    @property
    def has_data(self) -> bool:
        return any(p.has_run for p in self.pipelines.values())

    @property
    def total_rows_gold(self) -> int:
        gold = self.pipelines.get("gold")
        return gold.total_rows if gold else 0

    @property
    def all_done(self) -> bool:
        return all(p.is_done for p in self.pipelines.values())


# ============================================================
# LOADER
# ============================================================

def load_etl_status() -> ETLStatus:
    """
    Load and parse etl_status.json.
    Returns an ETLStatus with pending pipelines if the file doesn't exist.
    """
    raw = _read_raw()
    pipelines = {}

    for name in ("bronze", "silver", "gold"):
        p_raw = raw.get("pipelines", {}).get(name, {})
        stages = _parse_stages(p_raw.get("stages", {}))
        pipelines[name] = PipelineSummary(
            name       = name,
            label      = PIPELINE_LABELS[name],
            status     = p_raw.get("status", "pending"),
            started_at = p_raw.get("started_at"),
            ended_at   = p_raw.get("ended_at"),
            stages     = stages,
        )

    return ETLStatus(
        last_updated = raw.get("last_updated"),
        pipelines    = pipelines,
        raw          = raw,
    )


def _parse_stages(stages_raw: dict) -> list[StageStatus]:
    result = []
    for name, s in stages_raw.items():
        result.append(StageStatus(
            name        = name,
            label       = STAGE_LABELS.get(name, name.replace("_", " ").title()),
            status      = s.get("status", "pending"),
            rows        = s.get("rows", 0),
            duration_s  = s.get("duration_s", 0.0),
            updated_at  = s.get("updated_at"),
            checkpoints = s.get("checkpoints"),
            error       = s.get("error"),
        ))
    # Respect canonical order
    order = list(STAGE_LABELS.keys())
    result.sort(key=lambda s: order.index(s.name) if s.name in order else 99)
    return result


def _read_raw() -> dict:
    if not STATUS_PATH.exists():
        return {}
    try:
        with open(STATUS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
