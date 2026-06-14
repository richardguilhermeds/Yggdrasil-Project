"""Relatórios e dashboards por grupo homogêneo."""

from .dashboard import build_dashboard, save_dashboard
from .group_report import (
    group_report,
    group_reports_all,
    group_reports_to_html,
    is_monotonic,
)

__all__ = [
    "group_report",
    "group_reports_all",
    "group_reports_to_html",
    "is_monotonic",
    "build_dashboard",
    "save_dashboard",
]
