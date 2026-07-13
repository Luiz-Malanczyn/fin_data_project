"""Crypto pipeline entrypoint. Triggered by Cloud Scheduler at a higher
cadence than stocks (e.g. hourly), via a Cloud Run Job.
"""
from __future__ import annotations

from src.pipelines.runner import run_pipeline

if __name__ == "__main__":
    run_pipeline("crypto")
