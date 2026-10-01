"""Uvicorn logging for Railway (webapp/log_config.json, passed with
--log-config in the Dockerfile). Railway marks every stderr line as an
error, and uvicorn's default sends its plain INFO lines ("Started server
process", "Application startup complete") to stderr -- so a healthy deploy
looked red. INFO goes to stdout here; warnings and real errors (tracebacks)
still go to stderr."""
import logging


class BelowWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING
