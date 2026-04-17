from __future__ import annotations

from typing import Any

import logging as _logging

try:
    from codecarbon import EmissionsTracker as _CodeCarbonTracker

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

# Silence codecarbon's own noisy INFO/WARNING output — our sweep logger handles progress.
_logging.getLogger("codecarbon").setLevel(_logging.ERROR)


class EmissionsTracker:
    """Context manager wrapping codecarbon EmissionsTracker.

    Degrades gracefully when codecarbon is not installed — the block still
    runs and ``result`` returns None.

    Usage::

        with EmissionsTracker() as tracker:
            do_work()
        print(tracker.result)  # dict or None
    """

    def __init__(self, **kwargs: Any) -> None:
        self._tracker = None
        self._result: dict[str, Any] | None = None
        if _AVAILABLE:
            self._tracker = _CodeCarbonTracker(
                save_to_file=False,
                logging_logger=None,
                log_level="error",
                **kwargs,
            )

    def __enter__(self) -> "EmissionsTracker":
        if self._tracker is not None:
            self._tracker.start()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._tracker is None:
            return
        emissions_kg = self._tracker.stop()
        data = self._tracker.final_emissions_data
        self._result = {
            "emissions_kg_co2": emissions_kg,
            "energy_consumed_kwh": getattr(data, "energy_consumed", None),
            "duration_s": getattr(data, "duration", None),
            "cpu_power_w": getattr(data, "cpu_power", None),
            "gpu_power_w": getattr(data, "gpu_power", None),
            "ram_power_w": getattr(data, "ram_power", None),
            "country_name": getattr(data, "country_name", None),
            "cloud_provider": getattr(data, "cloud_provider", None),
            "codecarbon_version": getattr(data, "codecarbon_version", None),
        }

    @property
    def result(self) -> dict[str, Any] | None:
        """Emission data dict, or None if codecarbon is unavailable."""
        return self._result
