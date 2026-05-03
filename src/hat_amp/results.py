"""Percolation result persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class PercolationResults(BaseModel):
    """Raw percolation trial arrays and metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tiling_type: str
    seed: int | None
    trials: int
    L_values: np.ndarray
    raw_SI: list[np.ndarray]
    raw_SU: list[np.ndarray]
    raw_BI: list[np.ndarray] | None = None
    raw_BU: list[np.ndarray] | None = None
    extra_meta: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    @property
    def has_bond(self) -> bool:
        """Return whether bond-percolation trial arrays are present."""
        return self.raw_BI is not None and self.raw_BU is not None

    @property
    def n_L(self) -> int:
        """Return the number of sampled frame sizes."""
        return int(len(self.L_values))

    def means_stds(self) -> tuple[np.ndarray, ...]:
        """Return means and standard deviations for site and bond arrays."""
        site_i_mean, site_i_std = _means_stds(self.raw_SI)
        site_u_mean, site_u_std = _means_stds(self.raw_SU)
        if self.raw_BI is None or self.raw_BU is None:
            empty = np.array([], dtype=np.float64)
            return (
                site_i_mean,
                site_i_std,
                site_u_mean,
                site_u_std,
                empty,
                empty,
                empty,
                empty,
            )

        bond_i_mean, bond_i_std = _means_stds(self.raw_BI)
        bond_u_mean, bond_u_std = _means_stds(self.raw_BU)
        return (
            site_i_mean,
            site_i_std,
            site_u_mean,
            site_u_std,
            bond_i_mean,
            bond_i_std,
            bond_u_mean,
            bond_u_std,
        )

    def save(self, path: str | Path) -> Path:
        """Save results to a `.npz` archive."""
        output = Path(path)
        payload: dict[str, Any] = {
            "meta_tiling_type": self.tiling_type,
            "meta_seed": -1 if self.seed is None else self.seed,
            "meta_seed_is_none": self.seed is None,
            "meta_timestamp": self.timestamp,
            "meta_trials": self.trials,
            "L_values": np.asarray(self.L_values, dtype=np.float64),
        }

        for key, value in self.extra_meta.items():
            payload[f"meta_{key}"] = value

        _store_raw(payload, "raw_SI", self.raw_SI)
        _store_raw(payload, "raw_SU", self.raw_SU)
        if self.raw_BI is not None:
            _store_raw(payload, "raw_BI", self.raw_BI)
        if self.raw_BU is not None:
            _store_raw(payload, "raw_BU", self.raw_BU)

        np.savez(output, **payload)
        return output

    @classmethod
    def load(cls, path: str | Path) -> "PercolationResults":
        """Load results from a `.npz` archive."""
        with np.load(Path(path), allow_pickle=True) as data:
            files = set(data.files)
            seed_is_none = bool(data["meta_seed_is_none"]) if "meta_seed_is_none" in files else False
            seed = None if seed_is_none else int(data["meta_seed"])
            extra_meta: dict[str, Any] = {}
            reserved = {
                "meta_tiling_type",
                "meta_seed",
                "meta_seed_is_none",
                "meta_timestamp",
                "meta_trials",
            }
            for key in files:
                if key.startswith("meta_") and key not in reserved:
                    value = data[key]
                    extra_meta[key.removeprefix("meta_")] = value.item() if value.shape == () else value

            return cls(
                tiling_type=str(data["meta_tiling_type"]),
                seed=seed,
                timestamp=str(data["meta_timestamp"]),
                trials=int(data["meta_trials"]),
                L_values=np.asarray(data["L_values"], dtype=np.float64),
                raw_SI=_load_raw(data, "raw_SI"),
                raw_SU=_load_raw(data, "raw_SU"),
                raw_BI=_load_raw(data, "raw_BI") if _has_raw(data, "raw_BI") else None,
                raw_BU=_load_raw(data, "raw_BU") if _has_raw(data, "raw_BU") else None,
                extra_meta=extra_meta,
            )


def _means_stds(raw: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    means = np.array([np.mean(values) for values in raw], dtype=np.float64)
    stds = np.array([np.std(values) for values in raw], dtype=np.float64)
    return means, stds


def _store_raw(payload: dict[str, Any], prefix: str, values: list[np.ndarray]) -> None:
    payload[f"{prefix}_count"] = len(values)
    for i, array in enumerate(values):
        payload[f"{prefix}_{i}"] = np.asarray(array, dtype=np.float64)


def _has_raw(data: np.lib.npyio.NpzFile, prefix: str) -> bool:
    return f"{prefix}_count" in data.files


def _load_raw(data: np.lib.npyio.NpzFile, prefix: str) -> list[np.ndarray]:
    count = int(data[f"{prefix}_count"])
    return [np.asarray(data[f"{prefix}_{i}"], dtype=np.float64) for i in range(count)]
