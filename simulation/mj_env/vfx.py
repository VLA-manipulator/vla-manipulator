"""The imaging layer: one place that turns a rendered frame into what the policy sees.

Every level exists in two imaging conditions -- clean (V0) and underwater
(V1/V2/V3) -- with the same seed and layout, so a paired comparison differs in
nothing but the imaging.  That requirement is why this module exists and why it
is the *only* place the water transform is applied:

* collection and evaluation must call the same function with the same
  parameters, or the two sides see different worlds and every number downstream
  is meaningless;
* the parameters that were actually used are written next to the dataset, so a
  pair can be proven to be a pair rather than assumed to be one.

``WYSD/underwater_vision`` provides the model itself (Jaffe-McGlamey: per-channel
attenuation, backscatter, forward-scatter blur, sensor noise, vignetting).  This
module adds what the SO101 project needs on top: a named ladder loaded from a
calibrated config, the clean side as a first-class rung rather than the absence
of one, and a single ``render_frame`` entry point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

#: Rungs in weakest-to-strongest order.  ``v0_clean`` is the control side of
#: every pair and is implemented by *not* applying the transform at all.
CLEAN_RUNG = "v0_clean"
LADDER: tuple[str, ...] = (CLEAN_RUNG, "v1_light", "v2_medium", "v3_heavy")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "vfx" / "so101_tabletop.json"


class VfxUnavailable(RuntimeError):
    """Raised when a water rung is requested without ``underwater_vision``."""


@dataclass(frozen=True)
class Rung:
    """One imaging condition: how to build it, and what it measured."""

    name: str
    clean: bool
    params: Any = None  # underwater_vision.WaterParams | None
    visibility_m: float | None = None
    hash: str = ""
    measured: Mapping[str, Any] | None = None

    def describe(self) -> str:
        if self.clean:
            return f"{self.name}: clean (no water transform)"
        visibility = "n/a" if self.visibility_m is None else f"{self.visibility_m:.2f} m"
        return f"{self.name}: visibility {visibility}, hash {self.hash[:12]}"

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "mode": "clean" if self.clean else "water",
            "visibility_m": self.visibility_m,
            "hash": self.hash,
            "params": None if self.params is None else self.params.to_dict(),
            "measured": dict(self.measured or {}),
        }


def _import_water():
    try:
        from underwater_vision import WaterParams  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise VfxUnavailable(
            "underwater_vision is not importable; add the WYSD checkout to PYTHONPATH "
            "(e.g. PYTHONPATH=E:\\projects\\mujoco\\WYSD)"
        ) from exc
    return WaterParams


def load_rungs(path: str | Path | None = None) -> dict[str, Rung]:
    """Load the calibrated ladder.  Returns an empty dict if not calibrated yet."""
    source = Path(CONFIG_PATH if path is None else path)
    if not source.is_file():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    WaterParams = _import_water()
    rungs: dict[str, Rung] = {}
    for name, entry in payload.get("rungs", {}).items():
        if entry.get("mode") == "clean":
            rungs[name] = Rung(name=name, clean=True, measured=entry.get("measured"))
            continue
        params = WaterParams.from_dict(entry["params"])
        rungs[name] = Rung(
            name=name,
            clean=False,
            params=params,
            visibility_m=entry.get("visibility_m"),
            hash=entry.get("hash", params.hash()),
            measured=entry.get("measured"),
        )
    return rungs


def get_rung(name: str, path: str | Path | None = None) -> Rung:
    if name == CLEAN_RUNG:
        return Rung(name=CLEAN_RUNG, clean=True)
    rungs = load_rungs(path)
    if name not in rungs:
        raise KeyError(
            f"unknown imaging rung {name!r}; calibrated rungs: "
            f"{sorted(rungs) or 'none -- run mj_env.scripts.water_probe first'}"
        )
    return rungs[name]


class Imaging:
    """Renders frames for one imaging rung, clean or underwater.

    Use one instance per episode: the renderer holds the depth buffer and the
    parameters, and ``params_hash`` is what proves two episodes were imaged the
    same way.
    """

    def __init__(
        self,
        model: Any,
        width: int,
        height: int,
        rung: Rung | str = CLEAN_RUNG,
        *,
        seed: int | None = None,
        use_depth: bool = True,
    ) -> None:
        self.rung = get_rung(rung) if isinstance(rung, str) else rung
        self.width = width
        self.height = height
        self._model = model
        self._renderer = None
        self._raw = None
        if not self.rung.clean:
            from underwater_vision import WaterRenderer  # noqa: PLC0415

            self._raw = WaterRenderer(
                model,
                width,
                height,
                params=self.rung.params,
                seed=seed,
                use_depth=use_depth,
            )

    # -- identity ------------------------------------------------------------
    @property
    def clean(self) -> bool:
        return self.rung.clean

    @property
    def params_hash(self) -> str:
        """``""`` for the clean rung; the water hash otherwise.

        Written into every episode record.  Two episodes with the same
        ``layout_seed`` and different ``vfx_hash`` are a valid pair; two with the
        same ``vfx_hash`` are not a comparison at all.
        """
        return "" if self.rung.clean else self.rung.hash

    def to_json(self) -> dict:
        return self.rung.to_json()

    # -- rendering -----------------------------------------------------------
    def render(self, data: Any, camera: Any) -> np.ndarray:
        """The one entry point.  Collection and evaluation both call this."""
        if self.rung.clean:
            return self._clean(data, camera)
        return self._raw.render(data, camera=camera)

    def _clean(self, data: Any, camera: Any) -> np.ndarray:
        import mujoco  # noqa: PLC0415

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, self.height, self.width)
        self._renderer.update_scene(data, camera=camera)
        return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

    def close(self) -> None:
        if self._raw is not None:
            self._raw.close()
            self._raw = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "Imaging":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_config(payload: dict, path: str | Path | None = None) -> Path:
    target = Path(CONFIG_PATH if path is None else path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target