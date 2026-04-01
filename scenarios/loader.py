"""Load scenarios from YAML config files."""

from __future__ import annotations

import yaml
from pathlib import Path

from data.generator.scenario import Scenario, ChannelConfig, ControlConfig

SCENARIOS_DIR = Path(__file__).parent


def load_scenario(name: str) -> Scenario:
    path = SCENARIOS_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _parse(cfg)


def load_all_scenarios() -> list[Scenario]:
    scenarios = []
    for path in sorted(SCENARIOS_DIR.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        scenarios.append(_parse(cfg))
    return scenarios


def _parse(cfg: dict) -> Scenario:
    channels = [ChannelConfig(**ch) for ch in cfg.pop("channels")]
    controls_raw = cfg.pop("controls", [])
    controls = [ControlConfig(**c) for c in controls_raw]
    return Scenario(channels=channels, controls=controls, **cfg)
