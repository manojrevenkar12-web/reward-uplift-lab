"""Command-line entry point: ``uplift-lab simulate`` and ``uplift-lab criteo``.

Each command writes ``results.json`` (every number), ``REPORT.md`` (rendered from that
JSON) and ``figures/*.png`` to the output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from uplift_lab.data import load_criteo
from uplift_lab.pipeline import StudyConfig, run_criteo_study, run_simulation_study
from uplift_lab.report import render_criteo_report, render_simulation_report


def _to_json(value: Any) -> Any:
    """Convert numpy scalars/arrays and non-finite floats into strict JSON values."""
    if isinstance(value, dict):
        return {str(k): _to_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_json(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write(out_dir: Path, results: dict[str, Any], report: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(_to_json(results), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uplift-lab",
        description="Experimentation and uplift-modelling study for reward targeting.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="run the full study on simulated data")
    sim.add_argument("--n-users", type=int, default=StudyConfig.n_users)
    sim.add_argument("--seed", type=int, default=StudyConfig.seed)
    sim.add_argument("--aa-splits", type=int, default=StudyConfig.aa_splits)
    sim.add_argument("--peeking-sims", type=int, default=StudyConfig.peeking_sims)
    sim.add_argument("--out", type=Path, default=Path("reports/simulation"))

    cri = sub.add_parser("criteo", help="run the observable study on the Criteo Uplift data")
    cri.add_argument("path", type=Path, help="path to criteo-research-uplift-v2.1.csv(.gz)")
    cri.add_argument("--outcome", choices=("visit", "conversion"), default="visit")
    cri.add_argument("--sample-frac", type=float, default=None)
    cri.add_argument("--seed", type=int, default=0)
    cri.add_argument("--out", type=Path, default=Path("reports/criteo"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "simulate":
        config = StudyConfig(
            n_users=args.n_users,
            seed=args.seed,
            aa_splits=args.aa_splits,
            peeking_sims=args.peeking_sims,
        )
        results = run_simulation_study(config, args.out)
        _write(args.out, results, render_simulation_report(results))
    else:
        df = load_criteo(args.path, sample_frac=args.sample_frac, seed=args.seed)
        results = run_criteo_study(df, args.out, outcome=args.outcome, seed=args.seed)
        _write(args.out, results, render_criteo_report(results))
    print(f"Wrote {args.out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
