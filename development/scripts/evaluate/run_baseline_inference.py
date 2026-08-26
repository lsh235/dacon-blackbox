#!/usr/bin/env python3
"""Run selected baseline inference functions and save validated CSV outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from blackbox.inference import predict_stage1, predict_stage2, predict_stage3


PREDICTORS = {
    1: predict_stage1,
    2: predict_stage2,
    3: predict_stage3,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stages", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3])
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    for stage in dict.fromkeys(args.stages):
        started = time.perf_counter()
        frame = PREDICTORS[stage](
            args.data_root / f"stage{stage}",
            args.model_root / f"stage{stage}",
        )
        elapsed = time.perf_counter() - started
        output = args.output_root / f"stage{stage}_predictions.csv"
        frame.to_csv(output, index=False, encoding="utf-8-sig")
        timings[f"stage{stage}_seconds"] = round(elapsed, 3)
        print(f"[OK] Stage {stage}: rows={len(frame)} elapsed={elapsed:.3f}s output={output}")

    timing_path = args.output_root / "timings.json"
    timing_path.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
