"""Export TensorBoard scalar events to a lossless long-form CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tags",
        default="",
        help="Optional regular expression matched against scalar tag names.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if not args.log_dir.is_dir():
        raise FileNotFoundError(args.log_dir)
    tag_pattern = re.compile(args.tags) if args.tags else None
    accumulator = EventAccumulator(
        str(args.log_dir),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if tag_pattern is not None:
        tags = [tag for tag in tags if tag_pattern.search(tag)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=("tag", "step", "wall_time", "value"),
        )
        writer.writeheader()
        for tag in sorted(tags):
            for event in accumulator.Scalars(tag):
                writer.writerow(
                    {
                        "tag": tag,
                        "step": event.step,
                        "wall_time": "{:.9f}".format(event.wall_time),
                        "value": "{:.12g}".format(event.value),
                    }
                )
    print(
        "scalar_csv={} tags={}".format(
            args.output,
            len(tags),
        )
    )


if __name__ == "__main__":
    main()
