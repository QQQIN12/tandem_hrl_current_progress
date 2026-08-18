"""Extract rollout frames at physical interaction and safety events."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path


PREFIX = "Command/locomotion/TACTIC/"
OBJECT_COUNT = 6


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--frame_offset",
        type=int,
        default=1,
        help="Video frame index minus trace global_step.",
    )
    return parser.parse_args()


def _number(row, key, default=math.nan):
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _max_for_suffix(row, suffix):
    values = [
        _number(row, key)
        for key in row
        if key.startswith("Diagnostic/TACTIC/object_")
        and key.endswith(suffix)
    ]
    values = [value for value in values if math.isfinite(value)]
    return max(values) if values else math.nan


def _object_value(row, object_id, suffix):
    return _number(
        row,
        "Diagnostic/TACTIC/object_{}/{}".format(object_id, suffix),
    )


def _first(rows, predicate, start=0):
    for index in range(max(0, start), len(rows)):
        if predicate(rows[index]):
            return index
    return None


def _minimum(rows, key, start=0):
    candidates = [
        (index, _number(row, key))
        for index, row in enumerate(rows[start:], start=start)
    ]
    candidates = [
        item for item in candidates if math.isfinite(item[1])
    ]
    return min(candidates, key=lambda item: item[1])[0] if candidates else None


def _event_indices(rows, object_id):
    contact = _first(
        rows,
        lambda row: _object_value(
            row, object_id, "object_contact"
        )
        >= 0.5,
    )
    approach_stop = contact if contact is not None else len(rows)
    object_tcp_key = "Diagnostic/TACTIC/object_{}/tcp_distance".format(
        object_id
    )
    approach = _minimum(rows[:approach_stop], object_tcp_key)
    if approach is None:
        approach = _minimum(
            rows[:approach_stop],
            PREFIX + "tcp_object_distance",
        )
    lift = _first(
        rows,
        lambda row: _object_value(
            row, object_id, "object_lift_memory"
        )
        >= 0.20,
        start=contact or 0,
    )
    carry = _first(
        rows,
        lambda row: _object_value(
            row, object_id, "object_carrying"
        )
        >= 0.5,
        start=lift or contact or 0,
    )
    transport = _first(
        rows,
        lambda row: _object_value(
            row, object_id, "object_transport_memory"
        )
        >= 0.5,
        start=carry or lift or 0,
    )
    near_target = _first(
        rows,
        lambda row: (
            _number(row, PREFIX + "object_target_distance") <= 0.30
            and int(
                _number(
                    row,
                    PREFIX + "effective_object_id",
                    default=-1,
                )
            )
            == object_id
            and _object_value(row, object_id, "object_carrying")
            >= 0.5
        ),
        start=carry or 0,
    )
    release = None
    if carry is not None:
        for index in range(carry + 1, len(rows)):
            previous_carry = _object_value(
                rows[index - 1], object_id, "object_carrying"
            )
            current_carry = _object_value(
                rows[index], object_id, "object_carrying"
            )
            closure = _number(
                rows[index], PREFIX + "gripper_closure"
            )
            if (
                previous_carry >= 0.5
                and current_carry < 0.5
                and closure < 0.35
            ):
                release = index
                break
    completion = _first(
        rows,
        lambda row: (
            _object_value(row, object_id, "object_completion") >= 0.5
            or _number(row, PREFIX + "mission_completion") >= 0.999
        ),
        start=release or transport or 0,
    )
    max_tilt = max(
        range(len(rows)),
        key=lambda index: _number(
            rows[index], PREFIX + "base_tilt", default=-math.inf
        ),
    )
    tilt_failure = _first(
        rows,
        lambda row: _number(row, "Termination/tilt", 0.0) >= 0.5,
    )
    return (
        ("approach", approach),
        ("contact", contact),
        ("lift", lift),
        ("carry", carry),
        ("transport", transport),
        ("near_target", near_target),
        ("release", release),
        ("completion", completion),
        ("max_tilt", max_tilt),
        ("tilt_failure", tilt_failure),
    )


def _episode_object_score(rows, object_id):
    def event_max(name):
        values = [
            _object_value(row, object_id, name) for row in rows
        ]
        values = [value for value in values if math.isfinite(value)]
        return max(values) if values else 0.0

    return (
        event_max("object_completion"),
        event_max("object_transport_memory"),
        event_max("object_carrying"),
        event_max("object_lift_memory"),
        event_max("object_contact"),
        len(rows),
    )


def _select_episode(rows):
    grouped = defaultdict(list)
    for row in rows:
        episode_id = int(_number(row, "episode_id", 0))
        grouped[episode_id].append(row)
    candidates = [
        (
            _episode_object_score(episode_rows, object_id),
            -episode_id,
            -object_id,
            episode_id,
            episode_rows,
            object_id,
        )
        for episode_id, episode_rows in grouped.items()
        for object_id in range(OBJECT_COUNT)
    ]
    _, _, _, episode_id, episode_rows, object_id = max(
        candidates,
        key=lambda item: item[:3],
    )
    return episode_id, episode_rows, object_id


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(ffmpeg, video, frame_index, output):
    expression = "select=eq(n\\,{})".format(frame_index)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video),
            "-vf",
            expression,
            "-frames:v",
            "1",
            "-vsync",
            "vfr",
            str(output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main():
    args = _parse_args()
    if not args.trace.is_file():
        raise FileNotFoundError(args.trace)
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    with args.trace.open(
        "r", encoding="utf-8", newline=""
    ) as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise RuntimeError("Trace is empty")

    (
        selected_episode_id,
        episode_rows,
        selected_object_id,
    ) = _select_episode(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for order, (event, row_index) in enumerate(
        _event_indices(episode_rows, selected_object_id), 1
    ):
        item = {
            "order": order,
            "event": event,
            "available": int(row_index is not None),
            "selected_episode_id": selected_episode_id,
            "selected_object_id": selected_object_id,
        }
        if row_index is not None:
            row = episode_rows[row_index]
            global_step = int(_number(row, "global_step", row_index))
            frame_index = max(0, global_step + args.frame_offset)
            output = args.output_dir / (
                "{:02d}_{}_step{:05d}.png".format(
                    order, event, global_step
                )
            )
            _extract(args.ffmpeg, args.video, frame_index, output)
            item.update(
                {
                    "trace_row": row_index,
                    "global_step": global_step,
                    "frame_index": frame_index,
                    "sim_time_s": _number(row, "sim_time_s"),
                    "episode_id": int(
                        _number(row, "episode_id", -1)
                    ),
                    "task_id": int(
                        _number(row, PREFIX + "task_id", -1)
                    ),
                    "skill_id": int(
                        _number(row, PREFIX + "skill_id", -1)
                    ),
                    "effective_object_id": int(
                        _number(
                            row,
                            PREFIX + "effective_object_id",
                            -1,
                        )
                    ),
                    "reward": _number(row, "reward"),
                    "base_tilt": _number(
                        row, PREFIX + "base_tilt"
                    ),
                    "cbf_margin": _number(
                        row, PREFIX + "cbf_margin"
                    ),
                    "object_target_distance": _number(
                        row, PREFIX + "object_target_distance"
                    ),
                    "image": output.name,
                }
            )
        metadata.append(item)

    metadata_path = args.output_dir / "event_frame_alignment.csv"
    fields = []
    for row in metadata:
        for key in row:
            if key not in fields:
                fields.append(key)
    with metadata_path.open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata)

    manifest = {
        "trace": str(args.trace.resolve()),
        "trace_sha256": _sha256(args.trace),
        "video": str(args.video.resolve()),
        "video_sha256": _sha256(args.video),
        "frame_offset": args.frame_offset,
        "selected_episode_id": selected_episode_id,
        "selected_object_id": selected_object_id,
        "episode_score": list(
            _episode_object_score(
                episode_rows,
                selected_object_id,
            )
        ),
        "alignment_csv": metadata_path.name,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print("keyframe_dir={}".format(args.output_dir))


if __name__ == "__main__":
    main()
