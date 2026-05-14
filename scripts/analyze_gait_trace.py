#!/usr/bin/env python3
"""Summarize a hexapod playback trace into gait-quality metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TRIPOD_A = ("FL_tibia_1", "MR_tibia_1", "BL_tibia_1")
TRIPOD_B = ("FR_tibia_1", "ML_tibia_1", "BR_tibia_1")


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _contact_transitions(values: list[int]) -> int:
    return sum(1 for prev, cur in zip(values, values[1:]) if prev != cur)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _foot_names(fieldnames: list[str]) -> list[str]:
    suffix = "_contact"
    return sorted(name[: -len(suffix)] for name in fieldnames if name.endswith(suffix))


def _joint_summary(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    rows = _load_rows(path)
    by_joint: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_joint.setdefault(row["joint_name"], []).append(row)
    joints = {}
    coxa_ranges = []
    coxa_saturation_rates = []
    for joint_name, joint_rows in by_joint.items():
        positions = [_float(row, "joint_pos") for row in joint_rows]
        velocities = [abs(_float(row, "joint_vel")) for row in joint_rows]
        margins = [_float(row, "action_bound_margin") for row in joint_rows]
        target_lower_margins = [_float(row, "target_lower_margin", math.inf) for row in joint_rows]
        target_upper_margins = [_float(row, "target_upper_margin", math.inf) for row in joint_rows]
        action_saturation_rate = _mean([1.0 if margin < 0.05 else 0.0 for margin in margins])
        limit_target_rate = _mean(
            [1.0 if min(lower, upper) < 0.05 else 0.0 for lower, upper in zip(target_lower_margins, target_upper_margins)]
        )
        pos_range = max(positions) - min(positions) if positions else 0.0
        joints[joint_name] = {
            "position_range_rad": pos_range,
            "mean_abs_velocity_rad_s": _mean(velocities),
            "action_saturation_rate": action_saturation_rate,
            "target_near_limit_rate": limit_target_rate,
        }
        if "coxa" in joint_name:
            coxa_ranges.append(pos_range)
            coxa_saturation_rates.append(action_saturation_rate)
    return {
        "coxa_position_range_mean_rad": _mean(coxa_ranges),
        "coxa_position_range_stdev_rad": _stdev(coxa_ranges),
        "coxa_action_saturation_rate_mean": _mean(coxa_saturation_rates),
        "joints": joints,
    }


def analyze(trace_csv: Path, joint_trace_csv: Path | None = None) -> dict:
    rows = _load_rows(trace_csv)
    if not rows:
        raise RuntimeError(f"No trace rows in {trace_csv}")
    fieldnames = list(rows[0].keys())
    feet = _foot_names(fieldnames)
    time_s = [_float(row, "time_s") for row in rows]
    duration = max(time_s[-1] - time_s[0], 1.0e-6)
    first = rows[0]
    last = rows[-1]

    contacts = {foot: [int(float(row[f"{foot}_contact"])) for row in rows] for foot in feet}
    drag = {foot: [_float(row, f"{foot}_drag_speed_w") for row in rows] for foot in feet}
    forces = {foot: [_float(row, f"{foot}_force") for row in rows] for foot in feet}

    tripod_a = [foot for foot in TRIPOD_A if foot in contacts]
    tripod_b = [foot for foot in TRIPOD_B if foot in contacts]
    phase_values = []
    all_contact_counts = []
    tripod_like_steps = 0
    all_six_contact_steps = 0
    too_few_contact_steps = 0
    for idx in range(len(rows)):
        a_contacts = sum(contacts[foot][idx] for foot in tripod_a)
        b_contacts = sum(contacts[foot][idx] for foot in tripod_b)
        total_contacts = a_contacts + b_contacts
        all_contact_counts.append(total_contacts)
        a_fraction = a_contacts / len(tripod_a) if tripod_a else 0.0
        b_fraction = b_contacts / len(tripod_b) if tripod_b else 0.0
        phase = a_fraction - b_fraction
        phase_values.append(phase)
        if (a_contacts >= 2 and b_contacts <= 1) or (b_contacts >= 2 and a_contacts <= 1):
            tripod_like_steps += 1
        if total_contacts >= 6:
            all_six_contact_steps += 1
        if total_contacts < 3:
            too_few_contact_steps += 1
    signs = [1 if value > 0.2 else -1 if value < -0.2 else 0 for value in phase_values]
    clean_signs = [sign for sign in signs if sign != 0]
    tripod_switches = _contact_transitions(clean_signs)

    foot_summaries = {}
    for foot in feet:
        contact_values = contacts[foot]
        foot_summaries[foot] = {
            "contact_duty": _mean([float(value) for value in contact_values]),
            "contact_transitions": _contact_transitions(contact_values),
            "mean_drag_speed_mps": _mean(drag[foot]),
            "max_drag_speed_mps": max(drag[foot]) if drag[foot] else 0.0,
            "mean_contact_force_n": _mean(forces[foot]),
        }

    dx = _float(last, "x_w") - _float(first, "x_w")
    dy = _float(last, "y_w") - _float(first, "y_w")
    yaw_values = [_float(row, "yaw_w") for row in rows if row.get("yaw_w", "") != ""]
    yaw_delta = yaw_values[-1] - yaw_values[0] if len(yaw_values) >= 2 else 0.0
    while yaw_delta > math.pi:
        yaw_delta -= 2.0 * math.pi
    while yaw_delta < -math.pi:
        yaw_delta += 2.0 * math.pi

    result = {
        "trace_csv": str(trace_csv),
        "duration_s": duration,
        "forward_displacement_m": dy,
        "lateral_displacement_m": dx,
        "forward_speed_mps": dy / duration,
        "abs_lateral_drift_per_forward": abs(dx) / max(abs(dy), 1.0e-6),
        "yaw_delta_rad": yaw_delta,
        "mean_abs_vx_w_mps": _mean([abs(_float(row, "vx_w")) for row in rows]),
        "mean_vy_w_mps": _mean([_float(row, "vy_w") for row in rows]),
        "mean_flat_orientation": _mean([_float(row, "flat_orientation") for row in rows]),
        "mean_forward_tilt": _mean([_float(row, "forward_tilt") for row in rows]),
        "mean_base_height_m": _mean([_float(row, "z_w") for row in rows]),
        "min_base_height_m": min(_float(row, "z_w") for row in rows),
        "mean_abs_action": _mean([_float(row, "action_mean_abs") for row in rows]),
        "mean_action_max_abs": _mean([_float(row, "action_max_abs") for row in rows]),
        "mean_abs_action_delta": _mean([_float(row, "action_delta_mean_abs") for row in rows]),
        "mean_abs_joint_velocity": _mean([_float(row, "joint_vel_mean_abs") for row in rows]),
        "mean_num_feet_in_contact": _mean([float(value) for value in all_contact_counts]),
        "tripod_like_fraction": tripod_like_steps / len(rows),
        "tripod_phase_separation_mean": _mean([abs(value) for value in phase_values]),
        "tripod_phase_switches": tripod_switches,
        "all_six_contact_fraction": all_six_contact_steps / len(rows),
        "too_few_contact_fraction": too_few_contact_steps / len(rows),
        "feet": foot_summaries,
        "joint_summary": _joint_summary(joint_trace_csv),
    }
    return result


def write_markdown(result: dict, out_path: Path) -> None:
    lines = [
        "# Gait Trace Summary",
        "",
        f"Trace: `{result['trace_csv']}`",
        "",
        "## Motion",
        f"- Duration: {result['duration_s']:.2f} s",
        f"- Forward displacement: {result['forward_displacement_m']:.3f} m",
        f"- Forward speed: {result['forward_speed_mps']:.3f} m/s",
        f"- Lateral displacement: {result['lateral_displacement_m']:.3f} m",
        f"- Lateral drift / forward: {result['abs_lateral_drift_per_forward']:.3f}",
        f"- Yaw delta: {result['yaw_delta_rad']:.3f} rad",
        f"- Mean flat orientation: {result['mean_flat_orientation']:.4f}",
        f"- Mean forward tilt: {result['mean_forward_tilt']:.4f}",
        f"- Mean/min base height: {result['mean_base_height_m']:.3f} / {result['min_base_height_m']:.3f} m",
        "",
        "## Gait",
        f"- Mean feet in contact: {result['mean_num_feet_in_contact']:.2f}",
        f"- Tripod-like fraction: {result['tripod_like_fraction']:.3f}",
        f"- Tripod phase separation mean: {result['tripod_phase_separation_mean']:.3f}",
        f"- Tripod phase switches: {result['tripod_phase_switches']}",
        f"- All-six-contact fraction: {result['all_six_contact_fraction']:.3f}",
        f"- Too-few-contact fraction: {result['too_few_contact_fraction']:.3f}",
        "",
        "## Smoothness",
        f"- Mean abs action: {result['mean_abs_action']:.3f}",
        f"- Mean max abs action: {result['mean_action_max_abs']:.3f}",
        f"- Mean abs action delta: {result['mean_abs_action_delta']:.3f}",
        f"- Mean abs joint velocity: {result['mean_abs_joint_velocity']:.3f} rad/s",
        "",
        "## Feet",
        "| Foot | Duty | Transitions | Mean Drag | Max Drag | Mean Force |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for foot, values in sorted(result["feet"].items()):
        lines.append(
            f"| {foot} | {values['contact_duty']:.3f} | {values['contact_transitions']} | "
            f"{values['mean_drag_speed_mps']:.3f} | {values['max_drag_speed_mps']:.3f} | "
            f"{values['mean_contact_force_n']:.1f} |"
        )
    joint_summary = result.get("joint_summary") or {}
    if joint_summary:
        lines.extend(
            [
                "",
                "## Coxa Joints",
                f"- Mean coxa range: {joint_summary['coxa_position_range_mean_rad']:.3f} rad",
                f"- Coxa range stdev: {joint_summary['coxa_position_range_stdev_rad']:.3f} rad",
                f"- Mean coxa action saturation rate: {joint_summary['coxa_action_saturation_rate_mean']:.3f}",
            ]
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_csv", type=Path, required=True)
    parser.add_argument("--joint_trace_csv", type=Path, default=None)
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--out_md", type=Path, default=None)
    args = parser.parse_args()

    result = analyze(args.trace_csv, args.joint_trace_csv)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(result, args.out_md)
    if args.out_json is None and args.out_md is None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
