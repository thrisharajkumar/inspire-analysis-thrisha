"""
Loads INSPIRE subjects and tags every timestamped reading with its phase (pre/peri/post),
using each subject's real orin_time/orout_time from get_operations_by_orin_time().

This module wraps the existing, working `subject.py` (in ../../../src/) rather than
reimplementing JSON parsing — read_subjects() and the Subject class already do that
correctly. This module adds exactly two things on top: (1) phase tagging, (2) the
full-scale memory mitigations already proven at the 10,942-patient run (category dtypes,
MAX_SUBJECTS_PER_CLASS smoke-testing).
"""

import os
import sys
import random
import numpy as np
import pandas as pd

# Reuse the existing, tested subject-loading code rather than duplicating it.
_ORIGINAL_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src")
sys.path.insert(0, os.path.abspath(_ORIGINAL_SRC))
import subject as subject_module  # noqa: E402  (existing repo module)

from inspire.features.organ_systems import Phase, FEATURE_MAP  # noqa: E402


def load_subjects(subjects_dir, max_per_class=None, seed=42):
    """
    Thin wrapper over the existing read_subjects(), with an optional random cap per
    class for full-scale smoke-testing (config: data.max_subjects_per_class).

    Returns: dict[subject_id -> Subject]
    """
    all_subjects = subject_module.read_subjects(parent_dir=subjects_dir)

    if max_per_class is None:
        return all_subjects

    rng = random.Random(seed)
    died = [s for s in all_subjects.values() if s.died()]
    survived = [s for s in all_subjects.values() if not s.died()]
    rng.shuffle(died)
    rng.shuffle(survived)
    sampled = died[:max_per_class] + survived[:max_per_class]
    print(f"Smoke-test sample: {len(died[:max_per_class])} died, "
          f"{len(survived[:max_per_class])} survived (from {len(died)}/{len(survived)} full)")
    return {s.get_subject_id(): s for s in sampled}


def assign_phase(chart_time_sec, orin_time_sec, orout_time_sec, post_op_window_sec=72 * 3600):
    """
    Tags a single timestamped reading with its phase, given the patient's own
    orin_time (surgery start) and orout_time (surgery end) — all as integer seconds,
    matching the real chart_time/orin_time/orout_time format used throughout subject.py.

    post_op_window_sec: only used if you choose the fixed-duration post-op window
    (config: phases.post_op_window == "72h"). If using "discharge" instead, this needs
    a real discharge timestamp wired in — left as a TODO, since get_operations() doesn't
    currently expose one directly; would need sourcing from get_chart_time_range() or
    the raw JSON's admission/discharge fields if present.
    """
    if chart_time_sec < orin_time_sec:
        return Phase.PRE
    if orin_time_sec <= chart_time_sec <= orout_time_sec:
        return Phase.PERI
    # TODO(post_op_window="discharge"): compare against a real discharge timestamp
    # instead of a fixed window, once that's plumbed through.
    if chart_time_sec <= orout_time_sec + post_op_window_sec:
        return Phase.POST
    return None  # outside the modelled window entirely — drop, don't silently include


def build_phase_tagged_frame(sub: "subject_module.Subject", post_op_window_hours=72):
    """
    Flattens one subject's labs + vitals + ward_vitals into a single long-format
    DataFrame with columns [feature, value, chart_time, phase, system, role, source_table],
    ready to feed the phase-aware timeline encoder (models/encoders.py).

    Field names below match the real record format used throughout subject.py:
    each reading is a dict with string keys 'item_name', 'chart_time' (stringified int,
    seconds), and 'value' (stringified float) — see get_lab()/convert_vitals_to_dictionary().

    Memory note (full-scale): use category dtype for 'feature'/'system' immediately —
    proven to more than halve memory on the long-format ward_vitals table at the
    10,942-patient scale (same mitigation as the existing pipeline).
    """
    op = sub.get_first_operation()  # TODO: confirm first- vs last-operation anchor choice
    # matches the label-recomputation decision already made in dnn_mortality_data_real.py
    if op is None:
        return None
    try:
        orin_time = int(op["orin_time"])
        orout_time = int(op["orout_time"])
    except (KeyError, ValueError, TypeError):
        return None  # can't phase-tag without real, parseable op timestamps

    rows = []
    readings_by_table = {
        "labs": sub.get_labs(),
        "vitals": sub.get_vitals(),
        "ward_vitals": sub.get_ward_vitals(),
    }
    for table_name, readings in readings_by_table.items():
        for r in readings:
            feature = r.get("item_name")
            if feature not in FEATURE_MAP:
                continue  # not one of our 117 mapped parameters — skip rather than error
            system, source_table, phases, role = FEATURE_MAP[feature]
            try:
                chart_time = int(str(r["chart_time"]).strip())
                value = float(str(r["value"]).strip())
            except (KeyError, ValueError, TypeError):
                continue  # malformed reading — drop, don't crash the whole subject
            phase = assign_phase(chart_time, orin_time, orout_time, post_op_window_hours * 3600)
            # NOTE: post_op_window_hours is passed in hours by the caller (matching the
            # config's human-readable "72h"); assign_phase itself works in seconds.
            if phase is None or phase not in phases:
                continue  # e.g. a stray peri-op reading for a pre/post-only feature
            rows.append({
                "feature": feature, "value": value, "chart_time": chart_time,
                "phase": phase.value, "system": system, "role": role.value,
                "source_table": table_name,
            })

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["feature"] = df["feature"].astype("category")
    df["system"] = df["system"].astype("category")
    df["phase"] = df["phase"].astype("category")
    return df
