#!/usr/bin/env python3
"""Standalone offline reanalysis for the paced-r1 Frigate-browser latency run.

Reads ONLY saved artifacts (no camera/HA/device/browser access):
  - receipts/frigate-browser-marker-pairs-paced-r1-20261018.json  (raw capture,
    preserved untouched together with its INVALID stats)
  - receipts/browser-identity-marker-20261018/transition-schedule.json
    (frozen ground-truth identity schedule, SHA recorded below)

Why this exists: the capture's own matcher rejected all 19+19 edges
(stats.valid=false, "insufficient-or-ambiguous-transition-identity") because its
8-interval signature keys quantize each interval to a 160 ms bucket; the real
projected marker has correlated per-edge timing skew (display/camera/threshold)
of up to ~480 ms, flipping 13 of 18 buckets between the two sides. This script
re-establishes identity NUMERICALLY and INDEPENDENTLY per side:

  * each side is fitted alone against the frozen schedule by enumerating ALL
    strictly increasing ordinal embeddings whose bridged intervals match the
    measured intervals within TAU ms (never assumes same index between sides);
  * identity is accepted only if each side has EXACTLY ONE embedding
    (0 = no fit, >1 = ambiguous — the periodic failure mode);
  * browser/direct pairs are then joined on the independently fitted schedule
    ordinal, and browser-minus-direct deltas + bracket bounds are reported.

TAU = 500 ms: the schedule's interval values are multiples of 1000 with a
minimum separation of 1000 ms, so any bridged value within 500 ms of a candidate
is closer to it than to any other single value; multi-interval bridges are
disambiguated by the whole-sequence uniqueness requirement, not per value.

Negative/positive controls are executed first on synthetic fixtures built from
the same schedule (positive +350 lag, delayed +3350 lag, dropped edge, periodic
ambiguous) — the real reanalysis only reports if every control behaves.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

RECEIPTS = Path("/home/kom/proj/cat-tv-play/receipts")
RAW = RECEIPTS / "frigate-browser-marker-pairs-paced-r1-20261018.json"
SCHED = RECEIPTS / "browser-identity-marker-20261018" / "transition-schedule.json"
OUT = RECEIPTS / "frigate-browser-marker-pairs-paced-r1-reanalysis-20261018.json"
TAU = 500  # ms; see module docstring
MIN_IDENTITY_EDGES = 4  # frozen sampler contract
MIN_CONTRAST = 20.0  # frozen sampler contract


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nr(values: list[float], q: float):
    if not values:
        return None
    b = sorted(values)
    return b[max(0, math.ceil(q * len(b)) - 1)]


def calibrate(lumas: list[float]):
    if not lumas:
        return None
    lo, hi = nr(lumas, 0.1), nr(lumas, 0.9)
    if lo is None or hi is None:
        return None
    return {"lo": round(lo, 2), "hi": round(hi, 2), "contrast": round(hi - lo, 2),
            "threshold": (lo + hi) / 2}


def intervals(edges: list[dict]) -> list[float]:
    return [edges[i + 1]["tMid"] - edges[i]["tMid"] for i in range(len(edges) - 1)]


def embeds(edges: list[dict], gdirs: list[str], gints: list[float],
           tau: float = TAU) -> list[tuple]:
    """All strictly-increasing ordinal embeddings of edges into the schedule.

    A bridge from ordinal o to o2>o spans gints[o:o2]; it must equal the measured
    interval within tau and land on an event whose direction equals the observed
    edge direction. Full paths only; length feasibility checked each step.
    """
    m, n = len(gdirs), len(edges)
    if n == 0:
        return []
    cur: dict[int, list[tuple]] = {}
    for o in range(m):
        if gdirs[o] == edges[0]["dir"]:
            cur.setdefault(o, []).append((o,))
    ivs = intervals(edges)
    for i, iv in enumerate(ivs):
        nxt: dict[int, list[tuple]] = {}
        for o, paths in cur.items():
            for o2 in range(o + 1, m):
                if gdirs[o2] != edges[i + 1]["dir"]:
                    continue
                if abs(iv - sum(gints[o:o2])) > tau:
                    continue
                remaining_edges = n - (i + 2)
                if (m - 1) - o2 < remaining_edges:
                    continue
                for p in paths:
                    nxt.setdefault(o2, []).append(p + (o2,))
        cur = nxt
        if not cur:
            return []
    return [p for paths in cur.values() for p in paths]


def fit_side(edges: list[dict], gdirs: list[str], gints: list[float]) -> dict:
    cands = embeds(edges, gdirs, gints)
    fit = {"n_edges": len(edges), "candidates": len(cands), "unique": len(cands) == 1}
    if len(cands) == 1:
        emb = cands[0]
        fit["start_ordinal0"] = emb[0]
        fit["end_ordinal0"] = emb[-1]
        fit["ordinals0"] = list(emb)
        ivs = intervals(edges)
        resid = [abs(ivs[i] - sum(gints[emb[i]:emb[i + 1]])) for i in range(len(ivs))]
        fit["max_interval_residual_ms"] = round(max(resid), 1) if resid else 0.0
    elif len(cands) > 1:
        fit["candidate_starts0"] = sorted({c[0] for c in cands})
    return fit


def pair_on_ordinals(fit_b: dict, edges_b: list[dict], fit_d: dict,
                     edges_d: list[dict]) -> list[dict]:
    if not (fit_b.get("unique") and fit_d.get("unique")):
        return []
    ob = dict(zip(fit_b["ordinals0"], edges_b))
    od = dict(zip(fit_d["ordinals0"], edges_d))
    pairs = []
    for o in sorted(set(ob) & set(od)):
        b, d = ob[o], od[o]
        pairs.append({
            "ordinal0": o, "dir": b["dir"],
            "browser_t_mid": b["tMid"], "direct_t_mid": d["tMid"],
            "delta_point_ms": round(b["tMid"] - d["tMid"], 1),
            "delta_lower_ms": round(b["tLo"] - d["tHi"], 1),
            "delta_upper_ms": round(b["tHi"] - d["tLo"], 1),
            "browser_bracket_ms": round(b["tHi"] - b["tLo"], 1),
            "direct_bracket_ms": round(d["tHi"] - d["tLo"], 1),
        })
    return pairs


def stats_for(pairs: list[dict], edges_b: list[dict], edges_d: list[dict],
              fit_b: dict, fit_d: dict) -> dict:
    deltas = [p["delta_point_ms"] for p in pairs]
    ob = set(fit_b.get("ordinals0") or [])
    od = set(fit_d.get("ordinals0") or [])
    return {
        "n_paired": len(pairs),
        "n_browser_edges": len(edges_b),
        "n_direct_edges": len(edges_d),
        "n_unmatched_browser": len(ob - od),
        "n_unmatched_direct": len(od - ob),
        "point_min_ms": min(deltas) if deltas else None,
        "point_p50_ms": nr(deltas, 0.5),
        "point_p95_ms": nr(deltas, 0.95),
        "point_max_ms": max(deltas) if deltas else None,
        "lower_p50_ms": nr([p["delta_lower_ms"] for p in pairs], 0.5),
        "upper_p50_ms": nr([p["delta_upper_ms"] for p in pairs], 0.5),
        "bounds_envelope_min_lower_ms": min((p["delta_lower_ms"] for p in pairs), default=None),
        "bounds_envelope_max_upper_ms": max((p["delta_upper_ms"] for p in pairs), default=None),
        "browser_bracket_p50_ms": nr([p["browser_bracket_ms"] for p in pairs], 0.5),
        "direct_bracket_p50_ms": nr([p["direct_bracket_ms"] for p in pairs], 0.5),
        "percentile_method": "nearest-rank",
    }


def synthetic(events: list[dict], lag_ms: float, drop_index: int | None = None) -> list[dict]:
    out = []
    for i, e in enumerate(events):
        if drop_index is not None and i == drop_index:
            continue
        t = e["t_ms"] + lag_ms
        out.append({"dir": e["dir"], "tMid": float(t), "tLo": float(t - 20),
                    "tHi": float(t + 20)})
    return out


def run_controls(events: list[dict], gdirs: list[str], gints: list[float]) -> dict:
    ctrl: dict = {}

    # positive: exact schedule, +350 ms both sides -> unique fits, all deltas +350
    fb = fit_side(synthetic(events, 350), gdirs, gints)
    fd = fit_side(synthetic(events, 0), gdirs, gints)
    pairs = pair_on_ordinals(fb, synthetic(events, 350), fd, synthetic(events, 0))
    ds = [p["delta_point_ms"] for p in pairs]
    ctrl["positive_lag350"] = {
        "expect": "unique fits both sides, >=4 pairs, all deltas exactly +350",
        "candidates_b": fb["candidates"], "candidates_d": fd["candidates"],
        "n_paired": len(pairs),
        "all_equal_350": bool(ds) and all(d == 350.0 for d in ds),
        "min": min(ds) if ds else None, "max": max(ds) if ds else None,
    }

    # delayed: browser +3350 ms -> must REPORT +3350 (sequence identity, not phase alias)
    fb = fit_side(synthetic(events, 3350), gdirs, gints)
    fd = fit_side(synthetic(events, 0), gdirs, gints)
    pairs = pair_on_ordinals(fb, synthetic(events, 3350), fd, synthetic(events, 0))
    ds = [p["delta_point_ms"] for p in pairs]
    ctrl["delayed_lag3350"] = {
        "expect": "unique fits, all deltas exactly +3350 (never aliased to +350)",
        "candidates_b": fb["candidates"], "candidates_d": fd["candidates"],
        "n_paired": len(pairs),
        "all_equal_3350": bool(ds) and all(d == 3350.0 for d in ds),
        "min": min(ds) if ds else None, "max": max(ds) if ds else None,
    }

    # dropped: remove one browser edge -> still unique, remaining pairs stay +350,
    # one direct ordinal left unmatched, no cycle shift
    eb = synthetic(events, 350, drop_index=7)
    ed = synthetic(events, 0)
    fb = fit_side(eb, gdirs, gints)
    fd = fit_side(ed, gdirs, gints)
    pairs = pair_on_ordinals(fb, eb, fd, ed)
    ds = [p["delta_point_ms"] for p in pairs]
    ctrl["dropped_edge"] = {
        "expect": "unique fits, pairs = len(events)-1, all deltas exactly +350, "
                  "exactly one unmatched direct ordinal",
        "candidates_b": fb["candidates"], "candidates_d": fd["candidates"],
        "n_pairs_expected": len(events) - 1, "n_paired": len(pairs),
        "n_unmatched_direct": stats_for(pairs, eb, ed, fb, fd)["n_unmatched_direct"],
        "all_equal_350": bool(ds) and all(d == 350.0 for d in ds),
        "min": min(ds) if ds else None, "max": max(ds) if ds else None,
    }

    # periodic: the historic failure mode. A partial capture (first two events
    # missed, exactly like the real run) of a 1s/2s-periodic marker embeds at TWO
    # schedule offsets -> ambiguous -> must yield 0 pairs.
    p_events, t = [], 5000
    for i in range(21):
        p_events.append({"ordinal": i + 1, "t_ms": t,
                         "dir": "rising" if i % 2 == 0 else "falling"})
        t += 1000 if i % 2 == 0 else 2000
    p_gdirs = [e["dir"] for e in p_events]
    p_gints = [p_events[i + 1]["t_ms"] - p_events[i]["t_ms"] for i in range(len(p_events) - 1)]
    epb = [e for e in p_events[2:]]  # mirror real capture: first two events missed
    eb = synthetic(epb, 3350)
    ed = synthetic(p_events, 0)
    fb = fit_side(eb, p_gdirs, p_gints)
    fd = fit_side(ed, p_gdirs, p_gints)
    pairs = pair_on_ordinals(fb, eb, fd, ed) if fb["unique"] and fd["unique"] else []
    ctrl["periodic_ambiguous"] = {
        "expect": "browser-side partial periodic capture embeds at >=2 offsets "
                  "-> rejected, 0 pairs",
        "candidates_b": fb["candidates"], "candidates_d": fd["candidates"],
        "browser_unique": fb["unique"], "n_paired": len(pairs),
        "rejected_as_ambiguous": fb["candidates"] > 1 or fd["candidates"] > 1,
    }

    ctrl["all_passed"] = (
        ctrl["positive_lag350"]["all_equal_350"]
        and ctrl["positive_lag350"]["n_paired"] >= MIN_IDENTITY_EDGES
        and ctrl["positive_lag350"]["candidates_b"] == 1
        and ctrl["delayed_lag3350"]["all_equal_3350"]
        and ctrl["delayed_lag3350"]["n_paired"] >= MIN_IDENTITY_EDGES
        and ctrl["dropped_edge"]["all_equal_350"]
        and ctrl["dropped_edge"]["n_paired"] == len(events) - 1
        and ctrl["dropped_edge"]["candidates_b"] == 1
        and ctrl["dropped_edge"]["n_unmatched_direct"] == 1
        and ctrl["periodic_ambiguous"]["rejected_as_ambiguous"]
        and ctrl["periodic_ambiguous"]["n_paired"] == 0
    )
    return ctrl


def main() -> None:
    raw = json.loads(RAW.read_text())
    sched = json.loads(SCHED.read_text())
    events = sched["events"]
    gdirs = [e["dir"] for e in events]
    gints = [events[i + 1]["t_ms"] - events[i]["t_ms"] for i in range(len(events) - 1)]
    gints_field_matches = gints == sched["intervals_ms"]

    controls = run_controls(events, gdirs, gints)

    edges_b = raw["edgesBrowser"]
    edges_d = raw["edgesDirect"]

    # independent contrast recomputation from saved raw samples inside the window
    wf = raw.get("windowFilter") or {}
    lo_w, hi_w = wf.get("startWall", -1e30), wf.get("endWall", 1e30)
    lumas_b = [s["luma"] for s in raw["samplesBrowser"]
               if isinstance(s.get("luma"), (int, float)) and lo_w <= s["t"] <= hi_w]
    lumas_d = [s["luma"] for s in raw["samplesDirect"]
               if isinstance(s.get("luma"), (int, float)) and lo_w <= s["t"] <= hi_w]
    cal_b, cal_d = calibrate(lumas_b), calibrate(lumas_d)

    # order check (strict alternation), as in the sampler's contract
    def alternating(edges):
        return all(edges[i]["dir"] != edges[i - 1]["dir"] for i in range(1, len(edges)))

    order_ok = alternating(edges_b) and alternating(edges_d)

    fit_b = fit_side(edges_b, gdirs, gints)
    fit_d = fit_side(edges_d, gdirs, gints)

    # secondary cross-side check: browser vs direct directly (not via schedule)
    cross = []
    ib, idr = intervals(edges_b), intervals(edges_d)
    for k in range(-5, 6):
        n_ov = 0
        ok = True
        for i in range(len(ib)):
            j = i + k
            if not 0 <= j < len(idr):
                continue
            n_ov += 1
            if edges_b[i]["dir"] != edges_d[j]["dir"] or abs(ib[i] - idr[j]) > TAU:
                ok = False
                break
        if ok and n_ov >= MIN_IDENTITY_EDGES:
            cross.append(k)
    crosscheck = {"valid_offsets_browser_to_direct": cross,
                  "note": "offset 0 required and exclusive for a valid cross-side fit"}

    pairs = pair_on_ordinals(fit_b, edges_b, fit_d, edges_d)
    st = stats_for(pairs, edges_b, edges_d, fit_b, fit_d)

    contrast_ok = (cal_b and cal_d
                   and cal_b["contrast"] >= MIN_CONTRAST
                   and cal_d["contrast"] >= MIN_CONTRAST)
    identity_ok = fit_b["unique"] and fit_d["unique"] and st["n_paired"] >= MIN_IDENTITY_EDGES
    valid = bool(controls["all_passed"] and contrast_ok and order_ok and identity_ok
                 and cross == [0])
    invalid_reason = None
    if not valid:
        reasons = []
        if not controls["all_passed"]:
            reasons.append("control-failure")
        if not contrast_ok:
            reasons.append("contrast-below-floor")
        if not order_ok:
            reasons.append("edge-order-not-alternating")
        if not (fit_b["unique"] and fit_d["unique"]):
            reasons.append(f"identity-ambiguous-or-missing(browser={fit_b['candidates']},"
                           f"direct={fit_d['candidates']})")
        if st["n_paired"] < MIN_IDENTITY_EDGES:
            reasons.append("insufficient-identity-pairs")
        if cross != [0]:
            reasons.append(f"cross-side-offset-not-unique-zero({cross})")
        invalid_reason = ";".join(reasons)

    # renderer / transport identity captured by the sampler
    sockets = [{"url": s.get("url"), "firstText": s.get("firstText"),
                "recvBin": s.get("recvBin"), "recvBytes": s.get("recvBytes"),
                "firstBinAt": s.get("firstBinAt"), "lastBinAt": s.get("lastBinAt")}
               for s in raw.get("sockets", [])]
    kinds = sorted({s.get("kind") for s in raw["samplesBrowser"] if s.get("kind")})
    conts = (raw.get("pageAInfo") or {}).get("containers") or []
    video = next((c.get("video") for c in conts if c.get("video")), None)

    def span(samples):
        ts = [s["t"] for s in samples]
        return {"n_samples": len(samples), "t_first": min(ts) if ts else None,
                "t_last": max(ts) if ts else None,
                "span_ms": round(max(ts) - min(ts), 1) if ts else None}

    clock = raw.get("clockCheck") or []

    out = {
        "asOf": raw.get("asOfUTC"),
        "kind": "standalone-offline-reanalysis-of-saved-raw (no new capture)",
        "inputs": {
            "raw_json": RAW.name, "raw_sha256": sha256(RAW),
            "schedule_json": SCHED.name, "schedule_sha256": sha256(SCHED),
            "original_sampler_note": "paced-r1 capture used sampler SHA "
                                      "35ddc26bd84b6cf67cc23497051c24bbd5129c9944b05d8e753d88e8f671ba73 "
                                      "(4 missing config constants restored from frozen baseline "
                                      "config after run-1 ReferenceError)",
        },
        "method": {
            "identity": "per-side unique ordinal embedding into frozen schedule, "
                        "joined on ordinal (never same-index, never nearest-phase)",
            "tau_ms": TAU,
            "tau_rationale": "schedule intervals are multiples of 1000 ms with minimum "
                             "separation 1000 ms; TAU < 500 keeps single-interval matches "
                             "unique, multi-interval bridges are guarded by whole-sequence "
                             "uniqueness",
            "gints_field_matches_recomputed": gints_field_matches,
            "schedule_intervals_ms": gints,
        },
        "controls": controls,
        "preflight_reference": (raw.get("stats") or {}).get("preflight"),
        "original_capture_stats": raw.get("stats"),  # preserved INVALID verdict
        "independent_contrast_recomputed": {"browser": cal_b, "direct": cal_d,
                                            "min_contrast_required": MIN_CONTRAST},
        "edge_order_alternating": order_ok,
        "fits": {"browser": {k: v for k, v in fit_b.items() if k != "ordinals0"},
                 "direct": {k: v for k, v in fit_d.items() if k != "ordinals0"}},
        "crosscheck_browser_vs_direct": crosscheck,
        "reanalysis_stats": {**st, "identity_rule": "unique-schedule-embedding-per-side",
                             "min_identity_edges": MIN_IDENTITY_EDGES},
        "pairs": pairs,
        "valid": valid,
        "invalid_reason": invalid_reason,
        "renderer_transport_identity": {
            "browser_decoder_kinds": kinds,
            "selected_video": video,
            "sockets": sockets,
        },
        "raw_sample_spans": {"browser": span(raw["samplesBrowser"]),
                             "direct": span(raw["samplesDirect"])},
        "window_filter": raw.get("windowFilter"),
        "clock_check": {"n": len(clock),
                        "max_abs_a_minus_node_ms": max((abs(c["a_minus_node"]) for c in clock),
                                                       default=None),
                        "max_abs_b_minus_node_ms": max((abs(c["b_minus_node"]) for c in clock),
                                                       default=None)},
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "controls_all_passed": controls["all_passed"],
        "contrast": {"browser": cal_b, "direct": cal_d},
        "fits": out["fits"],
        "crosscheck": crosscheck,
        "valid": valid, "invalid_reason": invalid_reason,
        "reanalysis_stats": out["reanalysis_stats"],
        "out": str(OUT), "out_sha256": sha256(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
