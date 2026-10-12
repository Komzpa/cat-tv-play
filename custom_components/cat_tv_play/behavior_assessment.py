"""Independent, bounded camera behavior assessment; no delivery or actuator calls."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from datetime import UTC, datetime
from typing import Any, Callable

SCHEMA = 'cat_projector_behavior_assessment.v1'


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assess_recording(recording_dir: Path, *, model: dict[str, str],
                     score_records: Callable, confirm_render: Callable,
                     verify_display: Callable, max_chunks: int = 4,
                     min_probability: float = 0.90, min_height_cm: float = 45.0,
                     selected_chunks: list[Path] | None = None) -> dict[str, Any]:
    """A verified jump may establish play; sparse/absent detections always abstain."""
    manifest_path = recording_dir / 'manifest.json'
    verified_path = recording_dir / 'verified.json'
    manifest = json.loads(manifest_path.read_text())
    if not 1 <= max_chunks <= 4:
        raise ValueError('bounded camera chunk limit required')
    output = recording_dir / 'behavior_assessment.json'
    previous = json.loads(output.read_text()) if output.exists() else {}
    available = [p for p in sorted(recording_dir.glob('chunk_*.mp4')) if '.part.' not in p.name]
    policy = [max_chunks, min_probability, min_height_cm]
    revision = {'manifest_sha256': file_sha256(manifest_path), 'model': model, 'assessor_sha256': file_sha256(Path(__file__)),
                'verified_sha256': file_sha256(verified_path) if verified_path.is_file() else None, 'policy': policy}
    same_revision = all(previous.get('inputs', {}).get(k) == v for k, v in revision.items())
    assessed = dict(previous.get('coverage', {}).get('assessed_chunks', {})) if same_revision else {}
    assessed_stats = dict(previous.get('coverage', {}).get('assessed_chunk_stats', {})) if same_revision else {}
    if selected_chunks is not None:
        if not selected_chunks or len(selected_chunks) > max_chunks or any(p.resolve() not in {a.resolve() for a in available} for p in selected_chunks):
            raise ValueError('selected chunks must be bounded original recording chunks')
        chunks = selected_chunks
    else:
        chunks = []
        for p in available:
            stat = p.stat()
            if assessed_stats.get(str(p)) != [stat.st_size, stat.st_mtime_ns]:
                chunks.append(p)
                if len(chunks) == max_chunks:
                    break
    inputs = {'manifest_sha256': file_sha256(manifest_path), 'model': model,
              'assessor_sha256': file_sha256(Path(__file__)),
              'verified_sha256': file_sha256(verified_path) if verified_path.is_file() else None,
              'policy': [max_chunks, min_probability, min_height_cm],
              'chunks': {str(p): file_sha256(p) for p in chunks}}
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if previous:
        if not chunks and same_revision:
            return previous
        previous_inputs = previous.get('inputs', {})
        retained_positive = (previous.get('status') == 'confirmed'
            and previous_inputs.get('manifest_sha256') == inputs['manifest_sha256']
            and all(previous_inputs.get(k) == inputs[k] for k in ('model', 'assessor_sha256', 'verified_sha256', 'policy'))
            and bool(previous_inputs.get('chunks'))
            and all(Path(p).is_file() and file_sha256(Path(p)) == sha
                    for p, sha in previous_inputs.get('chunks', {}).items()))
        if previous.get('assessment_id') == identity or retained_positive:
            return previous
    result = {'schema': SCHEMA, 'assessment_id': identity, 'created_at': datetime.now(UTC).isoformat(),
              'session_id': manifest.get('ha_session_id', ''),
              'source_meow_event_id': manifest.get('source_meow_event_id', ''),
              'outcome': 'unknown', 'status': 'abstain', 'confidence': 0.0,
              'display_verified': False, 'visibility_verified': False,
              'model': model, 'inputs': inputs,
              'evidence_refs': {'manifest': str(manifest_path), 'verified': str(verified_path)},
              'coverage': {'max_chunks': max_chunks, 'scanned_chunks': len(chunks),
                           'whole_session_observed': False, 'assessed_chunks': assessed,
                           'assessed_chunk_stats': assessed_stats},
              'request_intention': 'not_inferred_from_post_start_behavior'}
    try:
        if not verified_path.is_file() or not chunks or not model.get('sha256'):
            result['reason'] = 'missing_verified_video_or_model'
        else:
            records = score_records(chunks)
            candidates = [r for r in records if r.get('reason') == 'source_subtracted_calibrated_high_jump'
                and isinstance(r.get('cat_probability'), (int, float))
                and math.isfinite(r['cat_probability']) and min_probability <= r['cat_probability'] <= 1
                and math.isfinite(float(r.get('height_cm') or 0))
                and float(r.get('height_cm') or 0) >= min_height_cm
                and Path(str(r.get('model_path') or '')).resolve() == Path(model.get('path', '')).resolve()]
            result['candidate_count'] = len(candidates)
            if not candidates:
                result['reason'] = 'no_confirmed_high_probability_jump; absence_is_not_negative'
            else:
                candidate = candidates[0]
                display = verify_display(chunks, candidate)
                result['display_evidence'] = display
                result['display_verified'] = display.get('verified') is True
                if not result['display_verified']:
                    result['reason'] = 'projected_source_not_observably_verified'
                else:
                    stats = confirm_render(chunks, candidate)
                    result['render_stats'] = stats
                    result['visibility_verified'] = (math.isfinite(float(stats.get('max_height_cm') or 0))
                        and float(stats.get('max_height_cm') or 0) >= min_height_cm
                        and int(stats.get('candidate_frame_count') or 0) >= 2
                        and bool(stats.get('canonical_alert_record'))
                        and stats.get('canonical_detector_route') == 'source_subtracted'
                        and Path(str(stats.get('canonical_detector_model') or '')).resolve() == Path(model.get('path', '')).resolve()
                        and isinstance(stats.get('canonical_detector_probability'), (int, float))
                        and math.isfinite(stats['canonical_detector_probability'])
                        and min_probability <= stats['canonical_detector_probability'] <= 1)
                    if result['visibility_verified']:
                        result.update(outcome='active_play', status='confirmed', confidence=float(candidate['cat_probability']),
                                      reason='source_matched_cat_jump_confirmed_in_render')
                        result['evidence_refs']['camera_chunk'] = str(next(p for p in chunks if str(candidate.get('chunk', '')).isdigit() and int(p.stem.split('_')[-1]) == int(candidate['chunk'])))
                    else:
                        result['reason'] = 'jump_not_confirmed_in_render'
    except Exception as exc:
        # Report the failure type, never endpoint/token-bearing exception text.
        result.update(outcome='unknown', status='abstain', confidence=0.0,
                      reason='assessment_error:' + type(exc).__name__)
    if not result['reason'].startswith('assessment_error:'):
        result['coverage']['assessed_chunks'].update(inputs['chunks'])
        result['coverage']['assessed_chunk_stats'].update({str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in chunks})
    if model.get('path') and Path(model['path']).is_file() and file_sha256(Path(model['path'])) != model.get('sha256'):
        result.update(outcome='unknown', status='abstain', confidence=0.0, reason='model_changed_during_assessment')
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    temporary.replace(output)
    return result
