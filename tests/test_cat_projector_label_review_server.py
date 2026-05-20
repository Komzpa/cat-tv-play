from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cat_projector_label_review_server.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("cat_projector_label_review_server", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
server = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = server
SCRIPT_SPEC.loader.exec_module(server)


def test_fake_smoke_builds_review_queue_and_action_records(tmp_path: Path) -> None:
    assert server.run_fake_smoke(tmp_path) == 0
    label_files = sorted((tmp_path / "state" / "label-review" / "labels").glob("*.json"))
    action_files = sorted((tmp_path / "state" / "label-review" / "actions").glob("*.json"))
    mask_files = sorted((tmp_path / "state" / "label-review" / "masks").glob("*/*.json"))
    assert len(label_files) == 2
    assert len(action_files) == 2
    assert len(mask_files) == 1


def test_segment_endpoint_requires_configured_sam_without_explicit_degraded_fallback(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"not-an-image")
    old_endpoint = server.SAM_ENDPOINT
    try:
        server.SAM_ENDPOINT = ""
        try:
            server._segment_with_optional_sam(
                image_path,
                [{"x": 1, "y": 1}],
                [],
                [],
            )
        except ValueError as exc:
            assert "CAT_PROJECTOR_SAM_ENDPOINT is not configured" in str(exc)
        else:
            raise AssertionError("segment unexpectedly fell back without allow_fallback")
    finally:
        server.SAM_ENDPOINT = old_endpoint
