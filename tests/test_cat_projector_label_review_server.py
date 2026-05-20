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
