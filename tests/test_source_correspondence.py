from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict, replace
from fractions import Fraction
from io import BytesIO
from itertools import combinations_with_replacement
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
NAME = "cat_tv_play_source_correspondence"
if NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(NAME, ROOT / "custom_components/cat_tv_play/source_correspondence.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    spec.loader.exec_module(module)
native = sys.modules[NAME]


def native_bundle(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    media = tmp_path / "chunk_1176.mp4"
    media.write_bytes(b"media identity, not a playable test fixture")
    pixels = Image.new("RGB", (16, 12), (40, 80, 120))
    pixels.save(tmp_path / "frame-0000.png")
    rgb_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
    (tmp_path / "framehash.txt").write_text(f"#tb 0: 1/12800\n0, 137, 137, 1, 576, {rgb_sha}\n")
    value = {
        "exit_status": 0, "same_decode_hash_identity": True, "width": 16, "height": 12,
        "media_sha256": native.sha256_file(media),
        "command": ["ffmpeg", "-copyts", "-i", str(media), "-filter_complex",
                    "[0:v]format=rgb24,split=2[pixels][identity]", "-fps_mode", "passthrough", "-enc_time_base:v", "filter"],
        "frames": [{"decode_frame_index": 0, "native_pts": 137, "native_time_base": "1/12800",
                    "rgb24_sha256": rgb_sha, "lossless_png": "frame-0000.png"}],
    }
    (tmp_path / "identity.json").write_text(json.dumps(value))
    identity = native._bundle(tmp_path, str(media), native.sha256_file(media))[0]
    return media, identity, value


def retained_index(tmp_path: Path, source_sha: str):
    path = tmp_path / "source-index.npz"
    geometry = {"descriptor_grid": [48, 32], "exclusions": [.69, .82],
                "polygon_1280": [[40.22, 57.27], [937.92, 101.], [908., 599.], [48.16, 680.97]],
                "source_subtraction_sha256": native.LEGACY_GEOMETRY_DEPENDENCY}
    metadata = {"geometry": geometry, "slots": 2, "descriptor_dim": 4203, "fps": 4,
                "implementation": {"index_decode": native.INDEX_DECODE},
                "source": {"sha256": source_sha, "url": "http://example.invalid/source.mp4"},
                "keys": {"source_sha256": source_sha, "geometry_sha256": native._digest(geometry),
                         "implementation_sha256": native.LEGACY_IMPLEMENTATION_KEY}}
    np.savez(path, features=np.zeros((2, 4203), np.float32),
             **{f"key_{key}": value for key, value in metadata["keys"].items()})
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return path, metadata


def bound_chunk(tmp_path: Path):
    media, identity, _value = native_bundle(tmp_path / "camera")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"cached source bytes")
    source_sha = native.sha256_file(source)
    index, metadata = retained_index(tmp_path, source_sha)
    geometry = native.ProjectorGeometry(tuple(tuple(point) for point in metadata["geometry"]["polygon_1280"]))
    frame = native.FrameCorrespondence(identity, None, None, (), "unresolved", ("no content support",),
                                       geometry.sha256, ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    chunk = native.ChunkCorrespondence(identity.media_sha256, source_sha, native.sha256_file(index),
                                      geometry.sha256, native.sha256_file(Path(native.__file__)), (frame,),
                                      {"camera_chunk": str(media), "projector_source": str(source),
                                       "source_binding": "current cached source bytes SHA-verified",
                                       "source_index": str(index), "index_metadata_sha256": native.sha256_file(index.with_suffix(".json")),
                                       "geometry": asdict(geometry),
                                       "proposal_implementation_sha256": native.sha256_file(ROOT / "scripts/cat_projector_frame_detector.py")},
                                      {}, (), str(tmp_path / "correspondence.json"))
    return chunk, identity


def test_native_pts_never_comes_from_chunk_number(tmp_path: Path) -> None:
    chunk, identity = bound_chunk(tmp_path)
    frame = native.correspondence_for_camera_frame(chunk, identity)
    assert frame.camera_identity.seconds == Fraction(137, 12800)
    assert frame.structural_status == "unresolved"
    assert frame.measured_exposure_seconds is None
    with pytest.raises(ValueError, match="exact native camera identity"):
        native.correspondence_for_camera_frame(chunk, replace(identity, pts=138))


def test_missing_pts_is_rejected_in_native_manifest(tmp_path: Path) -> None:
    media, _identity, value = native_bundle(tmp_path)
    del value["frames"][0]["native_pts"]
    (tmp_path / "identity.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="native PTS/RGB identity"):
        native._bundle(tmp_path, str(media), native.sha256_file(media))


def test_native_identity_requires_integer_pts_and_rational_timebase(tmp_path: Path) -> None:
    _media, identity, _value = native_bundle(tmp_path)
    with pytest.raises(ValueError, match="integer PTS"):
        replace(identity, pts=0.01)
    with pytest.raises(ValueError, match="time base"):
        replace(identity, time_base_den=0)


def test_native_rgb_substitution_is_rejected(tmp_path: Path) -> None:
    _media, identity, _value = native_bundle(tmp_path)
    assert native.load_native_rgb(identity).getpixel((0, 0)) == (40, 80, 120)
    Image.new("RGB", (16, 12), "white").save(identity.rgb_path)
    with pytest.raises(ValueError, match="substituted native RGB"):
        native.load_native_rgb(identity)


def test_framehash_substitution_is_rejected(tmp_path: Path) -> None:
    _media, identity, _value = native_bundle(tmp_path)
    native.load_native_rgb(identity)
    path = tmp_path / "framehash.txt"
    path.write_text(path.read_text().replace("137, 137", "138, 138"))
    with pytest.raises(ValueError, match="substituted native identity"):
        native.load_native_rgb(identity)


def test_ordered_native_path_allows_repeated_ids_and_nonunit_rate() -> None:
    scores = np.zeros((9, 5), np.float32)
    expected = (0, 4, 4, 5, 8)
    for slot, index in enumerate(expected):
        scores[index, slot] = 1
    path = native.ordered_native_path(scores, (0, 1, 2, 3, 4))
    assert path == expected
    assert native.evaluate_native_sequence(scores, path)["repeated_native_ids"]


def test_heldout_lookup_cannot_escape_discovery_bracket() -> None:
    scores = np.zeros((8, 5), np.float32)
    scores[2, 1] = scores[4, 3] = 1
    scores[7, 2] = 100
    scores[3, 2] = 1
    path = native.ordered_native_path(scores, (1, 3))
    assert path[2] == 3


def test_dense_heldout_completion_cannot_cross_between_discovery_anchors() -> None:
    scores = np.zeros((5, 5), np.float32)
    scores[0, 0] = scores[4, 4] = 10
    scores[3, 1] = 2
    scores[1, 2] = 3
    path = native.ordered_native_path(scores, (0, 4))
    assert all(b >= a for a, b in zip(path, path[1:]))
    assert path[2] == 1
    backward = native.ordered_native_path(scores, (0, 4), -1)
    assert all(b <= a for a, b in zip(backward, backward[1:]))


def test_same_oracle_penalizes_blank_reversed_and_decreasing_sequences() -> None:
    scores = np.eye(5, dtype=np.float32)
    ordered = native.evaluate_native_sequence(scores, (0, 1, 2, 3, 4))
    reversed_path = native.evaluate_native_sequence(scores, (4, 3, 2, 1, 0))
    fixed = native.evaluate_native_sequence(scores, (2, 2, 2, 2, 2))
    blank = native.evaluate_native_sequence(np.zeros_like(scores), (0, 1, 2, 3, 4))
    assert ordered["mean"] > reversed_path["mean"] > blank["mean"]
    assert ordered["mean"] > fixed["mean"]
    assert not reversed_path["nondecreasing_native_order"]
    assert fixed["repeated_native_ids"]
    with pytest.raises(ValueError, match="native score matrix"):
        native.evaluate_native_sequence(scores, (0, 1, 2, 3, 5))


def test_conditional_native_envelopes_match_every_complete_ordered_path() -> None:
    scores = np.array([[.2, .7, .1], [.8, .4, .6], [.3, .5, .9]], np.float64)
    expected = np.full(scores.shape, -np.inf)
    for path in combinations_with_replacement(range(3), 3):
        total = sum(scores[index, slot] for slot, index in enumerate(path))
        for slot, index in enumerate(path):
            expected[index, slot] = max(expected[index, slot], total)
    np.testing.assert_allclose(native.ordered_state_path_totals(scores), expected)


def test_local_null_failure_cannot_borrow_other_observations_support() -> None:
    scores = np.array([[100., 0., 100.], [1., 1., 1.]])
    local = scores > .5
    totals = native.ordered_state_path_totals(scores)
    qualified = local & (totals / scores.shape[1] > .5)
    assert totals[0, 1] == 200.  # a large complete path cannot supply local shape
    assert not qualified[0, 1]
    assert qualified[1, 1]


def test_missing_camera_proposal_does_not_reject_other_observed_states() -> None:
    scores = np.array([[1., 0., 1.], [1., 0., 1.]])
    totals = native.ordered_state_path_totals(scores)
    qualified = (scores > 0) & (totals / 3 > .5)
    assert qualified[:, 0].all() and qualified[:, 2].all()
    assert not qualified[:, 1].any()


def test_equal_native_hypotheses_and_repeated_states_all_survive() -> None:
    scores = np.ones((4, 3))
    totals = native.ordered_state_path_totals(scores)
    assert (totals == 3).all()  # not argmax-as-unique, and total != mean
    assert (totals / scores.shape[1] == 1).all()


def test_frozen_control_applicability_does_not_transfer_between_captures(tmp_path: Path) -> None:
    media, identity, _ = native_bundle(tmp_path / "null")
    index, metadata = retained_index(tmp_path, native.sha256_file(media))
    metadata["source"]["url"] = str(media)
    index.with_suffix(".json").write_text(json.dumps(metadata))
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"observed": true}')
    controls = index.with_suffix(".qualification-controls.json")
    controls.write_text(json.dumps({"kind": "frozen-native-structural-controls-v1",
        "source_sha256": identity.media_sha256, "index_sha256": native.sha256_file(index),
        "baseline_evidence": {"path": str(baseline), "sha256": native.sha256_file(baseline)},
        "controls": [{"identity_path": identity.identity_path, "identity_sha256": identity.decode_id,
            "decode_id": identity.decode_id, "roles": ["local_shape_null", "sequence_rejected"],
            "applicability": {"camera_sha256": "2" * 64, "geometry_sha256": "3" * 64}}]}))
    provenance, roles = native._qualification_controls(None, index, identity.media_sha256, "2" * 64, "3" * 64)
    assert roles[identity.decode_id] == {"local_shape_null", "sequence_rejected"}
    assert provenance["status"] == "applicable"
    provenance, roles = native._qualification_controls(None, index, identity.media_sha256, "4" * 64, "3" * 64)
    assert provenance["status"] == "inapplicable"
    assert not roles
    value = json.loads(controls.read_text())
    value["source_facts"] = [{"kind": "source-null-native-sample-v1", "source_sha256": identity.media_sha256,
        "identity_path": identity.identity_path, "identity_sha256": identity.decode_id, "decode_id": identity.decode_id}]
    controls.write_text(json.dumps(value))
    provenance, roles = native._qualification_controls(None, index, identity.media_sha256, "4" * 64, "5" * 64)
    assert roles[identity.decode_id] == {"local_shape_null"}  # source fact, NOT July sequence rejection
    assert provenance["inapplicable_decode_ids"] == [identity.decode_id]
    assert provenance["source_only_facts"][0]["native_identities"] == [asdict(identity)]
    baseline.write_text('{"substituted": true}')
    with pytest.raises(ValueError, match="baseline evidence changed"):
        native._qualification_controls(None, index, identity.media_sha256, "2" * 64, "3" * 64)


def test_camera_only_proposals_exclude_socket_and_boundary() -> None:
    image = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((610, 220, 680, 280), fill="black")
    draw.rectangle((60, 450, 120, 500), fill="black")
    draw.rectangle((35, 100, 50, 600), fill="black")
    polygon = ((40.22, 57.27), (937.92, 101.), (908., 599.), (48.16, 680.97))
    interior = native._projection(polygon, 1280, 720)
    excluded = np.zeros((720, 1280), bool)
    excluded[445:515, 55:130] = True
    support, boxes = native._camera_support(image, interior, excluded)
    assert support[240, 640]
    assert not support[445:515, 55:130].any()
    assert boxes


def test_temporal_support_does_not_label_static_wall_or_global_dimming() -> None:
    reference = np.tile(np.linspace(100, 220, 1280, dtype=np.float32), (720, 1))
    image = Image.fromarray((reference - 10).astype(np.uint8)).convert("RGB")
    interior = np.zeros((720, 1280), bool)
    interior[30:690, 30:1250] = True
    excluded = np.zeros_like(interior)
    excluded[445:515, 55:130] = True
    support, boxes = native._camera_support(image, interior, excluded, reference)
    assert not support.any()
    assert not boxes
    draw = ImageDraw.Draw(image)
    draw.ellipse((600, 220, 680, 300), fill="black")
    draw.rectangle((60, 450, 120, 500), fill="black")
    support, boxes = native._camera_support(image, interior, excluded, reference)
    assert support[260, 640]
    assert not support[excluded].any()
    assert not support[~interior].any()
    assert boxes


def test_dependency_migration_preserves_retained_generation_keys(tmp_path: Path) -> None:
    index, metadata = retained_index(tmp_path, "1" * 64)
    before = native.sha256_file(index)
    features, _metadata, proof = native.load_source_index(index, "1" * 64, metadata["source"]["url"])
    assert features.shape == (2, 4203)
    assert proof["index_rekeyed"] is False
    assert proof["original_generation_keys"] == metadata["keys"]
    assert native.sha256_file(index) == before


@pytest.mark.parametrize("key", ["source_sha256", "geometry_sha256", "implementation_sha256"])
def test_stale_index_keys_are_rejected(tmp_path: Path, key: str) -> None:
    index, metadata = retained_index(tmp_path, "1" * 64)
    metadata["keys"][key] = "0" * 64
    index.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="key|rebuild"):
        native.load_source_index(index, "1" * 64, metadata["source"]["url"])


@pytest.mark.parametrize("field", ["producer_sha256", "geometry_sha256", "source_sha256", "index_sha256"])
def test_stale_chunk_contract_is_rejected(tmp_path: Path, field: str) -> None:
    chunk, identity = bound_chunk(tmp_path)
    with pytest.raises(ValueError):
        native.correspondence_for_camera_frame(replace(chunk, **{field: "0" * 64}), identity)


def test_persistent_roundtrip_keeps_exposure_unknown(tmp_path: Path) -> None:
    chunk, identity = bound_chunk(tmp_path)
    Path(chunk.output_path).write_text(json.dumps(asdict(chunk)))
    restored = native.load_chunk_correspondence(Path(chunk.output_path))
    frame = native.correspondence_for_camera_frame(restored, identity)
    assert frame == chunk.frames[0]
    assert frame.measured_exposure_seconds is None


def test_cached_source_replacement_is_rejected(tmp_path: Path) -> None:
    chunk, identity = bound_chunk(tmp_path)
    Path(chunk.provenance["projector_source"]).write_bytes(b"different source under the same basename")
    with pytest.raises(ValueError, match="cached source bytes changed"):
        native.correspondence_for_camera_frame(chunk, identity)


def test_camera_chunk_replacement_is_rejected(tmp_path: Path) -> None:
    chunk, identity = bound_chunk(tmp_path)
    Path(chunk.provenance["camera_chunk"]).write_bytes(b"different camera chunk")
    with pytest.raises(ValueError, match="camera chunk bytes changed"):
        native.correspondence_for_camera_frame(chunk, identity)


def http_identity(tmp_path: Path):
    index, metadata = retained_index(tmp_path, "1" * 64)
    source = metadata["source"]
    value = {
        "kind": "server-sha256-bound-http-source-identity", "source": source,
        "server_proof": {"exit_status": 0, "source_file": "/canonical/source.mp4",
                         "sha256sum_stdout": f"{source['sha256']}  /canonical/source.mp4",
                         "stat_before": "123 stable-stat", "stat_after": "123 stable-stat"},
        "http": {"etag_before": '"server-hashed-representation"', "etag_after": '"server-hashed-representation"',
                 "content_length_before": 123, "content_length_after": 123, "accept_ranges": "bytes"},
    }
    path = index.with_name(f"http-source-identity-{source['sha256']}.json")
    path.write_text(json.dumps(value))
    return index, path, value


def test_http_source_requires_current_sha_bound_strong_validator(tmp_path: Path, monkeypatch) -> None:
    index, path, value = http_identity(tmp_path)
    source = value["source"]
    requests = []

    def open_head(request, timeout):
        requests.append(request)
        return nullcontext(SimpleNamespace(
            getcode=lambda: 200, geturl=lambda: source["url"],
            headers={"ETag": value["http"]["etag_before"], "Content-Length": "123", "Accept-Ranges": "bytes"},
        ))

    monkeypatch.setattr(native, "urlopen", open_head)
    identity = native._http_source_identity(source["url"], source["sha256"], index)
    assert identity["identity_sha256"] == native.sha256_file(path)
    assert requests[0].get_method() == "HEAD"
    assert requests[0].get_header("If-match") == value["http"]["etag_before"]
    assert requests[0].get_header("Accept-encoding") == "identity"
    value["server_proof"]["sha256sum_stdout"] = f"{'0' * 64}  /canonical/source.mp4"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="server SHA256"):
        native._http_source_identity(source["url"], source["sha256"], index)
    assert len(requests) == 1  # Retained metadata cannot repair a missing independent hash.


@pytest.mark.parametrize("change", ["etag", "weak", "range", "redirect", "length"])
def test_http_identity_rejects_stale_or_unbounded_representation(tmp_path: Path, monkeypatch, change: str) -> None:
    index, path, value = http_identity(tmp_path)
    headers = {"ETag": value["http"]["etag_before"], "Content-Length": "123", "Accept-Ranges": "bytes"}
    url = value["source"]["url"]
    if change == "weak":
        value["http"]["etag_before"] = value["http"]["etag_after"] = 'W/"weak"'
        path.write_text(json.dumps(value))
    elif change == "redirect":
        url += "?different-representation"
    else:
        key, replacement = {"etag": ("ETag", '"changed"'), "range": ("Accept-Ranges", "none"),
                            "length": ("Content-Length", "124")}[change]
        headers[key] = replacement
    monkeypatch.setattr(native, "urlopen", lambda *_args, **_kwargs: nullcontext(SimpleNamespace(
        getcode=lambda: 200, geturl=lambda: url, headers=headers,
    )))
    with pytest.raises(ValueError, match="validator|representation"):
        native._http_source_identity(value["source"]["url"], value["source"]["sha256"], index)


def test_remote_native_decode_keeps_url_native_pts_validator_and_input_bound(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "native"
    url, etag, sha = "https://example.invalid/source.mp4", '"verified"', "1" * 64
    identity = {"url": url, "source_sha256": sha, "etag": etag}
    pixels = bytes((40, 80, 120)) * 4
    digest = hashlib.sha256(pixels).hexdigest()
    commands = []

    def probe(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=json.dumps({"streams": [{"width": 2, "height": 2}]}))

    def decode(command, **_kwargs):
        commands.append(command)
        (output / "framehash.txt").write_text(f"#tb 0: 1/60000\n0, 601000, 601000, 1, 12, {digest}\n")
        return SimpleNamespace(stdout=BytesIO(pixels), wait=lambda **_kwargs: 0, poll=lambda: 0)

    monkeypatch.setattr(native.subprocess, "run", probe)
    monkeypatch.setattr(native.subprocess, "Popen", decode)
    frames = native._decode(url, output, sha, 10, 10., 11., source_identity=identity)
    assert frames[0].pts == 601000 and frames[0].seconds == Fraction(601000, 60000)
    assert frames[0].rgb_sha256 == digest
    for command in commands:
        assert command[command.index("-headers") + 1] == f"If-Match: {etag}\r\nAccept-Encoding: identity\r\n"
        assert command[command.index("-seekable") + 1] == "1"
    command = commands[1]
    assert command[command.index("-i") + 1] == url
    assert command.index("-to") < command.index("-i") and command[command.index("-to") + 1] == "11.0"
    assert json.loads((output / "identity.json").read_text())["http_source_identity"] == identity


@pytest.mark.parametrize("start,end", [(None, 11.), (10., None), (10., float("inf")), (11., 10.)])
def test_remote_decode_refuses_missing_or_nonfinite_nomination(tmp_path: Path, monkeypatch, start, end) -> None:
    monkeypatch.setattr(native.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("unbounded remote probe"))
    url = "https://example.invalid/source.mp4"
    with pytest.raises(ValueError, match="nominated finite interval"):
        native._decode(url, tmp_path, "1" * 64, 10, start, end,
                       source_identity={"url": url, "source_sha256": "1" * 64, "etag": '"verified"'})


def test_retained_automatic_producer_and_review_oracle(tmp_path: Path) -> None:
    retained = ROOT / "receipts/projected-source-correspondence-local-20261018"
    index = ROOT / "receipts/projected-source-correspondence-controls-20261018/source-index-0cd0e5a26643-c82b7271-091aebd1.npz"
    manifest_path = retained / "camera/identity.json"
    if not manifest_path.is_file() or not index.is_file():
        pytest.skip("named retained native oracle is not installed; this skip is not production acceptance")
    manifest = json.loads(manifest_path.read_text())
    camera = Path(manifest["command"][manifest["command"].index("-i") + 1])
    if not camera.is_file():
        pytest.skip("named native camera chunk is unavailable; no media substitution permitted")
    geometry = json.loads((retained / "geometry-fit.json").read_text())
    metadata = json.loads(index.with_suffix(".json").read_text())
    baseline = ROOT / "receipts/native-correspondence-support-repair-20261018/smoke.json"
    frozen = json.loads(baseline.read_text())
    assert all(frozen["real_negative_controls"].values())
    controls = tmp_path / "qualification-controls.json"
    control_rows = []
    for label, roles in (("known-blank", ["local_shape_null", "sequence_rejected"]),
                         ("known-decreasing", ["local_shape_null", "sequence_rejected"]),
                         ("known-rejected-ordered", ["sequence_rejected"])):
        identity = retained / label / "identity.json"
        control_rows.append({"identity_path": str(identity), "identity_sha256": native.sha256_file(identity),
            "decode_id": native.sha256_file(identity), "roles": roles,
            "applicability": {"camera_sha256": frozen["camera_sha256"], "geometry_sha256": frozen["geometry_sha256"]}})
    controls.write_text(json.dumps({"kind": "frozen-native-structural-controls-v1",
        "source_sha256": frozen["source_sha256"], "index_sha256": frozen["index_sha256"],
        "baseline_evidence": {"path": str(baseline), "sha256": native.sha256_file(baseline)}, "controls": control_rows}))
    result = native.produce_chunk_correspondence(
        camera_chunk=camera, projector_source=metadata["source"]["url"], source_index=index,
        projector_geometry=native.ProjectorGeometry(
            tuple(tuple(point) for point in geometry["fitted_polygon"]),
            frozen_interior_path=str(retained / "support-corrected/frozen-projection-interior.png")),
        output_dir=tmp_path, budget=native.CorrespondenceBudget(allow_source_decode=False),
        camera_bundle=retained / "camera",
        qualification_controls=controls,
        retained_source_bundles=tuple(retained / label for label in (
            "candidate-1", "candidate-2", "candidate-3", "known-blank", "known-decreasing", "known-rejected-ordered")),
    )
    assert result.search_coverage["camera_native_frames"] == 104
    assert result.search_coverage["camera_rgb_observations"] == 11
    assert result.search_coverage["retained_source_native_frames"] == 1722
    # The original automatic failure had no support at these exact native RGBs.
    expected = {
        0: "0c37123fdd67bcbb1c3c9ad286137e97858c5d878986e7a070a100be01eba4df",
        21: "be88caa598e404c36be8f877ef0af346116181a9296fbcb47cd619a093b0473d",
        103: "92e1ce554e2da57fcc73b204755046dcd4ebe85ac107ad23341d15db40e4e5ee",
    }
    frozen = np.asarray(Image.open(retained / "support-corrected/frozen-projection-interior.png")) > 0
    for record in result.frames:
        if record.camera_identity.frame_index not in expected:
            continue
        assert record.camera_identity.rgb_sha256 == expected[record.camera_identity.frame_index]
        support = np.asarray(Image.open(record.support_mask_path)) > 0
        assert support.any()
        assert not support[~frozen].any()
        assert not support[445:515, 55:130].any()
        assert support.sum() < frozen.sum() / 10  # not a giant wall/shadow rectangle
    positives = [row for row in result.sequence_evidence if row["structural_status"] == "supported"]
    assert len(positives) == 1, result.sequence_evidence
    positive = positives[0]
    assert 1559 <= positive["native_interval_seconds"][0] < 1566
    assert positive["variants"]["ordered"]["mean"] > max(
        row["variants"]["decreasing"]["mean"] for row in result.sequence_evidence)
    assert positive["variants"]["ordered"]["mean"] > positive["variants"]["exact_reversal"]["mean"]
    blank = next(row for row in result.sequence_evidence if row["family_id"] == "known-blank")
    competing = next(row for row in result.sequence_evidence if row["family_id"] == "known-rejected-ordered")
    assert positive["variants"]["ordered"]["mean"] > max(row["mean"] for row in blank["variants"].values())
    assert positive["variants"]["ordered"]["prefix"] > competing["variants"]["ordered"]["prefix"]
    assert positive["variants"]["ordered"]["suffix"] > competing["variants"]["ordered"]["suffix"]
    review = native._load_repository_script("cat_projector_label_review_server")

    frame = next(row for row in result.frames if row.camera_identity.frame_index == 52)
    consumed = review.correspondence_for_review_image(Path(frame.camera_identity.rgb_path), chunk=result,
                                                      camera_identity=frame.camera_identity)
    assert consumed.structural_status == "supported"
    assert consumed.measured_exposure_seconds is None
    assert len(consumed.candidate_families) == 1
    assert {row["family_id"] for row in result.sequence_evidence if row["structural_status"] == "rejected"} == {
        "known-blank", "known-decreasing", "known-rejected-ordered"}
    for record in result.frames:
        for family in record.candidate_families:
            assert family.family_id == "candidate-1"
            assert family.structural_status == "supported"
            assert len(family.source_frames) == family.qualification["coverage"]["qualified_native_states"]
            assert family.qualification["camera_identity"] == asdict(record.camera_identity)
            for state in family.qualification["native_states"]:
                assert state["conditional_path_mean"] > family.qualification["sequence_null_mean"]
                assert any(component["qualified"] and component["score"] > component["null_max"]
                           for component in state["component_scores"])
    restored = native.load_chunk_correspondence(Path(result.output_path))
    assert restored.frames == result.frames
