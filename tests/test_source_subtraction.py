from dataclasses import replace
import hashlib
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

MODULE_PATH = Path(__file__).resolve().parents[1] / "custom_components" / "cat_tv_play" / "source_subtraction.py"
SPEC = spec_from_file_location("cat_tv_play_source_subtraction", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
source_subtraction = module_from_spec(SPEC)
sys.modules[SPEC.name] = source_subtraction
SPEC.loader.exec_module(source_subtraction)

PROJECTOR_POLYGON = (
    (40.22, 57.27),
    (937.92, 101.0),
    (908.0, 599.0),
    (48.16, 680.97),
)


def _native_identity(tmp_path: Path, name: str, rgb: np.ndarray, media_sha: str):
    """Synthetic native bundles exercise the real identity loader, not acceptance evidence."""
    module = source_subtraction._source_correspondence_module()
    directory = tmp_path / name
    directory.mkdir()
    path = directory / "frame.png"
    Image.fromarray(rgb).save(path)
    rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    height, width = rgb.shape[:2]
    (directory / "framehash.txt").write_text(
        f"#tb 0: 1/25\n0, 0, 0, 1, {rgb.size}, {rgb_sha}\n"
    )
    manifest = directory / "identity.json"
    manifest.write_text(json.dumps({
        "exit_status": 0, "same_decode_hash_identity": True,
        "width": width, "height": height,
        "command": ["ffmpeg", "-copyts", "-filter_complex",
                    "[0:v]format=rgb24,split=2[pixels][identity]", "-fps_mode", "passthrough",
                    "-enc_time_base:v", "filter"],
        "frames": [{"native_pts": 0, "native_time_base": "1/25",
                    "rgb24_sha256": rgb_sha, "lossless_png": path.name}],
    }))
    return module.NativeFrameIdentity(
        media_sha, hashlib.sha256(manifest.read_bytes()).hexdigest(), 0, 0, 1, 25,
        width, height, rgb_sha, str(path), module.NATIVE_RGB_CONVERSION, str(manifest),
    )


@pytest.fixture
def explanation_case(tmp_path):
    pytest.importorskip("cv2")
    module = source_subtraction._source_correspondence_module()
    rgb = np.full((10, 12, 3), 208, dtype=np.uint8)
    rgb[2:8, 2:6] = (208, 32, 16)
    rgb[2:8, 7:10] = (16, 32, 208)
    camera = _native_identity(tmp_path, "camera", rgb, "a" * 64)
    source = _native_identity(tmp_path, "source", rgb, "b" * 64)
    mask = np.zeros((10, 12), dtype=bool)
    mask[3:7, 3:5] = True
    support = tmp_path / "support.png"
    Image.fromarray(mask.astype(np.uint8) * 255).save(support)
    family = module.SourceCandidateFamily("ordered", (source,), (0.0, 0.0), "supported")
    correspondence = module.FrameCorrespondence(
        camera, str(support), hashlib.sha256(support.read_bytes()).hexdigest(),
        (family,), "supported", (), "c" * 64,
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    bins = tuple(sorted(source_subtraction._candidate_rgb_bins(rgb, np.ones(mask.shape, dtype=bool))))
    digest = lambda label: hashlib.sha256(label.encode()).hexdigest()
    evidence = source_subtraction.PhotometricEvidence(
        model_id="projector-rgb10-v1",
        coefficients=(0.299, 0.587, 0.114, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        camera_media_sha256=camera.media_sha256, source_media_sha256=source.media_sha256,
        geometry_sha256=correspondence.geometry_sha256,
        fit_camera_rgb_sha256s=(digest("fit-camera"),),
        fit_source_rgb_sha256s=(digest("fit-source"),),
        fit_support_mask_sha256s=(digest("fit-support"),),
        fit_excluded_candidate_mask_sha256s=(digest("fit-exclusion"),),
        validation_camera_rgb_sha256s=(digest("validation-camera"),),
        validation_source_rgb_sha256s=(digest("validation-source"),),
        validation_support_mask_sha256s=(digest("validation-support"),),
        validation_excluded_candidate_mask_sha256s=(digest("validation-exclusion"),),
        fit_candidate_overlap_pixels=(0,), validation_candidate_overlap_pixels=(0,),
        fit_rgb_bins=bins, validation_rgb_bins=bins,
        fit_unclipped_luma_range=(47.0, 208.0), validation_unclipped_luma_range=(47.0, 208.0),
        validation_max_abs_error=1.0,
    )
    return Image.fromarray(rgb), correspondence, mask, evidence


def test_independent_response_explains_projected_pixels(explanation_case):
    image, correspondence, mask, evidence = explanation_case
    result = source_subtraction.explain_projected_candidate(
        camera_frame=image, correspondence=correspondence, candidate_mask=mask, photometry=evidence,
    )
    assert result.structural_valid and result.photometric_valid and result.explained
    assert result.candidate_error <= 1.0
    assert result.family_errors[0][0] == "ordered"


@pytest.mark.parametrize("columns", [(3, 5), (8, 9)], ids=["red-only", "blue-only"])
def test_candidate_color_domain_does_not_require_unrelated_colors(explanation_case, tmp_path, columns):
    image, correspondence, support, evidence = explanation_case
    candidate = np.zeros_like(support)
    candidate[3:7, columns[0]:columns[1]] = True
    support_path = tmp_path / "candidate-domain-support.png"
    Image.fromarray(candidate.astype(np.uint8) * 255).save(support_path)
    correspondence = replace(
        correspondence, support_mask_path=str(support_path),
        support_mask_sha256=hashlib.sha256(support_path.read_bytes()).hexdigest(),
    )
    bins = tuple(sorted(source_subtraction._candidate_rgb_bins(np.asarray(image), candidate)))
    evidence = replace(evidence, fit_rgb_bins=bins, validation_rgb_bins=bins)
    assert evidence.valid()
    result = source_subtraction.explain_projected_candidate(
        camera_frame=image, correspondence=correspondence,
        candidate_mask=candidate, photometry=evidence,
    )
    assert result.structural_valid and result.photometric_valid and result.explained
    assert result.candidate_error <= evidence.validation_max_abs_error


def test_neutral_only_calibration_does_not_cover_colored_candidate(explanation_case):
    image, correspondence, mask, evidence = explanation_case
    evidence = replace(
        evidence, fit_rgb_bins=(0xDDD, 0xEEE, 0xFFF), validation_rgb_bins=(0xDDD, 0xEEE, 0xFFF),
    )
    assert evidence.valid()
    result = source_subtraction.explain_projected_candidate(
        camera_frame=image, correspondence=correspondence, candidate_mask=mask, photometry=evidence,
    )
    assert not result.photometric_valid and not result.explained
    assert "candidate_color_outside_independent_support" in result.reasons


@pytest.mark.parametrize("missing_fold", ["fit", "validation"])
def test_candidate_domain_must_be_covered_in_each_independent_fold(explanation_case, missing_fold):
    image, correspondence, mask, evidence = explanation_case
    evidence = replace(evidence, **{f"{missing_fold}_rgb_bins": (0xDDD,)})
    assert evidence.valid()  # The shared neutral domain is valid, but not this red candidate.
    result = source_subtraction.explain_projected_candidate(
        camera_frame=image, correspondence=correspondence, candidate_mask=mask, photometry=evidence,
    )
    assert result.structural_valid and not result.photometric_valid and not result.explained
    assert result.reasons == ("candidate_color_outside_independent_support",)


def test_unsupported_candidate_color_is_not_extrapolated(explanation_case, monkeypatch):
    image, correspondence, mask, evidence = explanation_case
    evidence = replace(evidence, fit_rgb_bins=(0xDDD,), validation_rgb_bins=(0xDDD,))

    def unsupported_prediction(*args, **kwargs):
        pytest.fail("response must not be evaluated outside independently calibrated colors")

    monkeypatch.setattr(source_subtraction, "_predict_projector_rgb_luma", unsupported_prediction)
    result = source_subtraction.explain_projected_candidate(
        camera_frame=image, correspondence=correspondence, candidate_mask=mask, photometry=evidence,
    )
    assert result.structural_valid and not result.photometric_valid and not result.explained
    assert result.reasons == ("candidate_color_outside_independent_support",)


@pytest.mark.parametrize("fit_bins,validation_bins", [((), (0xDDD,)), ((0xDDD,), ()), ((0xD21,), (0x12D,))])
def test_calibration_requires_a_nonempty_shared_color_domain(explanation_case, fit_bins, validation_bins):
    image, correspondence, mask, evidence = explanation_case
    evidence = replace(evidence, fit_rgb_bins=fit_bins, validation_rgb_bins=validation_bins)
    assert not evidence.valid()
    result = source_subtraction.explain_projected_candidate(
        camera_frame=image, correspondence=correspondence, candidate_mask=mask, photometry=evidence,
    )
    assert not result.photometric_valid and not result.explained
    assert result.reasons == ("independent_photometric_evidence_unavailable",)


@pytest.mark.parametrize("occluded", [False, True])
def test_full_bbox_keeps_pixels_outside_structural_support(explanation_case, tmp_path, occluded):
    image, correspondence, support, evidence = explanation_case
    candidate = np.zeros_like(support)
    candidate[1:9, 1:11] = True
    assert np.any(candidate & ~support)
    rgb = np.asarray(image).copy()
    if occluded:
        # An occluder in a white bbox corner is outside the dark support, but
        # must still count rather than being cropped out of the explanation.
        rgb[1, 1] = 0
        camera = _native_identity(tmp_path, "bbox-occluded-camera", rgb, correspondence.camera_identity.media_sha256)
        correspondence = replace(correspondence, camera_identity=camera)
    result = source_subtraction.explain_projected_candidate(
        camera_frame=Image.fromarray(rgb), correspondence=correspondence,
        candidate_mask=candidate, photometry=evidence,
    )
    assert result.structural_valid and result.photometric_valid
    assert result.explained is (not occluded)
    if occluded:
        assert result.candidate_error > evidence.validation_max_abs_error
        assert "candidate_residual_exceeds_disjoint_validation_error" in result.reasons
    else:
        assert result.candidate_error <= evidence.validation_max_abs_error


def test_candidate_must_intersect_verified_structural_support(explanation_case):
    image, correspondence, support, evidence = explanation_case
    candidate = np.zeros_like(support)
    candidate[1, 1] = True
    assert not np.any(candidate & support)
    result = source_subtraction.explain_projected_candidate(
        camera_frame=image, correspondence=correspondence,
        candidate_mask=candidate, photometry=evidence,
    )
    assert not result.photometric_valid and not result.explained
    assert "candidate_disjoint_from_correspondence_support" in result.reasons


def test_invalid_native_source_geometry_cannot_be_constructed(explanation_case):
    _, correspondence, _, _ = explanation_case
    family = correspondence.candidate_families[0]
    with pytest.raises(ValueError, match="unsupported native geometry"):
        replace(family.source_frames[0], width=0)


def test_bbox_requires_native_projection_coverage_for_every_pixel(explanation_case):
    image, correspondence, support, evidence = explanation_case
    candidate = np.zeros_like(support)
    candidate[1:9, 1:11] = True
    correspondence = replace(
        correspondence,
        source_to_camera=((1.0, 0.0, 2.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    result = source_subtraction.explain_projected_candidate(
        camera_frame=image, correspondence=correspondence,
        candidate_mask=candidate, photometry=evidence,
    )
    assert not result.photometric_valid and not result.explained
    assert any("candidate outside native source projection coverage" in reason for reason in result.reasons)


def test_bbox_background_requires_independent_calibration_coverage(explanation_case):
    image, correspondence, support, evidence = explanation_case
    candidate = np.zeros_like(support)
    candidate[1:9, 1:11] = True
    # Keep red/blue and neutral calibration, but not this bbox's white bin.
    bins = tuple(value for value in evidence.fit_rgb_bins if value != 0xDDD) + (0xEEE,)
    evidence = replace(evidence, fit_rgb_bins=bins, validation_rgb_bins=bins)
    assert evidence.valid()
    result = source_subtraction.explain_projected_candidate(
        camera_frame=image, correspondence=correspondence,
        candidate_mask=candidate, photometry=evidence,
    )
    assert not result.photometric_valid and not result.explained
    assert "candidate_color_outside_independent_support" in result.reasons


def test_physical_object_mismatch_preserves_valid_calibration(explanation_case, tmp_path):
    image, correspondence, mask, evidence = explanation_case
    rgb = np.asarray(image).copy()
    rgb[mask] = 0
    camera = _native_identity(tmp_path, "occluded-camera", rgb, correspondence.camera_identity.media_sha256)
    result = source_subtraction.explain_projected_candidate(
        camera_frame=Image.fromarray(rgb), correspondence=replace(correspondence, camera_identity=camera),
        candidate_mask=mask, photometry=evidence,
    )
    assert result.structural_valid and result.photometric_valid
    assert not result.explained
    assert result.candidate_error > evidence.validation_max_abs_error
    assert result.family_errors[0][1][0] == result.candidate_error
    assert "candidate_residual_exceeds_disjoint_validation_error" in result.reasons


@pytest.mark.parametrize("invalid", ["missing", "overlap", "same-frame", "other-capture", "unclipped"])
def test_missing_or_invalid_calibration_cannot_explain(explanation_case, invalid):
    image, correspondence, mask, evidence = explanation_case
    if invalid == "missing":
        evidence = None
    elif invalid == "overlap":
        evidence = replace(evidence, validation_candidate_overlap_pixels=(1,))
    elif invalid == "same-frame":
        evidence = replace(evidence, fit_camera_rgb_sha256s=(correspondence.camera_identity.rgb_sha256,))
    elif invalid == "other-capture":
        evidence = replace(evidence, camera_media_sha256="d" * 64)
    elif invalid == "unclipped":
        evidence = replace(evidence, coefficients=(*evidence.coefficients[:-1], 1000.0))
    result = source_subtraction.explain_projected_candidate(
        camera_frame=image, correspondence=correspondence, candidate_mask=mask, photometry=evidence,
    )
    assert not result.photometric_valid
    assert not result.explained
    assert result.reasons


def test_blank_wall_cannot_explain_colored_object(explanation_case, tmp_path):
    image, correspondence, mask, evidence = explanation_case
    blank = np.full((image.height, image.width, 3), 208, dtype=np.uint8)
    source = _native_identity(tmp_path, "blank-source", blank, evidence.source_media_sha256)
    family = replace(correspondence.candidate_families[0], source_frames=(source,))
    result = source_subtraction.explain_projected_candidate(
        camera_frame=image, correspondence=replace(correspondence, candidate_families=(family,)),
        candidate_mask=mask, photometry=evidence,
    )
    assert result.photometric_valid
    assert not result.explained
    assert result.candidate_error > evidence.validation_max_abs_error


def test_ambiguity_cannot_choose_best_residual(explanation_case, tmp_path):
    image, correspondence, mask, evidence = explanation_case
    module = source_subtraction._source_correspondence_module()
    rgb = np.asarray(image).copy()
    rgb[mask] = (16, 32, 208)
    alternate = _native_identity(tmp_path, "alternate", rgb, evidence.source_media_sha256)
    family = module.SourceCandidateFamily("alternative", (alternate,), (0.0, 0.0), "supported")
    result = source_subtraction.explain_projected_candidate(
        camera_frame=image, correspondence=replace(correspondence, candidate_families=(*correspondence.candidate_families, family)),
        candidate_mask=mask, photometry=evidence,
    )
    assert result.photometric_valid and not result.explained
    errors = dict(result.family_errors)
    assert errors["ordered"][0] <= 1.0 < errors["alternative"][0]


def test_unresolved_alternative_cannot_explain(explanation_case):
    image, correspondence, mask, evidence = explanation_case
    family = replace(correspondence.candidate_families[0], family_id="unresolved", structural_status="unresolved")
    result = source_subtraction.explain_projected_candidate(
        camera_frame=image, correspondence=replace(correspondence, candidate_families=(*correspondence.candidate_families, family)),
        candidate_mask=mask, photometry=evidence,
    )
    assert not result.explained
    assert "surviving_source_family_incomplete" in result.reasons


def test_camera_rgb_substitution_cannot_explain(explanation_case):
    image, correspondence, mask, evidence = explanation_case
    result = source_subtraction.explain_projected_candidate(
        camera_frame=Image.new("RGB", image.size), correspondence=correspondence,
        candidate_mask=mask, photometry=evidence,
    )
    assert not result.explained
    assert "camera_rgb_identity_mismatch" in result.reasons


def test_room_background_keeps_static_cat_out_of_learning():
    empty = Image.new("L", (100, 80), 120)
    with_cat = empty.copy()
    ImageDraw.Draw(with_cat).rectangle((40, 55, 60, 75), fill=20)
    mask = np.ones((80, 100), dtype=bool)
    mask[45:, 30:70] = False
    background = source_subtraction.build_room_background([empty, with_cat], update_mask=mask)
    assert background.frame_count == 2
    assert np.asarray(background.image)[60, 50] == 120
    assert np.asarray(with_cat)[60, 50] == 20


def test_background_update_mask_keeps_possible_cat_path_out_of_average() -> None:
    base = Image.new("L", (100, 80), 120)
    current = Image.new("L", (100, 80), 180)
    ImageDraw.Draw(current).rectangle((40, 55, 60, 75), fill=20)
    update_mask = np.ones((80, 100), dtype=bool)
    update_mask[45:80, 30:70] = False

    updated = source_subtraction.update_room_background(base, current, update_mask=update_mask, alpha=0.5)
    updated_arr = np.asarray(updated)

    assert updated_arr[10, 10] == 150
    assert updated_arr[60, 50] == 120


def test_background_rejects_incompatible_update_mask():
    with pytest.raises(ValueError, match="background update mask shape"):
        source_subtraction.build_room_background(
            [Image.new("L", (100, 80))], update_mask=np.ones((10, 10), dtype=bool),
        )


def test_projector_rgb_mapping_keeps_bounded_channel_weights() -> None:
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(80, 100, 3), dtype=np.uint8)
    mask = np.ones((80, 100), dtype=bool)
    target = np.clip(0.04 * rgb[..., 0] + 0.62 * rgb[..., 1] + 0.18 * rgb[..., 2] + 21, 0, 255).astype(np.uint8)

    coefficients = source_subtraction._fit_projector_rgb_to_camera_luma(rgb, target, mask)

    assert coefficients is not None
    assert np.all(np.abs(coefficients[:-1]) <= 4.0)
    prediction = source_subtraction._projector_rgb_features(rgb) @ coefficients
    assert float(np.percentile(np.abs(prediction - target), 95)) < 3.0


def test_projector_rgb_mapping_subsamples_large_fit_masks() -> None:
    rng = np.random.default_rng(43)
    rgb = rng.integers(40, 240, size=(180, 240, 3), dtype=np.uint8)
    mask = np.ones((180, 240), dtype=bool)
    target = np.clip(0.12 * rgb[..., 0] + 0.48 * rgb[..., 1] + 0.22 * rgb[..., 2] + 17, 0, 255).astype(np.uint8)

    coefficients = source_subtraction._fit_projector_rgb_to_camera_luma(
        rgb,
        target,
        mask,
        max_fit_pixels=3000,
    )

    assert coefficients is not None
    assert np.all(np.abs(coefficients[:-1]) <= 4.0)
    prediction = source_subtraction._projector_rgb_features(rgb) @ coefficients
    assert float(np.percentile(np.abs(prediction - target), 95)) < 4.0


def test_projector_rgb_mapping_handles_mono_ir_like_camera_response() -> None:
    rng = np.random.default_rng(44)
    rgb = rng.integers(5, 256, size=(120, 160, 3), dtype=np.uint8)
    mask = np.ones((120, 160), dtype=bool)
    rgb_f = rgb.astype(np.float32)
    channel_max = rgb_f.max(axis=2)
    channel_min = rgb_f.min(axis=2)
    saturation = channel_max - channel_min
    sqrt_channel_max = np.sqrt(channel_max / 255.0) * 255.0
    target = np.clip(
        0.22 * rgb_f[..., 1] + 0.28 * channel_max + 0.10 * saturation + 0.16 * sqrt_channel_max + 9,
        0,
        255,
    ).astype(np.uint8)

    coefficients = source_subtraction._fit_projector_rgb_to_camera_luma(
        rgb,
        target,
        mask,
        max_fit_pixels=4000,
    )

    assert coefficients is not None
    assert np.all(np.abs(coefficients[:-1]) <= 4.0)
    prediction = source_subtraction._projector_rgb_features(rgb) @ coefficients
    assert float(np.percentile(np.abs(prediction - target), 95)) < 5.0


def test_robust_component_top_ignores_single_high_noise_pixel() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[10:30, 15:25] = 1
    labels[2, 5] = 1

    top = source_subtraction.robust_component_top_px(labels, 1)

    assert top is not None
    top_x, top_y = top
    assert 15 <= top_x <= 24
    assert 10 <= top_y <= 15


def test_robust_component_top_rejects_tiny_components() -> None:
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[4:6, 4:6] = 1

    assert source_subtraction.robust_component_top_px(labels, 1) is None
