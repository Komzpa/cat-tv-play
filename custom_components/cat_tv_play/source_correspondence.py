"""Bounded, content-only native camera/source correspondence.

The compact index nominates windows, never frame identities or a playback clock.
Native RGB and PTS from one decode refine those nominations. Structural support
is not photometric validity and never grants source-based privacy permission.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
INDEX_DECODE = "ffmpeg -i URL -vf fps=4 -pix_fmt gray -f rawvideo pipe:1 @1280x720 warp_source_to_camera"
# Captured before the residual cutover, while the whole-file digest equalled
# the retained index's generation dependency. These are NOT replacement keys.
LEGACY_GEOMETRY_DEPENDENCY = "19097e4f8f22b88015139281e13c62a698c4c0189a098b65984f3aa3f195b679"
GEOMETRY_SEGMENTS = {
    "projector_polygon_for_size": "ddec0bb82abfea959546c031b2203d49f24701711d79f8081ffef7e1c669fc77",
    "projection_mask": "3213831407d9d7857aa8e55da6bc6b8ff9c66630764f8b594430ca0ce2942157",
    "warp_source_to_camera": "e5965db82ee4c30390c09e7266070f2d8f7d4c83f44a01d74d80c06506c2daed",
}
LEGACY_IMPLEMENTATION_KEY = "091aebd12df88e16462cc2c6cf84dd3fd91f3bc9702a46947bc2944fa3dcfd3e"
LEGACY_GENERATION_SCRIPT_SHA256 = "e3552e17807e070dbdbba5d7379d91471f9116b0949abe99a8fae90c131c714e"
LEGACY_MASK_SOURCE_SHA256 = "82a35d6dde7f984c8aff5f014d45b3025667f0329d0b1140eae0d0d6d45d60e1"
LEGACY_DESCRIPTOR_SOURCE_SHA256 = "137374f3b9f8452e6cbb08d51dab553ea85f7f11aba744cadbeaa2340f65db36"
NATIVE_RGB_CONVERSION = "format=rgb24;native;passthrough"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class NativeFrameIdentity:
    media_sha256: str
    decode_id: str
    frame_index: int
    pts: int
    time_base_num: int
    time_base_den: int
    width: int
    height: int
    rgb_sha256: str
    rgb_path: str
    conversion: str
    identity_path: str = ""

    def __post_init__(self) -> None:
        if type(self.pts) is not int or type(self.frame_index) is not int:
            raise ValueError("native integer PTS/frame index required")
        if self.frame_index < 0 or self.time_base_num <= 0 or self.time_base_den <= 0:
            raise ValueError("invalid native frame/time base")
        if self.width <= 0 or self.height <= 0 or self.conversion != "format=rgb24;native;passthrough":
            raise ValueError("unsupported native geometry/conversion")
        for value in (self.media_sha256, self.decode_id, self.rgb_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("SHA256 native identity required")

    @property
    def seconds(self) -> Fraction:
        return self.pts * Fraction(self.time_base_num, self.time_base_den)


@dataclass(frozen=True)
class SourceCandidateFamily:
    family_id: str
    source_frames: tuple[NativeFrameIdentity, ...]
    candidate_interval: tuple[float, float]
    structural_status: str
    reasons: tuple[str, ...] = ()
    qualification: dict[str, Any] | None = None


@dataclass(frozen=True)
class FrameCorrespondence:
    camera_identity: NativeFrameIdentity
    support_mask_path: str | None
    support_mask_sha256: str | None
    candidate_families: tuple[SourceCandidateFamily, ...]
    structural_status: str
    reasons: tuple[str, ...]
    geometry_sha256: str
    source_to_camera: tuple[tuple[float, ...], ...]
    measured_exposure_seconds: float | None = None
    component_qualifications: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ProjectorGeometry:
    polygon_1280: tuple[tuple[float, float], ...]
    # Frozen camera geometry: screen-edge erosion and separately known socket.
    excluded_regions_1280: tuple[tuple[int, int, int, int], ...] = ((55, 445, 130, 515),)
    interior_margin_px: int = 15
    frozen_interior_path: str | None = None

    def __post_init__(self) -> None:
        if self.frozen_interior_path:
            object.__setattr__(self, "frozen_interior_path", str(Path(self.frozen_interior_path).expanduser().resolve()))

    @property
    def sha256(self) -> str:
        value = asdict(self)
        value["frozen_interior_sha256"] = sha256_file(Path(self.frozen_interior_path)) if self.frozen_interior_path else None
        return _digest(value)


@dataclass(frozen=True)
class CorrespondenceBudget:
    max_source_windows: int = 3
    index_window_slots: int = 24
    max_camera_frames: int = 512
    max_source_frames: int = 1800
    discovery_observations: int = 11
    allow_source_decode: bool = True


@dataclass(frozen=True)
class ChunkCorrespondence:
    camera_sha256: str
    source_sha256: str
    index_sha256: str
    geometry_sha256: str
    producer_sha256: str
    frames: tuple[FrameCorrespondence, ...]
    provenance: dict[str, Any]
    search_coverage: dict[str, Any]
    sequence_evidence: tuple[dict[str, Any], ...]
    output_path: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _http_source_identity(media: str, source_sha256: str, source_index: Path) -> dict[str, Any]:
    """Check a server-hashed representation, never promote a cached source key."""
    path = source_index.with_name(f"http-source-identity-{source_sha256}.json")
    raw = path.read_bytes()
    value = json.loads(raw)
    if (value.get("kind") != "server-sha256-bound-http-source-identity"
            or value["source"] != {"url": media, "sha256": source_sha256}):
        raise ValueError("HTTP source identity does not bind this URL and source SHA256")
    proof, http = value["server_proof"], value["http"]
    hashed = proof["sha256sum_stdout"].split(maxsplit=1)
    length, etag = http["content_length_before"], http["etag_before"]
    if (proof["exit_status"] != 0 or len(hashed) != 2
            or hashed != [source_sha256, proof["source_file"]]
            or proof["stat_before"] != proof["stat_after"]
            or int(proof["stat_before"].split(maxsplit=1)[0]) != length
            or type(length) is not int or length <= 0
            or http["content_length_after"] != length or http["etag_after"] != etag
            or http["accept_ranges"] != "bytes"
            or not isinstance(etag, str) or not etag.startswith('"') or not etag.endswith('"')
            or len(etag) < 3 or "\r" in etag or "\n" in etag):
        raise ValueError("server SHA256 and stable strong HTTP validator proof required")
    request = Request(media, method="HEAD", headers={
        "If-Match": etag, "Accept-Encoding": "identity", "Cache-Control": "no-cache",
    })
    with urlopen(request, timeout=15) as response:
        if (response.getcode() != 200 or response.geturl() != media
                or response.headers.get("ETag") != etag
                or response.headers.get("Content-Length") != str(length)
                or response.headers.get("Accept-Ranges") != "bytes"
                or response.headers.get("Content-Encoding", "identity") != "identity"):
            raise ValueError("current HTTP source representation changed or is not range-addressable")
    return {"url": media, "source_sha256": source_sha256, "etag": etag, "content_length": length,
            "identity_path": str(path.resolve()), "identity_sha256": hashlib.sha256(raw).hexdigest()}


def _framehash_rows(text: str) -> tuple[Fraction, list[tuple[int, int, str]]]:
    lines = text.splitlines()
    bases = [Fraction(line.split(":", 1)[1].strip()) for line in lines if line.startswith("#tb 0:")]
    if len(bases) != 1 or bases[0] <= 0:
        raise ValueError("same-decode native rational time base required")
    rows = []
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        stream, _dts, pts, _duration, byte_count, digest = [part.strip() for part in line.split(",")]
        if stream != "0" or len(digest) != 64:
            raise ValueError("invalid native framehash identity")
        rows.append((int(pts), int(byte_count), digest))
    return bases[0], rows


@lru_cache(maxsize=32)
def _verified_manifest(path_string: str, size: int, mtime_ns: int, digest: str) -> dict[str, Any]:
    path = Path(path_string)
    if sha256_file(path) != digest:
        raise ValueError("native decode manifest changed")
    value = _read_json(path)
    if value.get("exit_status") != 0 or value.get("same_decode_hash_identity") is not True:
        raise ValueError("same-decode RGB/hash proof required")
    command = value["command"]
    if "-copyts" not in command or "-fps_mode" not in command or "filter" not in command:
        raise ValueError("native passthrough conversion provenance missing")
    graph = command[command.index("-filter_complex") + 1]
    if "format=rgb24,split=2[pixels][identity]" not in graph or "fps=" in graph or "scale=" in graph:
        raise ValueError("unsupported native conversion provenance")
    time_base, hashed = _framehash_rows((path.parent / "framehash.txt").read_text())
    if len(hashed) != len(value["frames"]):
        raise ValueError("native framehash count mismatch")
    for row, (pts, byte_count, rgb_hash) in zip(value["frames"], hashed, strict=True):
        if (type(row.get("native_pts")) is not int or row["native_pts"] != pts
                or Fraction(row["native_time_base"]) != time_base
                or row["rgb24_sha256"] != rgb_hash
                or int(byte_count) != value["width"] * value["height"] * 3):
            raise ValueError("native PTS/RGB identity mismatch")
    value["framehash_sha256"] = sha256_file(path.parent / "framehash.txt")
    return value


def load_native_rgb(identity: NativeFrameIdentity) -> Image.Image:
    path = Path(identity.identity_path) if identity.identity_path else Path(identity.rgb_path).parent / "identity.json"
    stat = path.stat()
    manifest = _verified_manifest(str(path), stat.st_size, stat.st_mtime_ns, identity.decode_id)
    row = manifest["frames"][identity.frame_index]
    if (row["native_pts"] != identity.pts or Fraction(row["native_time_base"]) != Fraction(identity.time_base_num, identity.time_base_den)
            or row["rgb24_sha256"] != identity.rgb_sha256
            or manifest["width"] != identity.width or manifest["height"] != identity.height
            or not row["lossless_png"] or (path.parent / row["lossless_png"]).resolve() != Path(identity.rgb_path).resolve()
            or sha256_file(path.parent / "framehash.txt") != manifest["framehash_sha256"]):
        raise ValueError("substituted native identity")
    with Image.open(identity.rgb_path) as image:
        rgb = image.convert("RGB")
    if rgb.size != (identity.width, identity.height) or hashlib.sha256(rgb.tobytes()).hexdigest() != identity.rgb_sha256:
        raise ValueError("substituted native RGB pixels")
    return rgb


def _bundle(path: Path, media: str, media_sha256: str) -> tuple[NativeFrameIdentity, ...]:
    manifest_path = path / "identity.json"
    stat = manifest_path.stat()
    digest = sha256_file(manifest_path)
    value = _verified_manifest(str(manifest_path), stat.st_size, stat.st_mtime_ns, digest)
    command = value["command"]
    if command[command.index("-i") + 1] != media:
        raise ValueError("native bundle/media provenance mismatch")
    if value.get("media_sha256", media_sha256) != media_sha256:
        raise ValueError("native bundle/media SHA mismatch")
    frames = []
    for index, row in enumerate(value["frames"]):
        if row["decode_frame_index"] != index:
            raise ValueError("native decode ordering mismatch")
        base = Fraction(row["native_time_base"])
        frames.append(NativeFrameIdentity(media_sha256, digest, index, row["native_pts"], base.numerator, base.denominator,
                                          value["width"], value["height"], row["rgb24_sha256"],
                                          str((path / row["lossless_png"]).resolve()) if row["lossless_png"] else "",
                                          "format=rgb24;native;passthrough", str(manifest_path.resolve())))
    if any(b.seconds <= a.seconds for a, b in zip(frames, frames[1:])):
        raise ValueError("native PTS must increase within a decode")
    return tuple(frames)


def _decode(media: Path | str, output: Path, media_sha256: str, max_frames: int,
            start: float | None = None, end: float | None = None,
            source_identity: dict[str, Any] | None = None) -> tuple[NativeFrameIdentity, ...]:
    """One bounded native rgb24 split, checked against the same raw bytes."""
    options = []
    if urlparse(str(media)).scheme in {"http", "https"}:
        if (source_identity is None or source_identity["url"] != str(media)
                or source_identity["source_sha256"] != media_sha256
                or start is None or end is None or not math.isfinite(start) or not math.isfinite(end)
                or not 0 <= start < end):
            raise ValueError("remote native decode requires verified identity and a nominated finite interval")
        options = ["-rw_timeout", "15000000", "-seekable", "1", "-headers",
                   f"If-Match: {source_identity['etag']}\r\nAccept-Encoding: identity\r\n"]
    probe = subprocess.run(["ffprobe", "-v", "error", *options, "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "json", str(media)], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    output.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "2", "-copyts"]
    if start is not None:
        command += ["-ss", str(start), "-to", str(end)]
    graph = "format=rgb24"
    if start is not None:
        graph = f"trim=start={start}:end={end}," + graph
    command += [*options, "-i", str(media), "-filter_complex_threads", "1", "-filter_complex", f"[0:v]{graph},split=2[pixels][identity]",
                "-map", "[pixels]", "-an", "-c:v", "rawvideo", "-pix_fmt", "rgb24", "-enc_time_base:v", "filter",
                "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1", "-map", "[identity]", "-an", "-c:v", "rawvideo",
                "-pix_fmt", "rgb24", "-enc_time_base:v", "filter", "-fps_mode", "passthrough", "-f", "framehash", "-hash", "sha256",
                str(output / "framehash.txt")]
    digests = []
    with (output / "decode.log").open("wb") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
        try:
            assert process.stdout is not None
            frame_bytes = width * height * 3
            while True:
                data = bytearray()
                while len(data) < frame_bytes:
                    part = process.stdout.read(frame_bytes - len(data))
                    if not part:
                        break
                    data.extend(part)
                if not data:
                    break
                if len(data) != frame_bytes or len(digests) >= max_frames:
                    raise ValueError("native decode incomplete or frame budget exhausted")
                digest = hashlib.sha256(data).hexdigest()
                Image.frombytes("RGB", (width, height), bytes(data)).save(output / f"frame-{len(digests):04d}.png")
                digests.append(digest)
            status = process.wait(timeout=90)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    if status:
        raise ValueError(f"native decode failed with exit {status}")
    base, hashes = _framehash_rows((output / "framehash.txt").read_text())
    rows = []
    for pts, byte_count, digest in hashes:
        index = len(rows)
        if index >= len(digests) or digest != digests[index] or int(byte_count) != width * height * 3:
            raise ValueError("same-decode native RGB/framehash mismatch")
        rows.append({"decode_frame_index": index, "native_pts": int(pts), "native_time_base": str(base),
                     "rgb24_sha256": digest, "lossless_png": f"frame-{index:04d}.png"})
    if len(rows) != len(digests):
        raise ValueError("same-decode native frame count mismatch")
    _save_json(output / "identity.json", {"command": command, "exit_status": status, "same_decode_hash_identity": True,
                                         "media_sha256": media_sha256, "width": width, "height": height, "frames": rows,
                                         "http_source_identity": source_identity})
    return _bundle(output, str(media), media_sha256)


def load_source_index(index_path: Path, source_sha256: str, source_media: str | None = None) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    metadata = _read_json(index_path.with_suffix(".json"))
    if source_media is not None and metadata["source"]["url"] != source_media:
        raise ValueError("retained index source/media provenance mismatch")
    keys = metadata["keys"]
    if keys["source_sha256"] != source_sha256 or metadata["source"]["sha256"] != source_sha256:
        raise ValueError("cached source SHA key mismatch")
    if _digest(metadata["geometry"]) != keys["geometry_sha256"]:
        raise ValueError("stale index geometry key")
    implementation = keys["implementation_sha256"]
    if implementation != LEGACY_IMPLEMENTATION_KEY or metadata["implementation"]["index_decode"] != INDEX_DECODE:
        raise ValueError("stale descriptor implementation key; compatible index rebuild required")
    # Exact descriptor source was captured from the retained generation. Runtime
    # does not import or depend on a disposable diagnostic/receipt implementation.
    own = Path(__file__).read_text()
    own_descriptor = next(ast.get_source_segment(own, node) + "\n" for node in ast.parse(own).body
                          if isinstance(node, ast.FunctionDef) and node.name == "shape_descriptor")
    if hashlib.sha256(own_descriptor.encode()).hexdigest() != LEGACY_DESCRIPTOR_SOURCE_SHA256:
        raise ValueError("descriptor changed; compatible index rebuild required")
    dependency = ROOT / "custom_components/cat_tv_play/source_subtraction.py"
    current = dependency.read_text()
    segments = {node.name: hashlib.sha256(ast.get_source_segment(current, node).encode()).hexdigest()
                for node in ast.parse(current).body if isinstance(node, ast.FunctionDef) and node.name in GEOMETRY_SEGMENTS}
    old_dependency = metadata["geometry"]["source_subtraction_sha256"]
    if old_dependency != LEGACY_GEOMETRY_DEPENDENCY or segments != GEOMETRY_SEGMENTS:
        raise ValueError("geometry dependency changed; compatible index rebuild required")
    migration = {"original_generation_keys": keys, "original_geometry_dependency_sha256": old_dependency,
                 "current_geometry_dependency_sha256": sha256_file(dependency), "unchanged_geometry_source_segments": segments,
                 "generation_script_sha256": LEGACY_GENERATION_SCRIPT_SHA256,
                 "generation_mask_source_sha256": LEGACY_MASK_SOURCE_SHA256,
                 "generation_descriptor_source_sha256": LEGACY_DESCRIPTOR_SOURCE_SHA256,
                 "generation_descriptor_implementation_sha256": implementation,
                 "index_rekeyed": False, "migration": "unchanged exact geometric dependency source segments"}
    with np.load(index_path, allow_pickle=False) as archive:
        for key, expected in keys.items():
            if str(archive[f"key_{key}"]) != expected:
                raise ValueError(f"stale cached index {key}")
        features = archive["features"]
    if features.shape != (metadata["slots"], metadata["descriptor_dim"]) or not np.isfinite(features).all():
        raise ValueError("invalid retained index descriptor matrix")
    if metadata["geometry"]["descriptor_grid"] != [48, 32] or metadata["geometry"]["exclusions"] != [0.69, 0.82]:
        raise ValueError("unsupported retained geometry schema")
    return features, metadata, migration


def shape_descriptor(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compact local-contrast and gradient shape feature, invariant to global luma."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("projector interior mask is empty")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    support = mask[y0:y1, x0:x1]
    small = cv2.resize(crop, (48, 32), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(support.astype(np.uint8), (48, 32), interpolation=cv2.INTER_NEAREST).astype(bool)
    local = small - cv2.GaussianBlur(small, (0, 0), 1.0)
    dx = cv2.Sobel(local, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(local, cv2.CV_32F, 0, 1, ksize=3)
    values = np.concatenate((local[small_mask], dx[small_mask], dy[small_mask])).astype(np.float32)
    values -= values.mean()
    norm = float(np.linalg.norm(values))
    return values / norm if norm else values


def _projection(polygon: tuple[tuple[float, float], ...], width: int, height: int) -> np.ndarray:
    image = Image.new("L", (width, height))
    ImageDraw.Draw(image).polygon(polygon, fill=255)
    return np.asarray(image) > 0


def _homography(polygon: tuple[tuple[float, float], ...]) -> np.ndarray:
    return cv2.getPerspectiveTransform(np.float32([[0, 0], [1279, 0], [1279, 719], [0, 719]]), np.float32(polygon))


def _local_values(matrix: np.ndarray, columns: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix[..., columns], np.float32)
    values = values - values.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def _load_repository_script(script_name: str) -> Any:
    """Load the actual repository owner, not an unrelated installed scripts package."""
    module_name = f"cat_tv_play_correspondence_{script_name}"
    if module_name not in sys.modules:
        path = ROOT / "scripts" / f"{script_name}.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load canonical script {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def _camera_support(image: Image.Image, interior: np.ndarray, excluded: np.ndarray,
                    temporal_reference: np.ndarray | None = None) -> tuple[np.ndarray, tuple[tuple[int, int, int, int], ...]]:
    """Camera-only structural pixels, never prey truth or a no-cat background.

    A same-sequence temporal median removes stationary wall shading. Its dark
    departures nominate changing structure, including occluders and shadows;
    only the independent source oracle can test correspondence afterwards.
    """
    gray = np.asarray(image.convert("L"), np.float32)
    allowed = interior & ~excluded
    if not allowed.any():
        return np.zeros_like(allowed), ()
    if temporal_reference is not None:
        residual = temporal_reference - cv2.GaussianBlur(gray, (0, 0), 1)
        residual -= np.median(residual[allowed])
        noise = 1.4826 * np.median(np.abs(residual[allowed]))
        # Six robust noise deviations, with a one-code-value quantization floor.
        mask = residual > max(6 * noise, 1.)
    else:
        # Existing detector's interior dark-body rule, retaining pixels rather
        # than turning its bounding rectangle into structural evidence.
        mask = gray < min(140, float(np.median(gray[allowed])) - 35)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask & allowed).astype(np.uint8))
    support = np.zeros_like(allowed)
    boxes = []
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        # Reuse the detector's component-size requirements. Exclusions apply
        # per pixel; no socket-crossing bounding rectangle is admitted wholesale.
        if area < 120 or width < 8 or height < 8:
            continue
        support |= labels == label
        boxes.append((x, y, x + width, y + height))
    return support, tuple(boxes)


def ordered_native_path(scores: np.ndarray, discovery: tuple[int, ...], direction: int = 1) -> tuple[int, ...]:
    """Content-only non-strict monotone path; repeated native IDs are retained."""
    if direction not in {-1, 1} or not discovery:
        raise ValueError("native path requires discovery observations and direction")
    matrix = scores if direction > 0 else scores[::-1]

    def solve(nodes: np.ndarray) -> tuple[int, ...]:
        value = nodes[:, 0].copy()
        parents = []
        for column in range(1, nodes.shape[1]):
            maxima = np.maximum.accumulate(value)
            improved = np.r_[True, maxima[1:] > maxima[:-1]]
            best_index = np.maximum.accumulate(np.where(improved, np.arange(len(value)), 0))
            parents.append(best_index)
            value = value[best_index] + nodes[:, column]
        selected = [int(np.argmax(value))]
        for parent in reversed(parents):
            selected.append(int(parent[selected[-1]]))
        return tuple(reversed(selected))

    anchors = dict(zip(discovery, solve(matrix[:, discovery]), strict=True))
    bounded = np.full(matrix.shape, -np.inf, np.float64)
    for slot in range(matrix.shape[1]):
        if slot in anchors:
            bounded[anchors[slot], slot] = matrix[anchors[slot], slot]
            continue
        before = [d for d in discovery if d < slot]
        after = [d for d in discovery if d > slot]
        lo = anchors[before[-1]] if before else 0
        hi = anchors[after[0]] if after else len(matrix) - 1
        bounded[lo:hi + 1, slot] = matrix[lo:hi + 1, slot]
    # Complete ALL held-out observations together. Independent within-bracket
    # argmaxes can cross when several observations lie between two anchors.
    selected = solve(bounded)
    return selected if direction > 0 else tuple(len(scores) - 1 - index for index in selected)


def ordered_state_path_totals(scores: np.ndarray) -> np.ndarray:
    """Best complete non-strict ordered path THROUGH every native state.

    Prefix and suffix include the constrained node, so subtract it once. All
    arithmetic is in total-score units; callers divide by observation count.
    Local support is a separate per-state gate, never inferred from this total.
    """
    matrix = np.asarray(scores, np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("finite nonempty native score matrix required")
    prefix, suffix = np.empty_like(matrix), np.empty_like(matrix)
    prefix[:, 0], suffix[:, -1] = matrix[:, 0], matrix[:, -1]
    for slot in range(1, matrix.shape[1]):
        prefix[:, slot] = matrix[:, slot] + np.maximum.accumulate(prefix[:, slot - 1])
    for slot in range(matrix.shape[1] - 2, -1, -1):
        suffix[:, slot] = matrix[:, slot] + np.maximum.accumulate(suffix[::-1, slot + 1])[::-1]
    return prefix + suffix - matrix


def _qualification_controls(path: Path | None, source_index: Path, source_sha: str,
                            camera_sha: str, geometry_sha: str) -> tuple[dict[str, Any], dict[str, set[str]]]:
    """Validate source-index sidecar controls, with capture-specific applicability.

    An explicit receipt is useful for a frozen oracle. Normal callers discover
    only the canonical sidecar next to their configured index, never diagnostic
    bundle names or a repository-wide search. Missing evidence is not success.
    """
    path = Path(path) if path else source_index.with_suffix(".qualification-controls.json")
    if not path.is_file():
        return {"status": "unavailable", "path": str(path),
                "reason": "source-null evidence is not configured; internal producer evidence-wiring gap, not unavailable source data"}, {}
    value = _read_json(path)
    if (value.get("kind") != "frozen-native-structural-controls-v1"
            or value.get("source_sha256") != source_sha
            or value.get("index_sha256") != sha256_file(source_index)):
        raise ValueError("qualification controls source/index binding mismatch")
    baseline = value["baseline_evidence"]
    if sha256_file(Path(baseline["path"])) != baseline["sha256"]:
        raise ValueError("qualification controls baseline evidence changed")
    roles: dict[str, set[str]] = {}
    applicable, inapplicable = [], []
    source_facts = []
    for fact in value.get("source_facts", ()):
        if fact.get("kind") != "source-null-native-sample-v1" or fact.get("source_sha256") != source_sha:
            raise ValueError("unknown or substituted source-only null fact")
        identity = Path(fact["identity_path"])
        if sha256_file(identity) != fact["identity_sha256"] or fact["decode_id"] != fact["identity_sha256"]:
            raise ValueError("source-only native null manifest changed")
        frames = _bundle(identity.parent, _read_json(source_index.with_suffix(".json"))["source"]["url"], source_sha)
        # A source-only null sample is not a no-cat background or a July camera
        # rejection. Recheck its narrowly declared source-pixel property, then
        # recompute its gradient envelope under the CURRENT camera geometry.
        for frame in frames:
            rgb = load_native_rgb(frame)
            whole = np.ones((frame.height, frame.width), bool)
            support, _boxes = _camera_support(rgb, whole, np.zeros_like(whole))
            if support.any():
                raise ValueError("source-only null sample contains a native dark-body proposal")
        roles.setdefault(fact["decode_id"], set()).add("local_shape_null")
        source_facts.append({"decode_id": fact["decode_id"], "identity_path": str(identity),
                             "native_identities": [asdict(frame) for frame in frames],
                             "verified_property": "no native dark-body proposals under the existing camera-only rule",
                             "native_RGB_frames_verified": len(frames), "scope": "source pixels only; no camera/calibration transfer"})
    for entry in value["controls"]:
        identity = Path(entry["identity_path"])
        if sha256_file(identity) != entry["identity_sha256"]:
            raise ValueError("qualification control native manifest changed")
        frames = _bundle(identity.parent, _read_json(source_index.with_suffix(".json"))["source"]["url"], source_sha)
        if not frames or frames[0].decode_id != entry["decode_id"]:
            raise ValueError("qualification control native decode mismatch")
        scope = entry["applicability"]
        if scope != {"camera_sha256": camera_sha, "geometry_sha256": geometry_sha}:
            inapplicable.append(entry["decode_id"])
            continue
        declared = set(entry["roles"])
        if not declared or declared - {"local_shape_null", "sequence_rejected"}:
            raise ValueError("unknown qualification control role")
        roles.setdefault(entry["decode_id"], set()).update(declared)
        applicable.append(entry["decode_id"])
    return {"status": "applicable" if roles else "inapplicable", "path": str(path.resolve()),
            "sha256": sha256_file(path), "baseline_evidence": baseline,
            "source_only_facts": source_facts,
            "applicable_decode_ids": applicable, "inapplicable_decode_ids": inapplicable}, roles


def _gradient_vector(gray: np.ndarray, box: tuple[int, int, int, int], interior: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = cv2.GaussianBlur(gray[y0:y1, x0:x1].astype(np.float32), (0, 0), 1)
    mask = interior[y0:y1, x0:x1]
    vector = np.concatenate((cv2.Sobel(crop, cv2.CV_32F, 1, 0)[mask], cv2.Sobel(crop, cv2.CV_32F, 0, 1)[mask]))
    norm = np.linalg.norm(vector)
    return vector / norm if norm else np.zeros_like(vector)


def evaluate_native_sequence(scores: np.ndarray, indices: tuple[int, ...]) -> dict[str, Any]:
    """The same object-gradient oracle for every ordered/blank/reversed control."""
    if len(indices) != scores.shape[1] or any(index < 0 or index >= len(scores) for index in indices):
        raise ValueError("sequence is not bound to the native score matrix")
    values = [float(scores[index, slot]) for slot, index in enumerate(indices)]
    return {"native_decode_indices": list(indices), "per_observation": values,
            "mean": float(np.mean(values)), "prefix": values[0], "suffix": values[-1],
            "nondecreasing_native_order": all(b >= a for a, b in zip(indices, indices[1:])),
            "repeated_native_ids": len(set(indices)) < len(indices)}


def _nominate(features: np.ndarray, scores: np.ndarray, budget: CorrespondenceBudget, fps: float) -> list[dict[str, Any]]:
    count = budget.index_window_slots
    totals = []
    for start in range(len(features) - count + 1):
        window = scores[start:start + count]
        value = window[:, 0].copy()
        for column in range(1, window.shape[1]):
            value = np.maximum.accumulate(value) + window[:, column]
        totals.append(float(value.max() / window.shape[1]))
    nominations = []
    for start in np.argsort(totals)[::-1]:
        if any(abs(int(start) - row["index_window_start"]) < count for row in nominations):
            continue
        nominations.append({"index_window_start": int(start), "nominal_window_seconds": [float(start / fps), float((start + count) / fps)],
                            "score": totals[start], "native_pts_unknown_for_index_slots": True})
        if len(nominations) == budget.max_source_windows:
            break
    return nominations


def produce_chunk_correspondence(*, camera_chunk: Path, projector_source: Path | str, source_index: Path,
                                 projector_geometry: ProjectorGeometry, output_dir: Path,
                                 budget: CorrespondenceBudget = CorrespondenceBudget(), camera_bundle: Path | None = None,
                                 retained_source_bundles: tuple[Path, ...] = (),
                                 qualification_controls: Path | None = None) -> ChunkCorrespondence:
    camera_chunk = Path(camera_chunk).resolve()
    source_index = Path(source_index).resolve()
    output_dir = Path(output_dir).resolve()
    if budget.max_source_windows < 1 or budget.index_window_slots < 1 or budget.max_source_frames < 1:
        raise ValueError("finite positive correspondence budget required")
    camera_sha = sha256_file(camera_chunk)
    metadata = _read_json(source_index.with_suffix(".json"))
    source_sha = metadata["source"]["sha256"]
    source_media = str(projector_source)
    source_path = Path(source_media)
    remote_identity = None
    if source_path.is_file():
        source_path = source_path.resolve()
        if sha256_file(source_path) != source_sha:
            raise ValueError("cached source SHA256 mismatch")
        source_binding = "current cached source bytes SHA-verified"
    elif not budget.allow_source_decode and retained_source_bundles and source_media == metadata["source"]["url"]:
        source_binding = "retained source SHA-keyed decode provenance; current full source not rehashed"
    elif urlparse(source_media).scheme in {"http", "https"} and source_media == metadata["source"]["url"]:
        remote_identity = _http_source_identity(source_media, source_sha, source_index)
        source_binding = "current server SHA256 bound to strong HTTP validator"
    else:
        raise ValueError("verified local or SHA256-bound HTTP source required; no nominal fallback")
    features, metadata, migration = load_source_index(source_index, source_sha, metadata["source"]["url"])
    control_provenance, control_roles = _qualification_controls(
        qualification_controls, source_index, source_sha, camera_sha, projector_geometry.sha256)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = (_bundle(Path(camera_bundle), str(camera_chunk), camera_sha) if camera_bundle else
              _decode(camera_chunk, output_dir / "camera", camera_sha, budget.max_camera_frames))
    if len(camera) > budget.max_camera_frames:
        raise ValueError("camera frame budget exhausted")
    observed = tuple(frame for frame in camera if frame.rgb_path)
    if not observed:
        raise ValueError("no retained native camera RGB observations")
    width, height = observed[0].width, observed[0].height
    if any((frame.width, frame.height) != (width, height) for frame in observed):
        raise ValueError("native camera geometry changed")
    polygon = tuple((x * width / 1280, y * height / 720) for x, y in projector_geometry.polygon_1280)
    homography = _homography(polygon)
    interior = cv2.erode(_projection(polygon, width, height).astype(np.uint8),
                         np.ones((2 * projector_geometry.interior_margin_px + 1,) * 2, np.uint8)).astype(bool)
    if projector_geometry.frozen_interior_path:
        frozen = np.asarray(Image.open(projector_geometry.frozen_interior_path)) > 0
        if frozen.shape != interior.shape:
            raise ValueError("frozen projection geometry size mismatch")
        interior &= frozen
    excluded = np.zeros(interior.shape, bool)
    for x0, y0, x1, y1 in projector_geometry.excluded_regions_1280:
        excluded[round(y0 * height / 720):round(y1 * height / 720), round(x0 * width / 1280):round(x1 * width / 1280)] = True
    interior &= ~excluded
    index_polygon = tuple(tuple(point) for point in metadata["geometry"]["polygon_1280"])
    index_mask = _projection(index_polygon, 1280, 720)
    index_mask[int(720 * .69):, :] = False
    index_mask[:, int(1280 * .82):] = False
    ys, xs = np.nonzero(index_mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    small_mask = cv2.resize(index_mask[y0:y1, x0:x1].astype(np.uint8), (48, 32), interpolation=cv2.INTER_NEAREST).astype(bool)
    cells = int(small_mask.sum())
    if features.shape[1] != cells * 3:
        raise ValueError("retained spatial descriptor layout changed")
    camera_to_index = _homography(index_polygon) @ np.linalg.inv(homography)
    images = [load_native_rgb(frame) for frame in observed]
    grays = [cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY) for image in images]
    reference = (np.median(np.stack([
        cv2.GaussianBlur(np.asarray(image.convert("L"), np.float32), (0, 0), 1)
        for image in images]), axis=0) if len(images) >= 3 else None)
    supports, boxes = [], []
    for image in images:
        support, proposals = _camera_support(image, interior, excluded, reference)
        supports.append(support)
        boxes.append(proposals)
    sampled = tuple(map(int, np.unique(np.rint(np.linspace(0, len(observed) - 1, min(budget.discovery_observations, len(observed)))).astype(int))))
    discovery = tuple(slot for position, slot in enumerate(sampled) if position % 2 == 1 and supports[slot].any())
    coarse = []
    for slot in discovery:
        warped = cv2.warpPerspective(grays[slot], camera_to_index, (1280, 720))
        support = cv2.warpPerspective(supports[slot].astype(np.uint8), camera_to_index, (1280, 720), flags=cv2.INTER_NEAREST)
        small = cv2.resize(support[y0:y1, x0:x1], (48, 32), interpolation=cv2.INTER_AREA) > 0
        selected = np.flatnonzero(small[small_mask])
        columns = np.concatenate([selected + block * cells for block in range(3)])
        if not len(columns):
            continue
        query = _local_values(shape_descriptor(warped, index_mask), columns)
        coarse.append(_local_values(features, columns) @ query)
    nominations = _nominate(features, np.stack(coarse, axis=1), budget, metadata["fps"]) if coarse else []
    bundles = []
    retained_count = 0
    for path in retained_source_bundles:
        frames = _bundle(Path(path), metadata["source"]["url"], source_sha)
        retained_count += len(frames)
        bundles.append((Path(path).name, frames))
    if control_roles:
        retained_ids = {rows[0].decode_id for _, rows in bundles}
        control_value = _read_json(Path(control_provenance["path"]))
        for entry in [*control_value["controls"], *control_value.get("source_facts", ())]:
            if entry["decode_id"] in control_roles and entry["decode_id"] not in retained_ids:
                frames = _bundle(Path(entry["identity_path"]).parent, metadata["source"]["url"], source_sha)
                retained_count += len(frames)
                bundles.append((f"control-{len(bundles)}", frames))
                retained_ids.add(entry["decode_id"])
    if retained_count > budget.max_source_frames:
        raise ValueError("retained source native frame budget exhausted")
    coverage = []
    nominated_ids = set()
    for number, nomination in enumerate(nominations):
        lo, hi = nomination["nominal_window_seconds"]
        overlaps = [(max(0., min(hi, float(rows[-1].seconds)) - max(lo, float(rows[0].seconds))), label) for label, rows in bundles]
        overlap, label = max(overlaps, default=(0., ""))
        if overlap > 0:
            nominated_ids.add(label)
            coverage.append({**nomination, "native_bundle": label})
        elif budget.allow_source_decode:
            remaining = budget.max_source_frames - sum(len(rows) for _, rows in bundles)
            if remaining <= 0:
                coverage.append({**nomination, "native_bundle": None, "reason": "native frame budget exhausted"})
                continue
            rows = _decode(source_media if remote_identity else source_path, output_dir / f"source-{number}",
                           source_sha, remaining, lo, hi, source_identity=remote_identity)
            if rows:
                label = f"source-{number}"
                bundles.append((label, rows))
                nominated_ids.add(label)
                coverage.append({**nomination, "native_bundle": label})
        else:
            coverage.append({**nomination, "native_bundle": None, "reason": "no retained RGB; additional native windows forbidden"})
    component_masks = []
    components = []
    for slot, support in enumerate(supports):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8))
        masks, records = [], []
        for label in range(1, count):
            mask = labels == label
            x, y, w, h, area = map(int, stats[label])
            component_id = f"camera-{observed[slot].frame_index:04d}-component-{label}"
            path = output_dir / f"{component_id}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
            masks.append(mask)
            records.append({"component_id": component_id, "mask_path": str(path), "mask_sha256": sha256_file(path),
                            "bbox_xyxy": [x, y, x + w, y + h], "pixel_count": area,
                            "structural_status": "unresolved", "matched_families": []})
        component_masks.append(masks)
        components.append(records)
    vectors = [[_gradient_vector(grays[slot], tuple(record["bbox_xyxy"]), mask)
                for record, mask in zip(components[slot], component_masks[slot], strict=True)] for slot in range(len(observed))]
    scored = []
    for label, source_frames in bundles:
        matrix = np.zeros((len(source_frames), len(observed)), np.float32)
        component_scores = [np.zeros((len(source_frames), len(records)), np.float32) for records in components]
        for index, source in enumerate(source_frames):
            rgb = load_native_rgb(source)
            # Index-generation geometry is 1280x720; source resize is explicit
            # geometry only, never relabelled as a different native identity.
            warped = cv2.warpPerspective(np.asarray(rgb.resize((1280, 720))), homography, (width, height))
            gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
            for slot in range(len(observed)):
                values = [float(query @ _gradient_vector(gray, tuple(record["bbox_xyxy"]), mask))
                          for query, record, mask in zip(vectors[slot], components[slot], component_masks[slot], strict=True)]
                component_scores[slot][index] = values
                matrix[index, slot] = float(np.mean(values)) if values else 0.
        if discovery:
            path = ordered_native_path(matrix, discovery)
            backward = ordered_native_path(matrix, discovery, -1)
        else:
            path = backward = tuple(0 for _ in observed)
        variants = {"ordered": evaluate_native_sequence(matrix, path), "decreasing": evaluate_native_sequence(matrix, backward),
                    "exact_reversal": evaluate_native_sequence(matrix, tuple(reversed(path))),
                    "fixed": evaluate_native_sequence(matrix, tuple(int(np.argmax(matrix.mean(axis=1))) for _ in observed))}
        scored.append({"family_id": label, "frames": source_frames, "matrix": matrix, "path": path,
                       "component_scores": component_scores, "roles": control_roles.get(source_frames[0].decode_id, set()),
                       "nominated": label in nominated_ids, "variants": variants})
    local_nulls = [row for row in scored if "local_shape_null" in row["roles"]]
    rejected_sequences = [row for row in scored if "sequence_rejected" in row["roles"]]
    controls_complete = bool(local_nulls and any(row["nominated"] for row in scored) and len(discovery) >= 2)
    null_components = [np.maximum(0., np.max(np.concatenate([row["component_scores"][slot] for row in local_nulls]), axis=0))
                       if local_nulls and components[slot] else np.zeros(len(components[slot])) for slot in range(len(observed))]
    # Sequence-only negatives may repeat the true middle shape. They constrain
    # complete paths, never the local component null envelope.
    # Wrong-order and fixed-content counterfactuals are recomputed for THIS
    # capture. They require no imported July rejection labels or fitted weights.
    ordering_controls = [{"family_id": row["family_id"],
        "decreasing_complete_mean": float(np.max(ordered_state_path_totals(row["matrix"][::-1])) / len(observed)),
        "fixed_complete_mean": float(np.max(row["matrix"].astype(np.float64).mean(axis=1))),
        "exact_reversal_mean": row["variants"]["exact_reversal"]["mean"]} for row in scored if row["nominated"]]
    sequence_null = max([0.,
        *(value for control in ordering_controls for key, value in control.items() if key != "family_id"),
        *(float(np.max(ordered_state_path_totals(row["matrix"])) / len(observed)) for row in rejected_sequences)])
    evidence = []
    for row in scored:
        local = np.column_stack([np.any(values > null_components[slot], axis=1) if values.shape[1]
                                 else np.zeros(len(row["frames"]), bool)
                                 for slot, values in enumerate(row["component_scores"])])
        # Local positivity belongs to the constrained state. Missing proposals
        # elsewhere are unknown observations, not evidence rejecting this state.
        totals = ordered_state_path_totals(row["matrix"])
        row["conditional_totals"] = totals
        row["qualified"] = local & (totals / len(observed) > sequence_null) & controls_complete
        complete_mean = float(np.max(totals) / len(observed))
        rejected = ("sequence_rejected" in row["roles"] or
                    (controls_complete and row["nominated"] and complete_mean <= sequence_null))
        row["sequence_rejected"] = rejected
        score_path = output_dir / f"native-state-scores-{len(evidence):02d}.npz"
        np.savez_compressed(score_path, object_gradient_scores=row["matrix"], conditional_path_totals=totals,
                            local_support=local, qualified=row["qualified"],
                            **{f"component_scores_{slot}": values for slot, values in enumerate(row["component_scores"])},
                            **{f"component_null_{slot}": values for slot, values in enumerate(null_components)})
        row["score_evidence"] = {"path": str(score_path), "sha256": sha256_file(score_path),
                                 "native_identity_path": row["frames"][0].identity_path,
                                 "native_decode_id": row["frames"][0].decode_id,
                                 "camera_observation_identities": [asdict(identity) for identity in observed]}
        # Explicitly frozen rejected identities cannot survive merely because
        # their middle content matches. Unknown alternatives remain unresolved.
        status = "rejected" if rejected else ("supported" if row["qualified"].any() else "unresolved")
        evidence.append({"family_id": row["family_id"], "nominated": row["nominated"], "structural_status": status,
                         "score_evidence": row["score_evidence"],
                         "variants": row["variants"], "native_interval_seconds": [float(row["frames"][row["path"][0]].seconds), float(row["frames"][row["path"][-1]].seconds)],
                         "rejection_evidence": {"control_decode_id": row["frames"][0].decode_id,
                             "kind": "frozen-capture-rejection" if "sequence_rejected" in row["roles"] else "current-capture-ordered-null-envelope",
                             "control_receipt_sha256": control_provenance.get("sha256"),
                             "complete_ordered_mean": complete_mean,
                             "sequence_null_mean": sequence_null} if rejected else None,
                         "qualification_coverage": {"searched_native_states": int(row["matrix"].size),
                             "qualified_native_states": int(row["qualified"].sum())},
                         "reasons": ["camera component local-null and complete ordered-path qualification; no photometry/privacy permission",
                                     "exposure mixture and loop count remain unknown"]})
    frame_records = []
    observed_slot = {frame.frame_index: slot for slot, frame in enumerate(observed)}
    matrix_tuple = tuple(tuple(map(float, row)) for row in homography)
    for frame in camera:
        slot = observed_slot.get(frame.frame_index)
        families = []
        support_path = None
        support_sha = None
        reasons = ["exposure duration and loop count unknown; structural matching is not privacy permission"]
        if slot is not None:
            support_file = output_dir / f"support-{frame.frame_index:04d}.png"
            Image.fromarray(supports[slot].astype(np.uint8) * 255).save(support_file)
            support_path, support_sha = str(support_file), sha256_file(support_file)
            for row in scored:
                if not row["nominated"] or row["sequence_rejected"]:
                    continue
                qualified = np.flatnonzero(row["qualified"][:, slot])
                native = tuple(row["frames"][index] for index in qualified)
                states = []
                for index in qualified:
                    component_evidence = []
                    for component, record in enumerate(components[slot]):
                        score = float(row["component_scores"][slot][index, component])
                        matched = score > null_components[slot][component]
                        component_evidence.append({"component_id": record["component_id"], "score": score,
                            "null_max": float(null_components[slot][component]), "qualified": bool(matched)})
                    total = float(row["conditional_totals"][index, slot])
                    states.append({"source_identity": asdict(row["frames"][index]),
                                   "local_score": float(row["matrix"][index, slot]),
                                   "conditional_path_total": total, "conditional_path_mean": total / len(observed),
                                   "component_scores": component_evidence})
                qualification = {"method": "non-strict-ordered-state-envelope-v1", "camera_identity": asdict(frame),
                    "support_mask_sha256": support_sha, "control_receipt_sha256": control_provenance.get("sha256"),
                    "sequence_null_mean": sequence_null, "observation_count": len(observed),
                    "score_evidence": row["score_evidence"],
                    "coverage": {"searched_native_states": len(row["frames"]), "qualified_native_states": len(native)},
                    "native_states": states}
                # An unqualified nomination is still a genuine unresolved
                # alternative. Its search interval is NOT exposure membership.
                interval_frames = native or row["frames"]
                families.append(SourceCandidateFamily(row["family_id"], native,
                    (float(interval_frames[0].seconds), float(interval_frames[-1].seconds)),
                    "supported" if native else "unresolved",
                    ("only individually qualified native identities are exposure hypotheses",
                     "candidate interval is not measured exposure"), qualification))
            for component, record in enumerate(components[slot]):
                record["matched_families"] = [family.family_id for family in families
                    if family.qualification["native_states"] and all(
                        state["component_scores"][component]["qualified"] for state in family.qualification["native_states"])]
                all_families = {family.family_id for family in families}
                if all_families and set(record["matched_families"]) == all_families:
                    record["structural_status"] = "supported"
            if not boxes[slot]:
                reasons.append("insufficient camera-only component support")
        else:
            reasons.append("native camera identity retained but RGB unavailable; no interpolation or substitution")
        status = "supported" if families and all(family.structural_status == "supported" for family in families) else "unresolved"
        frame_records.append(FrameCorrespondence(frame, support_path, support_sha, tuple(families), status,
                                                  tuple(reasons), projector_geometry.sha256, matrix_tuple,
                                                  component_qualifications=tuple(components[slot]) if slot is not None else ()))
    result = ChunkCorrespondence(camera_sha, source_sha, sha256_file(source_index), projector_geometry.sha256,
                                 sha256_file(Path(__file__)), tuple(frame_records),
                                 {"source_binding": source_binding, "index_dependency_compatibility": migration,
                                  "camera_chunk": str(camera_chunk), "projector_source": str(source_path) if source_path.is_file() else source_media,
                                  "source_index": str(source_index.resolve()), "index_metadata_sha256": sha256_file(source_index.with_suffix(".json")),
                                  "http_source_identity": remote_identity,
                                  "qualification_controls": control_provenance,
                                  "camera_native_decode_id": camera[0].decode_id, "source_native_decode_ids": [rows[0].decode_id for _, rows in bundles],
                                  "geometry": asdict(projector_geometry), "proposal_implementation_sha256": sha256_file(ROOT / "scripts/cat_projector_frame_detector.py")},
                                 {"index_shape": list(features.shape), "camera_native_frames": len(camera), "camera_rgb_observations": len(observed),
                                  "retained_source_native_frames": retained_count, "source_native_frames_scored": sum(len(rows) for _, rows in bundles),
                                  "discovery_observation_slots": list(discovery), "nominations": coverage, "budget": asdict(budget),
                                  "qualification_controls_complete": controls_complete,
                                  "current_capture_ordering_controls": ordering_controls,
                                  "sequence_null_mean": sequence_null,
                                  "rejected_family_ids": [row["family_id"] for row in scored if row["sequence_rejected"]],
                                  "qualified_native_states": sum(int(row["qualified"].sum()) for row in scored if not row["sequence_rejected"]),
                                  "unsearched_source_content_remains": True, "native_identity_from_nominal_index": False},
                                 tuple(evidence), str(output_dir / "correspondence.json"))
    _save_json(Path(result.output_path), asdict(result))
    return result


def load_chunk_correspondence(path: Path) -> ChunkCorrespondence:
    """Restore records; every admission lookup still rechecks their bindings."""
    raw = json.loads(Path(path).read_text())
    frames = []
    for value in raw.pop("frames"):
        families = tuple(SourceCandidateFamily(
            family["family_id"], tuple(NativeFrameIdentity(**item) for item in family["source_frames"]),
            tuple(family["candidate_interval"]), family["structural_status"], tuple(family["reasons"]), family.get("qualification"))
            for family in value.pop("candidate_families"))
        value["camera_identity"] = NativeFrameIdentity(**value["camera_identity"])
        value["candidate_families"] = families
        value["reasons"] = tuple(value["reasons"])
        value["source_to_camera"] = tuple(tuple(row) for row in value["source_to_camera"])
        value["component_qualifications"] = tuple(value.get("component_qualifications", ()))
        frames.append(FrameCorrespondence(**value))
    raw["frames"] = tuple(frames)
    raw["sequence_evidence"] = tuple(raw["sequence_evidence"])
    return ChunkCorrespondence(**raw)


def correspondence_for_camera_frame(chunk: ChunkCorrespondence, camera_identity: NativeFrameIdentity) -> FrameCorrespondence:
    if camera_identity.media_sha256 != chunk.camera_sha256:
        raise ValueError("camera media identity mismatch")
    if chunk.producer_sha256 != sha256_file(Path(__file__)):
        raise ValueError("stale correspondence producer implementation")
    if chunk.provenance["proposal_implementation_sha256"] != sha256_file(ROOT / "scripts/cat_projector_frame_detector.py"):
        raise ValueError("stale camera-support implementation")
    if sha256_file(Path(chunk.provenance["camera_chunk"])) != chunk.camera_sha256:
        raise ValueError("camera chunk bytes changed")
    source_index = Path(chunk.provenance["source_index"])
    if sha256_file(source_index) != chunk.index_sha256 or sha256_file(source_index.with_suffix(".json")) != chunk.provenance["index_metadata_sha256"]:
        raise ValueError("stale source index/provenance")
    metadata = json.loads(source_index.with_suffix(".json").read_text())
    load_source_index(source_index, chunk.source_sha256, metadata["source"]["url"])
    geometry = ProjectorGeometry(**chunk.provenance["geometry"])
    if geometry.sha256 != chunk.geometry_sha256:
        raise ValueError("stale projection geometry")
    controls = chunk.provenance.get("qualification_controls", {})
    if controls.get("sha256"):
        current, _roles = _qualification_controls(Path(controls["path"]), source_index, chunk.source_sha256,
                                                  chunk.camera_sha256, chunk.geometry_sha256)
        if current != controls:
            raise ValueError("qualification control binding changed")
    if chunk.provenance["source_binding"] == "current cached source bytes SHA-verified":
        if sha256_file(Path(chunk.provenance["projector_source"])) != chunk.source_sha256:
            raise ValueError("cached source bytes changed")
    if chunk.provenance.get("http_source_identity") is not None:
        current_identity = _http_source_identity(chunk.provenance["projector_source"], chunk.source_sha256, source_index)
        if current_identity != chunk.provenance["http_source_identity"]:
            raise ValueError("HTTP source identity binding changed")
    for frame in chunk.frames:
        if frame.camera_identity == camera_identity:
            if frame.geometry_sha256 != chunk.geometry_sha256:
                raise ValueError("stale frame geometry")
            for family in frame.candidate_families:
                if any(source.media_sha256 != chunk.source_sha256 for source in family.source_frames):
                    raise ValueError("source frame media substitution")
                qualification = family.qualification
                if family.structural_status == "supported":
                    if (not qualification or qualification["camera_identity"] != asdict(camera_identity)
                            or qualification["support_mask_sha256"] != frame.support_mask_sha256
                            or qualification["control_receipt_sha256"] != controls.get("sha256")
                            or [state["source_identity"] for state in qualification["native_states"]]
                               != [asdict(source) for source in family.source_frames]):
                        raise ValueError("native component qualification binding changed")
            for component in frame.component_qualifications:
                if sha256_file(Path(component["mask_path"])) != component["mask_sha256"]:
                    raise ValueError("substituted camera component mask")
            if frame.support_mask_path and sha256_file(Path(frame.support_mask_path)) != frame.support_mask_sha256:
                raise ValueError("substituted camera support mask")
            if camera_identity.rgb_path:
                load_native_rgb(camera_identity)
            return frame
    raise ValueError("exact native camera identity unavailable; no nearest/modulo/nominal lookup")


def record_review_frame_bindings(*, camera_chunk: Path, framehash_path: Path,
                                 output_pattern: Path, command: list[str]) -> None:
    """Certify derivatives emitted alongside native RGB framehash in ONE decode."""
    filters = command[command.index("-filter_complex") + 1]
    if filters != "[0:v]format=rgb24,split=2[pictures][identity]" or "-copyts" not in command:
        raise ValueError("review frames require same-decode native RGB conversion")
    text = framehash_path.read_text()
    tb, hashes = _framehash_rows(text)
    if not hashes:
        raise ValueError("review decode has no native frame identities")
    media_sha = sha256_file(camera_chunk)
    framehash_sha = sha256_file(framehash_path)
    images = [Path(str(output_pattern).replace("%05d", f"{index:05d}")) for index in range(len(hashes))]
    if not all(image.is_file() for image in images):
        raise ValueError("review derivative/native frame counts disagree")
    for index, (image, (pts, _size, rgb_sha)) in enumerate(zip(images, hashes, strict=True)):
        _save_json(image.with_suffix(image.suffix + ".native.json"), {
            "image_sha256": sha256_file(image), "camera_sha256": media_sha,
            "frame_index": index, "pts": pts, "time_base_num": tb.numerator,
            "time_base_den": tb.denominator, "rgb_sha256": rgb_sha,
            "framehash_path": str(framehash_path.resolve()), "framehash_sha256": framehash_sha,
            "command": command, "conversion": NATIVE_RGB_CONVERSION,
        })


def _main() -> int:
    from dataclasses import replace

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-chunk", type=Path, required=True)
    parser.add_argument("--projector-source", required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--frozen-interior", type=Path)
    parser.add_argument("--camera-bundle", type=Path)
    parser.add_argument("--retained-source-bundle", type=Path, action="append", default=[])
    parser.add_argument("--qualification-controls", type=Path)
    parser.add_argument("--no-source-decode", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-retained-controls", action="store_true")
    args = parser.parse_args()
    # Share the concrete types with standalone review/detector module loaders.
    sys.modules["cat_tv_play_source_correspondence"] = sys.modules[__name__]
    geometry_value = _read_json(args.geometry)
    geometry_value = geometry_value.get("geometry", geometry_value)
    geometry = ProjectorGeometry(
        tuple(tuple(point) for point in geometry_value.get("fitted_polygon") or geometry_value["polygon_1280"]),
        frozen_interior_path=str(args.frozen_interior) if args.frozen_interior else None,
    )
    chunk = produce_chunk_correspondence(
        camera_chunk=args.camera_chunk, projector_source=args.projector_source,
        source_index=args.source_index, projector_geometry=geometry, output_dir=args.output_dir,
        budget=CorrespondenceBudget(allow_source_decode=not args.no_source_decode),
        camera_bundle=args.camera_bundle, retained_source_bundles=tuple(args.retained_source_bundle),
        qualification_controls=args.qualification_controls,
    )
    review = _load_repository_script("cat_projector_label_review_server")

    observed = [frame for frame in chunk.frames if frame.camera_identity.rgb_path]
    # One actual review consumer call, not a hand-authored correspondence row.
    chosen = observed[len(observed) // 2]
    consumed = review.correspondence_for_review_image(
        Path(chosen.camera_identity.rgb_path), chunk=chunk, camera_identity=chosen.camera_identity,
    )
    guards = {}
    invalid = {
        "missing_pts": lambda: replace(chosen.camera_identity, pts=None),
        "rgb_substitution": lambda: load_native_rgb(replace(chosen.camera_identity, rgb_path=observed[0].camera_identity.rgb_path)),
        "stale_source_key": lambda: load_source_index(args.source_index, "0" * 64, _read_json(args.source_index.with_suffix(".json"))["source"]["url"]),
        "stale_geometry_key": lambda: correspondence_for_camera_frame(replace(chunk, geometry_sha256="0" * 64), chosen.camera_identity),
        "stale_index_key": lambda: correspondence_for_camera_frame(replace(chunk, index_sha256="0" * 64), chosen.camera_identity),
        "stale_producer_key": lambda: correspondence_for_camera_frame(replace(chunk, producer_sha256="0" * 64), chosen.camera_identity),
    }
    for name, operation in invalid.items():
        try:
            operation()
        except ValueError as error:
            guards[name] = {"rejected": True, "reason": str(error)}
        else:
            guards[name] = {"rejected": False}
    supported = [row for row in chunk.sequence_evidence if row["structural_status"] == "supported"]
    positive = next((row for row in supported if 1559 <= row["native_interval_seconds"][0] < 1566), None)
    blank = next((row for row in chunk.sequence_evidence if row["family_id"] == "known-blank"), None)
    competing = next((row for row in chunk.sequence_evidence if row["family_id"] == "known-rejected-ordered"), None)
    controls = {}
    if positive:
        oracle = positive["variants"]["ordered"]
        controls = {
            "blank": blank is not None and oracle["mean"] > max(row["mean"] for row in blank["variants"].values()),
            "decreasing": oracle["mean"] > max(row["variants"]["decreasing"]["mean"] for row in chunk.sequence_evidence),
            "exact_reversal": oracle["mean"] > positive["variants"]["exact_reversal"]["mean"],
            "288_prefix": competing is not None and oracle["prefix"] > competing["variants"]["ordered"]["prefix"],
            "288_suffix": competing is not None and oracle["suffix"] > competing["variants"]["ordered"]["suffix"],
        }
    passed = bool(positive and consumed is not None and consumed.structural_status == "supported"
                  and controls and all(controls.values()) and all(row["rejected"] for row in guards.values()))
    summary = {
        "status": "supported_retained_producer_review_oracle" if passed else "positive_or_control_blocker",
        "camera_sha256": chunk.camera_sha256, "source_sha256": chunk.source_sha256,
        "index_sha256": chunk.index_sha256, "producer_sha256": chunk.producer_sha256,
        "geometry_sha256": chunk.geometry_sha256,
        "changed_path_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
            Path(__file__), ROOT / "scripts/cat_projector_label_review_server.py",
            ROOT / "scripts/cat_projector_active_learning.py", ROOT / "tests/test_source_correspondence.py",
            ROOT / "tests/test_cat_projector_label_review_server.py", ROOT / "docs/label-review.md")},
        "index_dependency_compatibility": chunk.provenance["index_dependency_compatibility"],
        "coverage": chunk.search_coverage,
        "review_consumer_frame": {"native_pts": chosen.camera_identity.pts,
            "time_base": f"{chosen.camera_identity.time_base_num}/{chosen.camera_identity.time_base_den}",
            "rgb_sha256": chosen.camera_identity.rgb_sha256,
            "structural_status": consumed.structural_status if consumed else None},
        "camera_support_counts": {str(frame.camera_identity.frame_index):
            int(np.count_nonzero(np.asarray(Image.open(frame.support_mask_path)))) if frame.support_mask_path else None
            for frame in observed},
        "sequence_evidence": chunk.sequence_evidence, "real_negative_controls": controls,
        "identity_rejection_controls": guards,
        "ambiguity": ["candidate intervals are not measured exposure", "loop count unknown",
                      "retained competing families and unsearched intervals remain", "no photometry/privacy admission"],
        "correspondence_path": chunk.output_path,
    }
    _save_json(args.output_dir / "smoke.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if not args.check_retained_controls or passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
