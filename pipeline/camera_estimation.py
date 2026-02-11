"""
Per-image camera extrinsic estimation for Stage 2.

Goal: produce world-to-camera (R, t) that matches train_3dgs exactly.
Wrong extrinsics → 3DGS converges to blur again.

------------------------------------------------------------
[Convention - must match train_3dgs]
------------------------------------------------------------
- world = canonical body at origin (SMPL T-pose space).
- camera extrinsic = world-to-camera: p_cam = R @ p_world + t.
- train_3dgs _camera_to_view_proj: builds view = [R|t], then flips view[2,:]
  so rasterizer sees +Z forward. So we output R, t in this convention.
- OpenCV-style camera: X right, Y down, Z into scene (before Z-flip in 3DGS,
  "in front" has negative z_cam; 3DGS flips so rasterizer gets +Z forward).
- For body at origin to be in front: we need t[2] < 0 (tz negative).
  Depth = |t[2]|; tz ∝ -depth.

------------------------------------------------------------
[ROMP output caveats]
------------------------------------------------------------
- global_orient: usually axis-angle (3,) → convert via Rodrigues.
- translation: may be weak-perspective (no real depth) or scale-dependent.
- We do NOT use ROMP tz blindly: we correct tz from bbox (tz ∝ 1/bbox_height)
  so depth is stable. ROMP R is used; txy from ROMP, tz from bbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Depth: body at origin, camera at distance. tz = -depth (negative).
DEFAULT_DEPTH = 2.5
DEPTH_MIN = 1.0
DEPTH_MAX = 6.0
# Focal: 1.2~1.5 * max(w,h) per requirement. K compatible with 3DGS.
DEFAULT_FOCAL_SCALE = 1.2
FOCAL_SCALE_MIN = 1.2
FOCAL_SCALE_MAX = 1.5

# Failure detection thresholds
VAR_CENTER_WARN = 0.01 ** 2   # variance of camera centers (per axis) below this → warn
R_IDENTITY_TOL = 0.15         # max Frobenius |R - I| to consider "all identity"
TZ_SAME_TOL = 0.05            # max std of tz to consider "all tz same"
FOREGROUND_RATIO_MIN = 0.02   # bbox heuristic: foreground pixels ratio below → warn


def _rotation_axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) to 3x3 rotation matrix. Rodrigues formula."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis_angle / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def _bbox_depth_and_center(
    image_path: Path,
) -> tuple[float, float, float, float]:
    """
    Return (cx_norm, cy_norm, depth, foreground_ratio).
    depth from tz ∝ 1 / bbox_height (normalized by image height) for stability.
    """
    try:
        import cv2
    except ImportError:
        return 0.5, 0.5, DEFAULT_DEPTH, 0.0
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.5, 0.5, DEFAULT_DEPTH, 0.0
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ys, xs = np.where(thresh > 0)
        n_foreground = len(xs)
        foreground_ratio = n_foreground / max(w * h, 1)
        if n_foreground < 10:
            return 0.5, 0.5, DEFAULT_DEPTH, foreground_ratio
        cx = float(np.mean(xs)) / max(w, 1)
        cy = float(np.mean(ys)) / max(h, 1)
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox_h = max(y_max - y_min, 1)
        # tz ∝ 1/bbox_height (normalized): larger bbox → closer → smaller depth
        # depth = k * (h / bbox_h); clamp to [DEPTH_MIN, DEPTH_MAX]
        depth = DEFAULT_DEPTH * (h / max(bbox_h, 1)) * 0.5
        depth = float(np.clip(depth, DEPTH_MIN, DEPTH_MAX))
        return cx, cy, depth, foreground_ratio
    except Exception:
        pass
    return 0.5, 0.5, DEFAULT_DEPTH, 0.0


def _estimate_with_romp(image_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Use ROMP/simple_romp if available. Returns (R 3x3, t 3) or None.
    ROMP translation may be weak-perspective; caller should correct tz from bbox.
    """
    for mod_name in ("romp", "simple_romp"):
        try:
            romp = __import__(mod_name)
            break
        except ImportError:
            continue
    else:
        return None
    try:
        if hasattr(romp, "predict"):
            result = romp.predict(str(image_path), output_dir=None)
        elif hasattr(romp, "ROMP"):
            model = getattr(romp, "ROMP", lambda **kw: None)(mode="image")
            if model is None:
                return None
            result = model(str(image_path))
        else:
            return None
        if result is None:
            return None
        if isinstance(result, (list, tuple)) and len(result) > 0:
            result = result[0]
        if isinstance(result, dict):
            params = result.get("smpl_params") or result
        else:
            params = result
        if isinstance(params, dict):
            go = np.array(params.get("global_orient", np.zeros(3)), dtype=np.float32).reshape(3)
            trans = np.array(params.get("translation", np.zeros(3)), dtype=np.float32).reshape(3)
        else:
            go = getattr(params, "global_orient", np.zeros(3))
            trans = getattr(params, "translation", np.zeros(3))
            if hasattr(go, "detach"):
                go = go.detach().cpu().numpy().flatten()
            if hasattr(trans, "detach"):
                trans = trans.detach().cpu().numpy().flatten()
            go = np.asarray(go, dtype=np.float32).reshape(3)
            trans = np.asarray(trans, dtype=np.float32).reshape(3)
        R = _rotation_axis_angle_to_matrix(go)
        return R, trans
    except Exception:
        return None


def estimate_camera_from_image(
    image_path: Path,
    *,
    use_romp: bool = True,
    fallback_depth: float = DEFAULT_DEPTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate world-to-camera (R, t) for one image.
    Convention: p_cam = R @ p_world + t; body at origin → t[2] < 0 (tz negative).
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"[camera_estimation] Image not found: {image_path}")

    cx_norm, cy_norm, bbox_depth, foreground_ratio = _bbox_depth_and_center(image_path)
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            w, h = im.size
    except Exception:
        w, h = 1024, 1024
    f = max(w, h) * DEFAULT_FOCAL_SCALE

    R_romp, t_romp = None, None
    if use_romp:
        out = _estimate_with_romp(image_path)
        if out is not None:
            R_romp, t_romp = out

    if R_romp is not None and t_romp is not None:
        R = R_romp
        # ROMP translation may be weak-perspective; force tz from bbox for stability.
        tz = -float(bbox_depth)
        tx = (cx_norm * w - w / 2) * tz / f
        ty = (cy_norm * h - h / 2) * tz / f
        t = np.array([tx, ty, tz], dtype=np.float32)
    else:
        R = np.eye(3, dtype=np.float32)
        tz = -float(bbox_depth)
        tx = (cx_norm * w - w / 2) * tz / f
        ty = (cy_norm * h - h / 2) * tz / f
        t = np.array([tx, ty, tz], dtype=np.float32)
    return R, t


def get_bbox_foreground_ratio(image_path: Path) -> float:
    """Return foreground pixel ratio for failure detection."""
    _, _, _, ratio = _bbox_depth_and_center(image_path)
    return ratio


def build_camera_entry(
    image_path: Path,
    R: np.ndarray,
    t: np.ndarray,
    focal_scale: float = DEFAULT_FOCAL_SCALE,
) -> dict[str, Any]:
    """Build one camera dict compatible with cameras.json. K = focal_scale * max(w,h)."""
    focal_scale = float(np.clip(focal_scale, FOCAL_SCALE_MIN, FOCAL_SCALE_MAX))
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            w, h = im.size
    except Exception:
        w, h = 1024, 1024
    f = max(w, h) * focal_scale
    K = [[float(f), 0, w / 2], [0, float(f), h / 2], [0, 0, 1]]
    return {
        "image": image_path.name,
        "K": K,
        "R": R.tolist(),
        "t": t.tolist(),
        "width": w,
        "height": h,
    }


def camera_center_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-space camera position: c = -R.T @ t."""
    return (-R.T @ np.asarray(t).reshape(3, 1)).flatten()


def run_sanity_check(
    cameras: list[dict[str, Any]],
    *,
    tz_values: list[float] | None = None,
    romp_used_per_image: list[bool] | None = None,
    foreground_ratios: list[float] | None = None,
) -> None:
    """
    Log camera center min/max/mean/variance. Emit warnings for:
    - center variance too small (all cameras in one spot)
    - all R near identity (ROMP likely failed)
    - all tz nearly same (depth collapse)
    - low bbox foreground ratio
    """
    if not cameras:
        return
    centers = []
    R_list = []
    tz_list = []
    for c in cameras:
        R = np.array(c["R"], dtype=np.float64)
        t = np.array(c["t"], dtype=np.float64)
        centers.append(camera_center_from_Rt(R, t))
        R_list.append(R)
        tz_list.append(float(t[2]))
    centers = np.array(centers)
    R_list = np.array(R_list)
    if tz_values is not None:
        tz_list = np.asarray(tz_values, dtype=np.float64)
    else:
        tz_list = np.array(tz_list)

    # Log min/max/mean/variance of camera centers
    c_min = centers.min(axis=0)
    c_max = centers.max(axis=0)
    c_mean = centers.mean(axis=0)
    c_var = centers.var(axis=0)
    c_var_total = float(np.sum(c_var))
    print(
        "[camera_estimation] Camera centers (world): "
        f"min=({c_min[0]:.3f},{c_min[1]:.3f},{c_min[2]:.3f}) "
        f"max=({c_max[0]:.3f},{c_max[1]:.3f},{c_max[2]:.3f}) "
        f"mean=({c_mean[0]:.3f},{c_mean[1]:.3f},{c_mean[2]:.3f}) "
        f"variance=({c_var[0]:.4f},{c_var[1]:.4f},{c_var[2]:.4f}) total={c_var_total:.4f}",
        flush=True,
    )

    # Failure / warning conditions
    if c_var_total < VAR_CENTER_WARN:
        print(
            "[camera_estimation] WARNING: Camera center variance very small. "
            "Cameras may be collapsed to one point; 3DGS may blur. Consider --camera-mode synthetic.",
            flush=True,
        )
    if np.all(np.abs(centers[:, 2] - centers[:, 2].mean()) < 1e-6):
        print(
            "[camera_estimation] WARNING: All camera centers have nearly the same Z. "
            "Distribution may be degenerate.",
            flush=True,
        )

    # All R near identity?
    I = np.eye(3, dtype=np.float64)
    frob_diffs = [np.linalg.norm(R - I, "fro") for R in R_list]
    if all(f < R_IDENTITY_TOL for f in frob_diffs):
        print(
            "[camera_estimation] WARNING: All rotations near identity. "
            "ROMP may have failed or returned no pose; using fallback for all.",
            flush=True,
        )

    # All tz nearly same?
    tz_std = float(np.std(tz_list))
    if tz_std < TZ_SAME_TOL:
        print(
            "[camera_estimation] WARNING: All tz values nearly identical "
            f"(std={tz_std:.4f}). Depth collapse; 3DGS may not recover shape.",
            flush=True,
        )

    # Low foreground ratio (bbox heuristic)
    if foreground_ratios is not None:
        low_fg = [r for r in foreground_ratios if r < FOREGROUND_RATIO_MIN]
        if low_fg:
            print(
                f"[camera_estimation] WARNING: {len(low_fg)} image(s) have very low foreground ratio "
                f"(< {FOREGROUND_RATIO_MIN}). Bbox-based depth may be unreliable.",
                flush=True,
            )


def run_pre_training_checks(
    cameras: list[dict[str, Any]],
    mesh_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bool:
    """
    Pre-training sanity: (1) centers surround the subject, (2) cameras look at scene,
    (3) no camera inside object. Returns True if checks pass (no hard failure).
    """
    if not cameras:
        return True
    centers = np.array([camera_center_from_Rt(np.array(c["R"]), np.array(c["t"])) for c in cameras])
    origin = np.array(mesh_origin, dtype=np.float64)
    dists = np.linalg.norm(centers - origin, axis=1)
    # (1) Spread around: variance of centers should be positive
    if centers.var(axis=0).sum() < VAR_CENTER_WARN:
        print("[camera_estimation] Pre-train check: camera distribution very tight.", flush=True)
    # (2) All look toward origin: camera forward = -R.T third column; dot with (origin - c) should be > 0
    for i, c in enumerate(cameras):
        R = np.array(c["R"], dtype=np.float64)
        t = np.array(c["t"], dtype=np.float64)
        cam_pos = camera_center_from_Rt(R, t)
        forward = -R.T[:, 2]
        to_origin = origin - cam_pos
        if np.dot(forward, to_origin) < -0.1:
            print(f"[camera_estimation] Pre-train check: camera {i} may not face origin.", flush=True)
    # (3) No camera inside: distance from origin should be > small epsilon
    if np.any(dists < 0.1):
        print("[camera_estimation] Pre-train check: at least one camera very close to origin (inside object?).", flush=True)
    return True
