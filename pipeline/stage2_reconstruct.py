"""
Stage 2: Human-Prior 기반 3D 복원 (COLMAP 미사용).

- 입력: Stage 1 출력 data/processed_images/
- 처리: 인체 prior(ROMP/SMPL 또는 synthetic 카메라)로 canonical mesh + per-image 카메라 생성
- 출력: data/human_prior/ (canonical_mesh.ply, cameras.json, image_list.txt)
- 목표: deformation/stylization 가능한 기하 구조, 3DGS 초기화용 뷰 제공

Geometry 기준: Canonical body space (T-pose, 원점=몸 중심). 자세한 설계는 docs/STAGE2_HUMAN_PRIOR.md 참고.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# -----------------------------------------------------------------------------
# 경로
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_IMAGES_DIR = PROJECT_ROOT / "data" / "processed_images"
HUMAN_PRIOR_DIR = PROJECT_ROOT / "data" / "human_prior"

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Human-prior 옵션
USE_ROMP_IF_AVAILABLE = True
MIN_IMAGES = 3
# Synthetic fallback: 원형 궤도 반지름(미터 단위 개념), 카메라가 바라보는 원점 = 인체 중심
SYNTHETIC_ORBIT_RADIUS = 3.0
SYNTHETIC_FOCAL_SCALE = 1.2  # focal = max(W,H) * this


def _list_image_paths(dir_path: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(dir_path.glob(f"*{ext}"))
    return sorted(paths, key=lambda p: p.name.lower())


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return (im.width, im.height)


def _write_ply(vertices: np.ndarray, faces: np.ndarray, out_path: Path) -> None:
    """Write mesh to PLY (binary)."""
    nv, nf = len(vertices), len(faces)
    with open(out_path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {nv}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {nf}\n".encode())
        f.write(b"property list uchar int vertex_indices\nend_header\n")
        f.write(vertices.astype(np.float32).tobytes())
        # PLY face: count (3) + indices
        for row in faces:
            f.write(np.uint8(3).tobytes())
            f.write(row.astype(np.int32).tobytes())
    return


def _synthetic_cameras_and_mesh(
    image_paths: list[Path],
    orbit_radius: float,
    focal_scale: float,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Synthetic: 원형 궤도 카메라 + 단순 placeholder 메시(상자).
    세계 좌표: 원점 = 인체 중심. 카메라가 원 주위에서 원점을 바라봄.
    """
    n = len(image_paths)
    cameras = []
    for i, path in enumerate(image_paths):
        w, h = _image_size(path)
        angle = 2 * np.pi * i / max(n, 1)
        # 카메라 위치 (Y-up 가정, Z가 앞쪽)
        cx = orbit_radius * np.cos(angle)
        cz = orbit_radius * np.sin(angle)
        cam_pos = np.array([cx, 0.0, cz])
        # 원점을 바라보는 뷰
        forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        R = np.eye(3)
        R[0], R[1], R[2] = right, -up, -forward  # world-to-camera
        t = -R @ cam_pos
        f = max(w, h) * focal_scale
        K = [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]
        cameras.append({
            "image": path.name,
            "K": K,
            "R": R.tolist(),
            "t": t.tolist(),
            "width": w,
            "height": h,
        })
    # Placeholder mesh: 원점 중심 작은 상자 (canonical body 대체)
    s = 0.5
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [1, 5, 6], [1, 6, 2],
        [5, 4, 7], [5, 7, 6], [4, 0, 3], [4, 3, 7],
        [3, 2, 6], [3, 6, 7], [4, 5, 1], [4, 1, 0],
    ], dtype=np.int32)
    return cameras, vertices, faces


def _run_romp(
    image_paths: list[Path],
    output_dir: Path,
) -> tuple[list[dict[str, Any]] | None, np.ndarray | None, np.ndarray | None]:
    """
    ROMP으로 per-image SMPL + 카메라 추정 후,
    canonical mesh(첫 프레임 또는 평균 shape) 및 cameras 반환.
    실패 시 None 반환.
    """
    try:
        import cv2
        import romp
    except ImportError:
        return None, None, None

    try:
        settings = romp.main.default_settings
        model = romp.ROMP(settings)
    except Exception:
        return None, None, None

    all_cameras = []
    all_verts = []
    faces_ref = None

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        try:
            outputs = model(img)
            if outputs is None:
                continue
            # ROMP 출력: 버전에 따라 dict 또는 'results' 리스트
            res = outputs.get("results", outputs) if isinstance(outputs, dict) else outputs
            if isinstance(res, (list, tuple)) and len(res) > 0:
                res = res[0]
            if not isinstance(res, dict):
                continue
            if "verts" in res:
                v = np.asarray(res["verts"])
                if v.ndim == 3:
                    v = v[0]
                all_verts.append(v)
            cam = _romp_to_camera(res, path)
            if cam is not None:
                all_cameras.append(cam)
            if "faces" in res and faces_ref is None:
                faces_ref = np.asarray(res["faces"], dtype=np.int32)
        except Exception:
            continue

    if not all_cameras or len(all_cameras) < MIN_IMAGES:
        return None, None, None

    # Canonical mesh: 첫 프레임 메시 사용 (ROMP은 posed mesh 반환; T-pose는 smplx 등 별도 필요)
    vertices = None
    faces = faces_ref
    if all_verts and faces_ref is not None:
        vertices = all_verts[0]
        faces = faces_ref
    return all_cameras, vertices, faces


def _romp_to_camera(romp_result: dict, image_path: Path) -> dict | None:
    """ROMP 결과에서 K, R, t (world-to-camera) 구성. world = body at origin."""
    w, h = _image_size(image_path)
    # Weak-perspective 가정: scale, trans_xy
    scale = romp_result.get("cam_scale", romp_result.get("scale", 1.0))
    if isinstance(scale, (list, np.ndarray)):
        scale = float(scale[0]) if len(scale) else 1.0
    trans = romp_result.get("cam_trans", romp_result.get("trans", [0, 0]))
    trans = np.asarray(trans).flatten()
    tx_2d = trans[0] if len(trans) > 0 else 0.0
    ty_2d = trans[1] if len(trans) > 1 else 0.0
    # depth so that body center projects to (tx_2d, ty_2d)
    depth = max(w, h) * scale
    cx, cy = w / 2, h / 2
    fx = fy = max(w, h) * 1.2
    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    # 카메라가 원점(몸 중심)을 (tx_2d, ty_2d)에 투영하도록 R, t 설정
    # 원점이 이미지 (tx_2d, ty_2d)에 오려면: [0,0,depth] -> (cx, cy) + scale*(0,0) -> (tx_2d, ty_2d) => trans = (tx_2d - cx, ty_2d - cy)
    # weak persp: x_2d = s * (x_cam/z_cam) + tx, y_2d = s * (y_cam/z_cam) + ty. 원점이 몸이면 body_cam = (0,0,depth) 정도.
    # 단순화: R=I, t=(0,0,-depth) so 원점이 카메라 앞쪽 depth에 있음. 그럼 투영은 (cx, cy). 우리는 (tx_2d, ty_2d)로 옮기고 싶음.
    t = np.array([-(tx_2d - cx) * depth / fx, -(ty_2d - cy) * depth / fy, -depth], dtype=np.float64)
    R = np.eye(3).tolist()
    return {
        "image": image_path.name,
        "K": K,
        "R": R,
        "t": t.tolist(),
        "width": w,
        "height": h,
    }


def run(
    image_dir: Path | None = None,
    output_dir: Path | None = None,
    use_romp: bool | None = None,
) -> Path:
    """
    Stage 2: Human-prior 기반 canonical mesh + 카메라 생성.
    COLMAP 미사용.

    - image_dir: 입력 이미지 디렉터리 (기본: data/processed_images)
    - output_dir: 출력 디렉터리 (기본: data/human_prior)
    - use_romp: True면 ROMP 시도, False면 synthetic만. None이면 USE_ROMP_IF_AVAILABLE 사용.
    - 반환: output_dir (data/human_prior).
    """
    image_dir = image_dir or PROCESSED_IMAGES_DIR
    output_dir = output_dir or HUMAN_PRIOR_DIR
    image_dir = image_dir.resolve()
    output_dir = output_dir.resolve()

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"[Stage2] Input image directory not found: {image_dir}\n"
            "Run Stage 1 first to generate data/processed_images/."
        )

    image_paths = _list_image_paths(image_dir)
    if len(image_paths) < MIN_IMAGES:
        raise ValueError(
            f"[Stage2] Too few images: {len(image_paths)} (need at least {MIN_IMAGES}).\n"
            f"Image dir: {image_dir}"
        )
    print(f"[Stage2] Found {len(image_paths)} images (human-prior, no COLMAP).", flush=True)

    use_romp = use_romp if use_romp is not None else USE_ROMP_IF_AVAILABLE
    cameras = None
    vertices = None
    faces = None

    if use_romp:
        print("[Stage2] Trying ROMP for body + camera estimation...", flush=True)
        cameras, vertices, faces = _run_romp(image_paths, output_dir)

    if cameras is None:
        print("[Stage2] Using synthetic cameras (orbit + placeholder mesh).", flush=True)
        cameras, vertices, faces = _synthetic_cameras_and_mesh(
            image_paths,
            orbit_radius=SYNTHETIC_ORBIT_RADIUS,
            focal_scale=SYNTHETIC_FOCAL_SCALE,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # cameras.json
    cameras_path = output_dir / "cameras.json"
    with open(cameras_path, "w", encoding="utf-8") as f:
        json.dump({"cameras": cameras}, f, indent=2)
    print(f"[Stage2] Wrote {cameras_path} ({len(cameras)} cameras).", flush=True)

    # image_list.txt
    list_path = output_dir / "image_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for c in cameras:
            f.write(c["image"] + "\n")
    print(f"[Stage2] Wrote {list_path}.", flush=True)

    # canonical_mesh.ply
    if vertices is not None and faces is not None:
        mesh_path = output_dir / "canonical_mesh.ply"
        _write_ply(vertices, faces, mesh_path)
        print(f"[Stage2] Wrote {mesh_path} (vertices={len(vertices)}, faces={len(faces)}).", flush=True)
    else:
        # fallback: minimal box
        _, v, f = _synthetic_cameras_and_mesh(
            image_paths[:1], SYNTHETIC_ORBIT_RADIUS, SYNTHETIC_FOCAL_SCALE
        )
        _write_ply(v, f, output_dir / "canonical_mesh.ply")
        print("[Stage2] Wrote canonical_mesh.ply (placeholder).", flush=True)

    print("[Stage2] Success. Output: data/human_prior/", flush=True)
    return output_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2: Human-prior geometry — canonical mesh + cameras (no COLMAP)"
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=PROCESSED_IMAGES_DIR,
        help="Input image directory (default: data/processed_images)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HUMAN_PRIOR_DIR,
        help="Output directory (default: data/human_prior)",
    )
    parser.add_argument(
        "--no-romp",
        action="store_true",
        help="Disable ROMP; use synthetic cameras only",
    )
    args = parser.parse_args()

    try:
        run(
            image_dir=args.images,
            output_dir=args.out,
            use_romp=not args.no_romp,
        )
        print("Stage 2 done. Output: data/human_prior/ (cameras.json, canonical_mesh.ply, image_list.txt)", flush=True)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
