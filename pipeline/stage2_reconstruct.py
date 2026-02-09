"""
Stage 2: Human shape prior 생성 (COLMAP / ROMP 미사용).

- 입력: Stage 1 출력 data/processed_images/ (이미지 목록·해상도만 사용)
- 처리: SMPL_NEUTRAL.pkl 직접 로드 → canonical T-pose mesh. 카메라는 synthetic orbit.
- 출력: data/human_prior/ (canonical_mesh.ply ~6890 verts, cameras.json, image_list.txt)

Stage 3(gs/train_3dgs.py)에서 이 mesh 표면을 Gaussian 초기화에 사용.
"""

from __future__ import annotations

import json
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


def _inject_dummy_chumpy() -> None:
    """
    chumpy를 실제로 import하지 않고, pickle이 chumpy 클래스를 참조할 때
    사용할 더미 모듈을 sys.modules에 주입. Python 3.12 등에서 chumpy 크래시 방지.
    """
    if "chumpy" in sys.modules:
        return

    class _DummyChumpy:
        """Pickle이 역직렬화할 때 사용. .r에 numpy 배열을 노출 (chumpy 호환)."""

        def __new__(cls, *args):
            obj = object.__new__(cls)
            if args and (hasattr(args[0], "shape") or hasattr(args[0], "r")):
                obj.r = _extract_array(args[0]) or np.array([])
            else:
                obj.r = np.array([])
            return obj

        def __setstate__(self, state):
            arr = None
            if isinstance(state, tuple) and len(state) > 0:
                arr = _extract_array(state[0])
            elif isinstance(state, dict):
                if "r" in state:
                    arr = _extract_array(state["r"])
                else:
                    for v in state.values():
                        arr = _extract_array(v)
                        if arr is not None:
                            break
            else:
                arr = _extract_array(state)
            if arr is not None:
                self.r = arr

    def _extract_array(x):
        if isinstance(x, np.ndarray):
            return np.asarray(x, dtype=np.float64)
        if hasattr(x, "r"):
            return np.asarray(x.r, dtype=np.float64)
        if hasattr(x, "shape"):
            return np.asarray(x, dtype=np.float64)
        return None

    chumpy = types.ModuleType("chumpy")
    chumpy.ch = types.ModuleType("chumpy.ch")
    chumpy.chumpy = types.ModuleType("chumpy.chumpy")
    chumpy.Ch = _DummyChumpy
    chumpy.Core = _DummyChumpy
    chumpy.ch.Ch = _DummyChumpy
    chumpy.chumpy.Core = _DummyChumpy
    chumpy.chumpy.Chumpy = _DummyChumpy
    sys.modules["chumpy"] = chumpy
    sys.modules["chumpy.ch"] = chumpy.ch
    sys.modules["chumpy.chumpy"] = chumpy.chumpy


# -----------------------------------------------------------------------------
# 경로
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_IMAGES_DIR = PROJECT_ROOT / "data" / "processed_images"
HUMAN_PRIOR_DIR = PROJECT_ROOT / "data" / "human_prior"
# SMPL 모델: 프로젝트 내 data/smpl/ 또는 환경변수 SMPL_MODEL_PATH
SMPL_MODEL_DIR = PROJECT_ROOT / "data" / "smpl"
SMPL_PKL_NAME = "SMPL_NEUTRAL.pkl"

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

MIN_IMAGES = 3
SYNTHETIC_ORBIT_RADIUS = 3.0
SYNTHETIC_FOCAL_SCALE = 1.2


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
        for row in faces:
            f.write(np.uint8(3).tobytes())
            f.write(row.astype(np.int32).tobytes())
    return


def _to_numpy(x: Any) -> np.ndarray:
    """
    다양한 SMPL pkl 타입을 numpy ndarray로 변환.
    - numpy ndarray → 그대로 반환
    - hasattr(x, 'r') (chumpy/dummy) → np.asarray(x.r)
    - scipy sparse (toarray) → np.asarray(x.toarray())
    - list/tuple → np.asarray(x)
    실패 시 명확한 TypeError.
    """
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "r"):
        return np.asarray(x.r)
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    try:
        out = np.asarray(x)
        if out.dtype == object or out.ndim == 0:
            raise TypeError(f"Cannot convert to numpy: type={type(x).__name__}, repr={repr(x)[:80]}")
        return out
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"[Stage2] Cannot convert to numpy array: type={type(x).__name__}. "
            f"Expected ndarray, object with .r, scipy sparse with .toarray(), or list/tuple. {e}"
        ) from e


def load_smpl_canonical_tpose(
    pkl_path: Path,
    beta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SMPL_NEUTRAL.pkl에서 T-pose(pose=0) 메시만 생성.
    chumpy 미사용: 더미 chumpy를 sys.modules에 주입 후 pickle 로드.
    v_template, shapedirs, f만 numpy로 추출. posedirs/촘/chumpy graph 미사용.
    반환: (vertices, faces), vertices (6890, 3), faces (13776, 3).
    """
    path = Path(pkl_path)
    if not path.exists():
        raise FileNotFoundError(f"[Stage2] SMPL model not found: {path}")

    _inject_dummy_chumpy()
    with open(path, "rb") as f:
        try:
            model = pickle.load(f, encoding="latin1")
        except TypeError:
            model = pickle.load(f)

    v_template = _to_numpy(model["v_template"]).astype(np.float64)
    shapedirs = _to_numpy(model["shapedirs"])
    # (6890, 3, K) → (-1, K) 로 flatten 후 float64
    if shapedirs.ndim == 3:
        shapedirs = shapedirs.reshape(-1, shapedirs.shape[-1])
    shapedirs = shapedirs.astype(np.float64)
    faces = _to_numpy(model.get("f", model.get("faces", []))).astype(np.int32)

    num_betas = shapedirs.shape[1]
    if beta is None:
        beta = np.zeros(num_betas, dtype=np.float64)
    else:
        beta = np.asarray(beta, dtype=np.float64).flatten()
        if len(beta) < num_betas:
            beta = np.pad(beta, (0, num_betas - len(beta)))
        elif len(beta) > num_betas:
            beta = beta[:num_betas]

    # V = v_template + (shapedirs @ beta).reshape(-1, 3); shapedirs (6890*3, num_betas)
    v_shaped = v_template + (shapedirs @ beta).reshape(-1, 3)
    vertices = np.ascontiguousarray(v_shaped.astype(np.float32))
    return vertices, faces


def _synthetic_cameras(
    image_paths: list[Path],
    orbit_radius: float,
    focal_scale: float,
) -> list[dict[str, Any]]:
    """원형 궤도 카메라만 생성. 세계 좌표: 원점 = 인체 중심."""
    n = len(image_paths)
    cameras = []
    for i, path in enumerate(image_paths):
        w, h = _image_size(path)
        angle = 2 * np.pi * i / max(n, 1)
        cx = orbit_radius * np.cos(angle)
        cz = orbit_radius * np.sin(angle)
        cam_pos = np.array([cx, 0.0, cz])
        forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        R = np.eye(3)
        R[0], R[1], R[2] = right, -up, -forward
        t = (-R @ cam_pos).tolist()
        f = max(w, h) * focal_scale
        K = [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]
        cameras.append({
            "image": path.name,
            "K": K,
            "R": R.tolist(),
            "t": t,
            "width": w,
            "height": h,
        })
    return cameras


def _resolve_smpl_path(smpl_path: Path | None) -> Path:
    env = __import__("os").environ.get("SMPL_MODEL_PATH")
    if env:
        p = Path(env).resolve()
        if p.is_file():
            return p
        if (p / SMPL_PKL_NAME).is_file():
            return p / SMPL_PKL_NAME
    if smpl_path is not None:
        p = Path(smpl_path).resolve()
        if p.is_file():
            return p
        if (p / SMPL_PKL_NAME).is_file():
            return p / SMPL_PKL_NAME
    default = SMPL_MODEL_DIR / SMPL_PKL_NAME
    if default.is_file():
        return default
    raise FileNotFoundError(
        f"[Stage2] SMPL model not found. Place {SMPL_PKL_NAME} in:\n"
        f"  - {SMPL_MODEL_DIR}\n"
        f"  - or set env SMPL_MODEL_PATH to file or directory containing it."
    )


def run(
    image_dir: Path | None = None,
    output_dir: Path | None = None,
    smpl_path: Path | None = None,
    beta: np.ndarray | None = None,
) -> Path:
    """
    Stage 2: Human shape prior 생성.
    - SMPL_NEUTRAL.pkl → canonical T-pose mesh (canonical_mesh.ply).
    - Synthetic orbit cameras (cameras.json, image_list.txt).

    - image_dir: 입력 이미지 디렉터리 (기본: data/processed_images)
    - output_dir: 출력 디렉터리 (기본: data/human_prior)
    - smpl_path: SMPL_NEUTRAL.pkl 파일 경로 또는 해당 파일이 있는 디렉터리
    - beta: SMPL shape 계수 (10,). None이면 0 (평균 body).
    """
    image_dir = image_dir or PROCESSED_IMAGES_DIR
    output_dir = output_dir or HUMAN_PRIOR_DIR
    image_dir = Path(image_dir).resolve()
    output_dir = Path(output_dir).resolve()

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
    print(f"[Stage2] Found {len(image_paths)} images. Building human shape prior (no COLMAP/ROMP).", flush=True)

    pkl_path = _resolve_smpl_path(smpl_path)
    print(f"[Stage2] Loading SMPL from {pkl_path}", flush=True)
    vertices, faces = load_smpl_canonical_tpose(pkl_path, beta=beta)
    print(f"[Stage2] Canonical T-pose mesh: {len(vertices)} vertices, {len(faces)} faces.", flush=True)

    cameras = _synthetic_cameras(
        image_paths,
        orbit_radius=SYNTHETIC_ORBIT_RADIUS,
        focal_scale=SYNTHETIC_FOCAL_SCALE,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = output_dir / "canonical_mesh.ply"
    _write_ply(vertices, faces, mesh_path)
    print(f"[Stage2] Wrote {mesh_path} (vertices={len(vertices)}, faces={len(faces)}).", flush=True)

    cameras_path = output_dir / "cameras.json"
    with open(cameras_path, "w", encoding="utf-8") as f:
        json.dump({"cameras": cameras}, f, indent=2)
    print(f"[Stage2] Wrote {cameras_path} ({len(cameras)} cameras).", flush=True)

    list_path = output_dir / "image_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for c in cameras:
            f.write(c["image"] + "\n")
    print(f"[Stage2] Wrote {list_path}.", flush=True)

    print("[Stage2] Success. Output: data/human_prior/", flush=True)
    return output_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2: Human shape prior — SMPL T-pose mesh + synthetic cameras (no COLMAP/ROMP)"
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
        "--smpl",
        type=Path,
        default=None,
        help=f"Path to {SMPL_PKL_NAME} or directory containing it. Default: {SMPL_MODEL_DIR}/",
    )
    args = parser.parse_args()

    try:
        run(
            image_dir=args.images,
            output_dir=args.out,
            smpl_path=args.smpl,
        )
        print("Stage 2 done. Output: data/human_prior/ (canonical_mesh.ply, cameras.json, image_list.txt)", flush=True)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
