"""
Stage 2: Geometry / COLMAP — camera pose + sparse point cloud 생성.

- 입력: Stage 1 출력 data/processed_images/ (배경 제거·크기 정규화·실패 프레임 제거 완료)
- 이미지 품질 판단·마스크 처리 없음. 기하 복원만 수행.
- 출력: data/colmap/sparse/, data/colmap/database.db → Stage 3 (3DGS) 입력용.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Headless(Colab 등) 환경: Qt/OpenGL context 오류 방지
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# -----------------------------------------------------------------------------
# 경로
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_IMAGES_DIR = PROJECT_ROOT / "data" / "processed_images"
COLMAP_OUTPUT_DIR = PROJECT_ROOT / "data" / "colmap"
COLMAP_SPARSE_DIR = COLMAP_OUTPUT_DIR / "sparse"
COLMAP_DATABASE_PATH = COLMAP_OUTPUT_DIR / "database.db"

# Stage 1과 동일한 확장자
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# -----------------------------------------------------------------------------
# COLMAP 옵션 (상수로 명시)
# -----------------------------------------------------------------------------
# 카메라: 단일 카메라 + SIMPLE_PINHOLE
CAMERA_MODEL = "SIMPLE_PINHOLE"
SINGLE_CAMERA = True

# Feature extractor (SiftExtraction.* — COLMAP 3.7+ 문법)
# headless 환경에서 OpenGL context 생성 실패를 피하기 위해 CPU 고정
FEATURE_EXTRACTOR_MAX_IMAGE_SIZE = 3200
FEATURE_EXTRACTOR_USE_GPU = 0  # 0=CPU only (Colab/headless 안정 동작)
SIFT_MAX_NUM_FEATURES = 8192

# Matcher (exhaustive_matcher): headless에서 OpenGL 크래시 방지 위해 CPU 고정
FEATURE_MATCHING_USE_GPU = 0

# Mapper
MAPPER_MULTIPLE_MODELS = 0  # 단일 모델만 사용

# 복원 실패 판정 임계값
MIN_CAMERAS = 3
MIN_POINTS3D = 100


def get_colmap_exe() -> Path | None:
    """COLMAP 실행 파일 경로 반환. 없으면 None."""
    # Windows: COLMAP.bat 또는 colmap.exe
    if sys.platform == "win32":
        candidates = ["COLMAP.bat", "colmap.bat", "colmap.exe", "colmap"]
    else:
        candidates = ["colmap"]
    env_exe = os.environ.get("COLMAP_EXE")
    if env_exe:
        p = shutil.which(env_exe) or (Path(env_exe) if Path(env_exe).is_absolute() else None)
        if p is None and Path(env_exe).exists():
            p = Path(env_exe).resolve()
        if p is not None:
            return Path(p) if isinstance(p, str) else p
    for name in candidates:
        exe = shutil.which(name)
        if exe:
            return Path(exe)
    return None


def list_image_paths(dir_path: Path) -> list[Path]:
    """지원 확장자 이미지 목록을 파일명 기준 정렬해 반환."""
    paths: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(dir_path.glob(f"*{ext}"))
    return sorted(paths, key=lambda p: p.name.lower())


def run_command(
    exe: Path,
    args: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """COLMAP subprocess 실행. headless 환경을 위해 QT_QPA_PLATFORM 보장."""
    cmd = [str(exe)] + args
    proc_env = (os.environ if env is None else {**os.environ, **env}).copy()
    proc_env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=proc_env,
        capture_output=True,
        check=False,
        timeout=3600,
    )


def run_feature_extractor(
    colmap_exe: Path,
    database_path: Path,
    image_path: Path,
) -> None:
    """COLMAP feature_extractor 실행."""
    args = [
        "feature_extractor",
        f"--database_path={database_path}",
        f"--image_path={image_path}",
        f"--ImageReader.camera_model={CAMERA_MODEL}",
        f"--ImageReader.single_camera={1 if SINGLE_CAMERA else 0}",
        f"--SiftExtraction.max_image_size={FEATURE_EXTRACTOR_MAX_IMAGE_SIZE}",
        f"--SiftExtraction.use_gpu={FEATURE_EXTRACTOR_USE_GPU}",
        f"--SiftExtraction.max_num_features={SIFT_MAX_NUM_FEATURES}",
    ]
    proc = run_command(colmap_exe, args)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(
            f"COLMAP feature_extractor failed (exit {proc.returncode}).\nStderr:\n{stderr}"
        )
    print("[Stage2] feature_extractor done.", flush=True)


def run_matcher(colmap_exe: Path, database_path: Path) -> None:
    """COLMAP exhaustive_matcher 실행."""
    args = [
        "exhaustive_matcher",
        f"--database_path={database_path}",
        f"--SiftMatching.use_gpu={FEATURE_MATCHING_USE_GPU}",
    ]
    proc = run_command(colmap_exe, args)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(
            f"COLMAP exhaustive_matcher failed (exit {proc.returncode}).\nStderr:\n{stderr}"
        )
    print("[Stage2] exhaustive_matcher done.", flush=True)


def run_mapper(
    colmap_exe: Path,
    database_path: Path,
    image_path: Path,
    output_path: Path,
) -> None:
    """COLMAP mapper 실행."""
    args = [
        "mapper",
        f"--database_path={database_path}",
        f"--image_path={image_path}",
        f"--output_path={output_path}",
        f"--Mapper.multiple_models={MAPPER_MULTIPLE_MODELS}",
    ]
    proc = run_command(colmap_exe, args)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(
            f"COLMAP mapper failed (exit {proc.returncode}).\nStderr:\n{stderr}"
        )
    print("[Stage2] mapper done.", flush=True)


def get_model_stats(colmap_exe: Path, model_path: Path) -> tuple[int, int]:
    """
    model_analyzer 출력에서 카메라 수, 3D 포인트 수 파싱.
    반환: (num_cameras, num_points3d)
    """
    args = ["model_analyzer", f"--path={model_path}"]
    proc = run_command(colmap_exe, args)
    text = (proc.stdout or b"").decode("utf-8", errors="replace")
    # 예: "Cameras: 12" / "Registered images: 12" / "Points3D: 12345"
    cameras = 0
    points = 0
    for line in text.splitlines():
        if "Cameras:" in line or "Registered images:" in line:
            m = re.search(r"(\d+)", line)
            if m and cameras == 0:
                cameras = int(m.group(1))
        if "Points3D:" in line or "Points:" in line:
            m = re.search(r"(\d+)", line)
            if m:
                points = int(m.group(1))
    return cameras, points


def find_sparse_model_dir(sparse_root: Path) -> Path | None:
    """sparse/ 아래 첫 번째 모델 디렉터리(sparse/0 등) 반환."""
    if not sparse_root.is_dir():
        return None
    for name in sorted(sparse_root.iterdir(), key=lambda p: p.name):
        if name.is_dir() and (name / "cameras.bin").exists():
            return name
        if name.is_dir() and (name / "cameras.txt").exists():
            return name
    return None


def run(
    image_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """
    Stage 2 파이프라인: COLMAP feature_extractor → exhaustive_matcher → mapper.

    - image_dir: 입력 이미지 디렉터리 (기본: data/processed_images)
    - output_dir: COLMAP 출력 루트 (기본: data/colmap). sparse/, database.db 생성.
    - 반환: sparse 모델 디렉터리 (예: data/colmap/sparse/0).
    """
    image_dir = image_dir or PROCESSED_IMAGES_DIR
    output_dir = output_dir or COLMAP_OUTPUT_DIR

    image_dir = image_dir.resolve()
    output_dir = output_dir.resolve()

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"[Stage2] Input image directory not found: {image_dir}\n"
            "Run Stage 1 first to generate data/processed_images/."
        )

    image_paths = list_image_paths(image_dir)
    if len(image_paths) < MIN_CAMERAS:
        raise ValueError(
            f"[Stage2] Too few images: {len(image_paths)} (need at least {MIN_CAMERAS}).\n"
            f"Image dir: {image_dir}"
        )
    print(f"[Stage2] Found {len(image_paths)} images in {image_dir}", flush=True)

    colmap_exe = get_colmap_exe()
    if colmap_exe is None:
        raise RuntimeError(
            "[Stage2] COLMAP not found. Please install COLMAP and add it to PATH,\n"
            "or set environment variable COLMAP_EXE to the path of colmap (or COLMAP.bat on Windows).\n"
            "Download: https://github.com/colmap/colmap/releases"
        )
    print(f"[Stage2] Using COLMAP: {colmap_exe}", flush=True)

    database_path = output_dir / "database.db"
    sparse_root = output_dir / "sparse"

    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_root.mkdir(parents=True, exist_ok=True)

    # 기존 DB가 있으면 제거해 깨끗한 상태에서 실행
    if database_path.exists():
        database_path.unlink()
        print("[Stage2] Removed existing database.db", flush=True)

    run_feature_extractor(colmap_exe, database_path, image_dir)
    run_matcher(colmap_exe, database_path)
    run_mapper(colmap_exe, database_path, image_dir, sparse_root)

    model_dir = find_sparse_model_dir(sparse_root)
    if model_dir is None:
        raise RuntimeError(
            f"[Stage2] Sparse reconstruction produced no model under {sparse_root}.\n"
            "Check COLMAP mapper output above for errors."
        )

    num_cameras, num_points = get_model_stats(colmap_exe, model_dir)
    print(f"[Stage2] Model: {num_cameras} cameras, {num_points} points3D", flush=True)

    if num_cameras < MIN_CAMERAS:
        raise RuntimeError(
            f"[Stage2] Reconstruction failed: only {num_cameras} cameras registered "
            f"(minimum required: {MIN_CAMERAS}).\n"
            "Try more/better overlapping images or check image content."
        )
    if num_points < MIN_POINTS3D:
        raise RuntimeError(
            f"[Stage2] Reconstruction failed: only {num_points} 3D points "
            f"(minimum required: {MIN_POINTS3D}).\n"
            "Try more overlapping views or ensure sufficient texture/features."
        )

    print(f"[Stage2] Success. Sparse model: {model_dir}", flush=True)
    return model_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2: COLMAP geometry — camera poses + sparse point cloud"
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
        default=COLMAP_OUTPUT_DIR,
        help="COLMAP output directory (default: data/colmap)",
    )
    args = parser.parse_args()

    try:
        run(image_dir=args.images, output_dir=args.out)
        print("Stage 2 done. Output: data/colmap/sparse/, data/colmap/database.db", flush=True)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
