"""
Stage 2 후반: 3D Gaussian Splatting 학습 (Human-Prior 기반).

COLMAP 대신 data/human_prior/ (canonical_mesh.ply, cameras.json, image_list.txt)를 사용.
- 카메라: cameras.json (각 이미지의 K, R, t in canonical body space)
- 초기 점: canonical_mesh.ply 표면 샘플링 또는 기본 구/랜덤

Geometry 기준: Canonical body space. docs/STAGE2_HUMAN_PRIOR.md 참고.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HUMAN_PRIOR_DIR = PROJECT_ROOT / "data" / "human_prior"
PROCESSED_IMAGES_DIR = PROJECT_ROOT / "data" / "processed_images"
OUTPUT_GS_DIR = PROJECT_ROOT / "data" / "gs_output"


def load_human_prior(data_dir: Path) -> dict[str, Any]:
    """
    data/human_prior/ 에서 cameras.json, image_list.txt, canonical_mesh.ply 로드.
    반환: {
      "cameras": [ {"image", "K", "R", "t", "width", "height"}, ... ],
      "image_list": [ "a.jpg", "b.jpg", ... ],
      "vertices": (N,3) or None,
      "faces": (F,3) or None,
    }
    """
    data_dir = Path(data_dir)
    out = {"cameras": [], "image_list": [], "vertices": None, "faces": None}

    cameras_path = data_dir / "cameras.json"
    if not cameras_path.exists():
        raise FileNotFoundError(f"[train_3dgs] Not found: {cameras_path}. Run Stage 2 first.")
    with open(cameras_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out["cameras"] = data.get("cameras", [])

    list_path = data_dir / "image_list.txt"
    if list_path.exists():
        with open(list_path, "r", encoding="utf-8") as f:
            out["image_list"] = [line.strip() for line in f if line.strip()]
    else:
        out["image_list"] = [c["image"] for c in out["cameras"]]

    mesh_path = data_dir / "canonical_mesh.ply"
    if mesh_path.exists():
        verts, faces = _load_ply(mesh_path)
        out["vertices"] = verts
        out["faces"] = faces
    return out


def _load_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from binary PLY (as written by stage2)."""
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break
        # Assume binary_little_endian, vertex float x,y,z, face list uchar int
        nv = int(next((l for l in header.decode().split("\n") if "element vertex" in l)).split()[-1])
        nf = int(next((l for l in header.decode().split("\n") if "element face" in l)).split()[-1])
        vertices = np.frombuffer(f.read(nv * 4 * 3), dtype=np.float32).reshape(-1, 3)
        faces = []
        for _ in range(nf):
            n = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
            idx = np.frombuffer(f.read(n * 4), dtype=np.int32)
            if n >= 3:
                faces.append(idx[:3])
        faces = np.array(faces, dtype=np.int32)
    return vertices, faces


def sample_points_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int = 50_000,
    sigma: float = 0.02,
) -> np.ndarray:
    """
    메시 표면에서 삼각형 면적 비율로 샘플링한 뒤, 약간의 노이즈(sigma)를 더해
    3DGS 초기 Gaussian 중심으로 사용할 점 cloud 반환. (N, 3).
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    areas = np.maximum(areas, 1e-8)
    areas /= areas.sum()
    tri_idx = np.random.choice(len(faces), size=num_points, replace=True, p=areas)
    u = np.random.rand(num_points, 1)
    v = np.random.rand(num_points, 1)
    u = np.minimum(u, 1 - v)
    w = 1 - u - v
    pts = (
        w * vertices[faces[tri_idx, 0]]
        + u * vertices[faces[tri_idx, 1]]
        + v * vertices[faces[tri_idx, 2]]
    )
    if sigma > 0:
        pts = pts + np.random.randn(*pts.shape).astype(np.float32) * sigma
    return pts.astype(np.float32)


def get_cameras_for_3dgs(prior: dict[str, Any]) -> list[dict[str, Any]]:
    """
    3DGS에서 사용할 카메라 리스트. 각 항목: K (3x3), R (3x3), t (3,), width, height, image (파일명).
    """
    return [
        {
            "K": np.array(c["K"]),
            "R": np.array(c["R"]),
            "t": np.array(c["t"]),
            "width": c["width"],
            "height": c["height"],
            "image": c["image"],
        }
        for c in prior["cameras"]
    ]


def create_initial_point_cloud(
    prior: dict[str, Any],
    num_points: int = 50_000,
    surface_sigma: float = 0.02,
    fallback_radius: float = 1.0,
) -> np.ndarray:
    """
    Canonical mesh가 있으면 표면 샘플링, 없으면 원점 중심 구 내부 균일 샘플링.
    반환: (N, 3) float32.
    """
    if prior.get("vertices") is not None and prior.get("faces") is not None:
        return sample_points_from_mesh(
            prior["vertices"],
            prior["faces"],
            num_points=num_points,
            sigma=surface_sigma,
        )
    # Fallback: 구 내부
    r = fallback_radius
    pts = np.random.randn(num_points, 3).astype(np.float32)
    pts = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8) * (r * np.random.rand(num_points, 1) ** (1 / 3))
    return pts


def run(
    human_prior_dir: Path | None = None,
    image_dir: Path | None = None,
    output_dir: Path | None = None,
    num_init_points: int = 50_000,
) -> Path:
    """
    Human-prior 데이터를 로드하고, 3DGS 학습용 초기 점 cloud와 카메라를 준비.
    실제 3DGS 학습 루프는 외부 라이브러리(diff-gaussian-splatting 등)와 연동 시
    여기서 만든 point_cloud + cameras를 넘기면 됨.

    - human_prior_dir: data/human_prior (Stage 2 출력)
    - image_dir: 전처리 이미지 디렉터리 (이미지 경로 해석용)
    - output_dir: 초기화 결과 저장 (point_cloud.npy, cameras.json 등)
    - 반환: output_dir
    """
    human_prior_dir = human_prior_dir or HUMAN_PRIOR_DIR
    image_dir = image_dir or PROCESSED_IMAGES_DIR
    output_dir = output_dir or OUTPUT_GS_DIR
    human_prior_dir = Path(human_prior_dir).resolve()
    output_dir = Path(output_dir).resolve()

    prior = load_human_prior(human_prior_dir)
    cameras = get_cameras_for_3dgs(prior)
    points = create_initial_point_cloud(prior, num_points=num_init_points)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "point_cloud.npy", points)
    # 카메라를 3DGS에서 쓰기 쉬운 형태로 저장 (numpy 불가하므로 JSON에 list로)
    cam_export = [
        {
            "image": c["image"],
            "K": c["K"].tolist(),
            "R": c["R"].tolist(),
            "t": c["t"].tolist(),
            "width": c["width"],
            "height": c["height"],
        }
        for c in cameras
    ]
    with open(output_dir / "cameras.json", "w", encoding="utf-8") as f:
        json.dump({"cameras": cam_export}, f, indent=2)
    with open(output_dir / "image_list.txt", "w", encoding="utf-8") as f:
        for c in cameras:
            f.write(c["image"] + "\n")

    print(f"[train_3dgs] Loaded {len(cameras)} cameras, {len(points)} initial points.", flush=True)
    print(f"[train_3dgs] Saved {output_dir}/point_cloud.npy, cameras.json, image_list.txt.", flush=True)
    print("[train_3dgs] To train 3DGS: use these cameras + point_cloud as initial Gaussians (e.g. diff-gaussian-splatting).", flush=True)
    return output_dir


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="3DGS init from human-prior (no COLMAP)")
    parser.add_argument("--human-prior", type=Path, default=HUMAN_PRIOR_DIR, help="data/human_prior dir")
    parser.add_argument("--images", type=Path, default=PROCESSED_IMAGES_DIR, help="Processed images dir")
    parser.add_argument("--out", type=Path, default=OUTPUT_GS_DIR, help="Output dir for point_cloud + cameras")
    parser.add_argument("--num-points", type=int, default=50_000, help="Initial point cloud size")
    args = parser.parse_args()
    try:
        run(
            human_prior_dir=args.human_prior,
            image_dir=args.images,
            output_dir=args.out,
            num_init_points=args.num_points,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
