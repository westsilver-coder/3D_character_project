"""
Stage 4: 템플릿 기반 캐릭터 생성 (파이프라인 최종 단계).

고정 템플릿(sd_chibi_base.ply)을 로드하고, Stage 3 조건(condition_deltas.json)을
적용하여 렌더 가능한 캐릭터 geometry/appearance를 생성한다.
렌더링(턴테이블 등)은 이 단계가 아닌 gs/render_character.py로 선택 수행.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = PROJECT_ROOT / "character" / "templates" / "sd_chibi_base.ply"
DEFAULT_CONDITION_JSON = PROJECT_ROOT / "data" / "gs_output" / "condition_deltas.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "characters"
DEFAULT_OUT_PLY = "character_output.ply"

# property type → byte size (binary PLY)
_PLY_PROP_SIZE = {"float": 4, "double": 8, "int": 4, "uint": 4, "uchar": 1, "short": 2}


def _load_ply_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load vertices (xyz only) and faces from binary PLY.
    Supports templates with extra vertex properties (normals, colors) by parsing header stride.
    """
    with open(path, "rb") as f:
        lines = []
        while True:
            line = f.readline()
            lines.append(line)
            if b"end_header" in line:
                break
        header = b"".join(lines).decode("utf-8", errors="replace")
    lines_s = [s.strip() for s in header.split("\n")]

    nv = nf = 0
    vertex_stride = 0
    in_vertex = False
    for s in lines_s:
        if s.startswith("element vertex "):
            nv = int(s.split()[-1])
            in_vertex = True
            continue
        if s.startswith("element face "):
            nf = int(s.split()[-1])
            in_vertex = False
            continue
        if in_vertex and s.startswith("property "):
            parts = s.split()
            if len(parts) >= 3:
                t = parts[1].lower()
                vertex_stride += _PLY_PROP_SIZE.get(t, 4)

    if vertex_stride < 12:
        vertex_stride = 12  # at least x,y,z
    if nv == 0 or nf == 0:
        raise ValueError(f"Invalid PLY header: nv={nv}, nf={nf}")

    with open(path, "rb") as f:
        while True:
            if f.readline().rstrip() == b"end_header":
                break
        raw = f.read(nv * vertex_stride)
        remainder = f.read()
    # First 12 bytes per vertex = x, y, z (float32); stride may be larger if template has normals/colors
    vertices = np.zeros((nv, 3), dtype=np.float32)
    for i in range(nv):
        start = i * vertex_stride
        vertices[i] = np.frombuffer(raw[start : start + 12], dtype=np.float32)

    # Face format: uchar n, then n int32 indices
    faces = []
    offset = 0
    for _ in range(nf):
        if offset >= len(remainder):
            break
        n = int(remainder[offset])
        offset += 1
        if n >= 3 and offset + n * 4 <= len(remainder):
            idx = np.frombuffer(remainder[offset : offset + n * 4], dtype=np.int32)
            faces.append(idx[:3])
        offset += n * 4
    faces = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
    return vertices, faces


def _write_ply(vertices: np.ndarray, faces: np.ndarray, out_path: Path) -> None:
    """Write mesh to binary PLY (same format as stage2)."""
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


def apply_condition_deltas(
    vertices: np.ndarray,
    condition: dict[str, float],
) -> np.ndarray:
    """
    Apply Stage 3 condition deltas to vertex positions.
    Scale: X = 1 + shoulder_width_delta, Y = 1 + height_delta, Z = 1 + body_thickness_delta.
    Arm/leg deltas are not applied per-axis here (minimal: uniform or folded into overall scale).
    """
    center = vertices.mean(axis=0)
    v = vertices - center
    sx = 1.0 + condition.get("shoulder_width_delta", 0.0)
    sy = 1.0 + condition.get("height_delta", 0.0)
    sz = 1.0 + condition.get("body_thickness_delta", 0.0)
    v[:, 0] *= sx
    v[:, 1] *= sy
    v[:, 2] *= sz
    return v + center


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: Load template PLY, apply condition deltas, write final character PLY."
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help=f"Template PLY path (default: {DEFAULT_TEMPLATE})",
    )
    parser.add_argument(
        "--condition",
        type=Path,
        default=DEFAULT_CONDITION_JSON,
        help=f"Condition deltas JSON from Stage 3 (default: {DEFAULT_CONDITION_JSON})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default=DEFAULT_OUT_PLY,
        help=f"Output PLY filename (default: {DEFAULT_OUT_PLY})",
    )
    args = parser.parse_args()

    if not args.template.exists():
        print(f"Error: Template not found: {args.template}", file=sys.stderr)
        sys.exit(1)
    if not args.condition.exists():
        print(f"Error: Condition JSON not found: {args.condition}", file=sys.stderr)
        print("Run Stage 3 first: python -m pipeline.stage3_deform", file=sys.stderr)
        sys.exit(1)

    with open(args.condition, "r", encoding="utf-8") as f:
        condition = json.load(f)

    vertices, faces = _load_ply_vertices_faces(args.template)
    vertices = apply_condition_deltas(vertices, condition)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name
    _write_ply(vertices, faces, out_path)
    print(f"Stage 4 done. Character PLY written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
