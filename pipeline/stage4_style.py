"""
Stage 4: 템플릿 기반 SD/룩업 스타일 캐릭터 생성 (파이프라인 최종 단계).

고정 템플릿(sd_chibi_base.ply)을 로드하고, Stage 3 조건을 적용한 뒤
얼굴(눈/코/입), 머리카락, 의상(상의/하의) procedural geometry를 추가하여
하나의 PLY로 렌더 가능한 캐릭터를 출력한다.
Blender/딥러닝 없음, pure Python + numpy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = PROJECT_ROOT / "character" / "templates" / "sd_chibi_base.ply"
DEFAULT_CONDITION_JSON = PROJECT_ROOT / "data" / "gs_output" / "condition_deltas.json"
DEFAULT_COLORS_JSON = PROJECT_ROOT / "data" / "gs_output" / "character_colors.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "characters"
DEFAULT_OUT_PLY = "character_output.ply"

# SD 스타일 기본 색 (0–255). Stage 2에서 추출 시 character_colors.json으로 덮어쓸 수 있음.
DEFAULT_COLORS = {
    "eye_color": [65, 105, 225],       # 밝은 파랑
    "hair_color": [101, 67, 33],       # 갈색
    "skin_color": [255, 220, 200],     # 밝은 피부
    "top_color": [240, 248, 255],     # 상의 (밝은 흰색에 가까움)
    "bottom_color": [70, 130, 180],    # 하의 (진한 파랑)
    "mouth_color": [139, 90, 43],      # 입 (어두운 갈색)
    "nose_color": [255, 220, 200],     # 코 (피부와 동일)
}

# property type → byte size (binary PLY)
_PLY_PROP_SIZE = {"float": 4, "double": 8, "int": 4, "uint": 4, "uchar": 1, "short": 2}


def load_character_colors(path: Path | None) -> dict[str, list[int]]:
    """Load RGB colors (0–255) from JSON. Missing file or keys → DEFAULT_COLORS."""
    out = dict(DEFAULT_COLORS)
    if path is None or not path.exists():
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("eye_color", "hair_color", "skin_color", "top_color", "bottom_color", "mouth_color", "nose_color"):
            if key in data and isinstance(data[key], (list, tuple)) and len(data[key]) >= 3:
                out[key] = [int(max(0, min(255, c))) for c in data[key][:3]]
    except Exception:
        pass
    return out


def _template_regions(vertices: np.ndarray) -> dict[str, Any]:
    """Y-up: head = 상단, torso = 중간, legs = 하단. bbox 및 중심·반경 추정."""
    v = np.asarray(vertices, dtype=np.float64)
    y_min, y_max = float(np.min(v[:, 1])), float(np.max(v[:, 1]))
    x_min, x_max = float(np.min(v[:, 0])), float(np.max(v[:, 0]))
    z_min, z_max = float(np.min(v[:, 2])), float(np.max(v[:, 2]))
    height = y_max - y_min
    if height <= 0:
        height = 1.0
    # SD 비율: 머리 ~22%, 몸통 ~35%, 하체 ~43%
    head_frac, torso_frac = 0.22, 0.35
    y_head_bottom = y_max - height * head_frac
    y_torso_bottom = y_head_bottom - height * torso_frac
    head_mask = v[:, 1] >= y_head_bottom
    torso_mask = (v[:, 1] >= y_torso_bottom) & (v[:, 1] < y_head_bottom)
    leg_mask = v[:, 1] < y_torso_bottom
    def _center_radius(mask):
        if not np.any(mask):
            return np.array([0.0, 0.0, 0.0]), 0.0
        pts = v[mask]
        c = pts.mean(axis=0)
        r = float(np.max(np.linalg.norm(pts - c, axis=1)))
        return c, max(r, 1e-6)
    head_center, head_radius = _center_radius(head_mask)
    torso_center, torso_radius = _center_radius(torso_mask)
    leg_center, leg_radius = _center_radius(leg_mask)
    return {
        "y_min": y_min, "y_max": y_max, "height": height,
        "head_center": head_center, "head_radius": head_radius,
        "head_top_y": y_max, "head_bottom_y": y_head_bottom,
        "torso_center": torso_center, "torso_radius": torso_radius,
        "torso_top_y": y_head_bottom, "torso_bottom_y": y_torso_bottom,
        "leg_center": leg_center, "leg_radius": leg_radius,
        "leg_top_y": y_torso_bottom, "leg_bottom_y": y_min,
        "x_mid": (x_min + x_max) * 0.5, "z_mid": (z_min + z_max) * 0.5,
    }


def _make_ellipsoid(
    cx: float, cy: float, cz: float,
    rx: float, ry: float, rz: float,
    color: list[int],
    segments: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Y-up ellipsoid; returns vertices (N,3), faces (F,3), colors (N,3) uchar."""
    lat_steps = max(2, segments // 2)
    lon_steps = max(4, segments)
    n_cols = lon_steps + 1
    verts, colors_list = [], []
    for i in range(lat_steps + 1):
        lat = np.pi * i / lat_steps
        for j in range(lon_steps + 1):
            lon = 2 * np.pi * j / lon_steps
            sy, sxz = np.cos(lat), np.sin(lat)
            x = cx + rx * sxz * np.cos(lon)
            y = cy + ry * sy
            z = cz + rz * sxz * np.sin(lon)
            verts.append([x, y, z])
            colors_list.append(color)
    verts = np.array(verts, dtype=np.float32)
    colors_list = np.array(colors_list, dtype=np.uint8)
    faces = []
    for i in range(lat_steps):
        for j in range(lon_steps):
            a = i * n_cols + j
            b = a + 1
            c = a + n_cols
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return verts, np.array(faces, dtype=np.int32), colors_list


def _make_triangle(p0: list[float], p1: list[float], p2: list[float], color: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single triangle mesh."""
    v = np.array([p0, p1, p2], dtype=np.float32)
    c = np.tile(np.array(color, dtype=np.uint8), (3, 1))
    f = np.array([[0, 1, 2]], dtype=np.int32)
    return v, f, c


def _make_cylinder(
    cx: float, cy: float, cz: float,
    radius: float, height: float,
    color: list[int],
    axis: int = 1,
    segments: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cylinder along axis (0=X, 1=Y, 2=Z). Y-up default: axis=1. Includes cap centers."""
    verts, colors_list = [], []
    half = height / 2
    for s in range(segments + 1):
        t = 2 * np.pi * s / segments
        if axis == 0:
            y, z = radius * np.cos(t), radius * np.sin(t)
            verts.append([cx - half, cy + y, cz + z])
            verts.append([cx + half, cy + y, cz + z])
        elif axis == 1:
            x, z = radius * np.cos(t), radius * np.sin(t)
            verts.append([cx + x, cy - half, cz + z])
            verts.append([cx + x, cy + half, cz + z])
        else:
            x, y = radius * np.cos(t), radius * np.sin(t)
            verts.append([cx + x, cy + y, cz - half])
            verts.append([cx + x, cy + y, cz + half])
        colors_list.append(color)
        colors_list.append(color)
    # two cap center vertices
    if axis == 0:
        verts.append([cx - half, cy, cz])
        verts.append([cx + half, cy, cz])
    elif axis == 1:
        verts.append([cx, cy - half, cz])
        verts.append([cx, cy + half, cz])
    else:
        verts.append([cx, cy, cz - half])
        verts.append([cx, cy, cz + half])
    colors_list.append(color)
    colors_list.append(color)
    verts = np.array(verts, dtype=np.float32)
    colors_list = np.array(colors_list, dtype=np.uint8)
    faces = []
    n_ring = (segments + 1) * 2
    idx_bottom_center = n_ring
    idx_top_center = n_ring + 1
    for s in range(segments):
        a, b = s * 2, ((s + 1) % (segments + 1)) * 2
        c, d = b + 1, a + 1
        faces.append([a, b, c])
        faces.append([a, c, d])
    for s in range(segments):
        i0, i1 = 2 * s, 2 * ((s + 1) % (segments + 1))
        faces.append([idx_bottom_center, i1, i0])
        faces.append([idx_top_center, i0 + 1, i1 + 1])
    return verts, np.array(faces, dtype=np.int32), colors_list


def _make_dome_cap(
    cx: float, cy: float, cz: float,
    radius_x: float, radius_y: float, radius_z: float,
    color: list[int],
    segments: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper half of ellipsoid (Y-up dome). cy = bottom of dome center; cap extends upward."""
    lat_steps = max(2, segments // 2)
    lon_steps = max(4, segments)
    n_cols = lon_steps + 1
    verts, colors_list = [], []
    for i in range(lat_steps + 1):
        lat = 0.5 * np.pi * (1 - i / lat_steps)  # 0 = top, pi/2 = bottom
        for j in range(lon_steps + 1):
            lon = 2 * np.pi * j / lon_steps
            sy, sxz = np.cos(lat), np.sin(lat)
            x = cx + radius_x * sxz * np.cos(lon)
            y = cy + radius_y * sy
            z = cz + radius_z * sxz * np.sin(lon)
            verts.append([x, y, z])
            colors_list.append(color)
    verts = np.array(verts, dtype=np.float32)
    colors_list = np.array(colors_list, dtype=np.uint8)
    faces = []
    for i in range(lat_steps):
        for j in range(lon_steps):
            a = i * n_cols + j
            b = a + 1
            c = a + n_cols
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return verts, np.array(faces, dtype=np.int32), colors_list


def _merge_meshes(
    meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """List of (vertices, faces, colors). Returns (V, F, C) with reindexed faces."""
    all_v, all_f, all_c = [], [], []
    offset = 0
    for v, f, c in meshes:
        n = len(v)
        all_v.append(v)
        all_c.append(c)
        all_f.append(f + offset)
        offset += n
    return (
        np.concatenate(all_v, axis=0).astype(np.float32),
        np.concatenate(all_f, axis=0).astype(np.int32),
        np.concatenate(all_c, axis=0).astype(np.uint8),
    )


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
    """Write mesh to binary PLY (position only)."""
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


def _write_ply_colored(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    out_path: Path,
) -> None:
    """Write mesh to binary PLY with vertex colors (property uchar red green blue)."""
    nv, nf = len(vertices), len(faces)
    if colors.shape[0] != nv or colors.shape[1] != 3:
        colors = np.tile(np.array([255, 220, 200], dtype=np.uint8), (nv, 1))
    with open(out_path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {nv}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {nf}\n".encode())
        f.write(b"property list uchar int vertex_indices\nend_header\n")
        for i in range(nv):
            f.write(vertices[i].astype(np.float32).tobytes())
            f.write(colors[i].astype(np.uint8).tobytes())
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


def build_face_parts(regions: dict[str, Any], colors: dict[str, list[int]]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """눈(큰 ellipsoid), 코(작은 삼각형), 입(작은 삼각형). SD 비율, 실사 미사용."""
    meshes = []
    hc = regions["head_center"]
    hr = regions["head_radius"]
    eye_y = hc[1]
    eye_offset_x = hr * 0.32
    eye_z = hc[2] + hr * 0.25
    eye_rx, eye_ry, eye_rz = hr * 0.22, hr * 0.12, hr * 0.14
    for side in (-1, 1):
        v, f, c = _make_ellipsoid(
            hc[0] + side * eye_offset_x, eye_y, eye_z,
            eye_rx, eye_ry, eye_rz,
            colors["eye_color"],
            segments=10,
        )
        meshes.append((v, f, c))
    nose_z = hc[2] + hr * 0.15
    nose_dz = hr * 0.04
    v, f, c = _make_triangle(
        [hc[0], hc[1], nose_z],
        [hc[0] - hr * 0.02, hc[1], nose_z + nose_dz],
        [hc[0] + hr * 0.02, hc[1], nose_z + nose_dz],
        colors["nose_color"],
    )
    meshes.append((v, f, c))
    mouth_y = hc[1] - hr * 0.2
    mouth_z = hc[2] + hr * 0.08
    v, f, c = _make_triangle(
        [hc[0] - hr * 0.06, mouth_y, mouth_z],
        [hc[0] + hr * 0.06, mouth_y, mouth_z],
        [hc[0], mouth_y - hr * 0.03, mouth_z],
        colors["mouth_color"],
    )
    meshes.append((v, f, c))
    return meshes


def build_hair(regions: dict[str, Any], colors: dict[str, list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """머리 위 볼륨감 있는 둥근 cap. 실루엣 위주, 너무 풍성하지 않게."""
    hc = regions["head_center"]
    hr = regions["head_radius"]
    top_y = regions["head_top_y"]
    cap_cy = top_y  # dome bottom at head top; cap extends upward
    rx, ry, rz = hr * 1.06, hr * 0.75, hr * 1.06
    return _make_dome_cap(hc[0], cap_cy, hc[2], rx, ry, rz, colors["hair_color"], segments=14)


def build_clothing(
    regions: dict[str, Any],
    colors: dict[str, list[int]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """상의/하의 단순 실린더. 몸에 약간 여유 있는 핏."""
    meshes = []
    tc = regions["torso_center"]
    tr = regions["torso_radius"]
    th = regions["torso_top_y"] - regions["torso_bottom_y"]
    v, f, c = _make_cylinder(tc[0], tc[1], tc[2], tr * 1.06, th * 1.02, colors["top_color"], axis=1, segments=18)
    meshes.append((v, f, c))
    lc = regions["leg_center"]
    lr = regions["leg_radius"]
    lh = regions["leg_top_y"] - regions["leg_bottom_y"]
    if lh > 0 and lr > 0:
        v, f, c = _make_cylinder(lc[0], lc[1], lc[2], lr * 1.08, lh * 1.02, colors["bottom_color"], axis=1, segments=18)
        meshes.append((v, f, c))
    return meshes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: Template + condition + procedural face/hair/clothing → SD character PLY."
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
        "--colors",
        type=Path,
        default=DEFAULT_COLORS_JSON,
        help=f"Character colors JSON (eye/hair/skin/top/bottom); optional (default: {DEFAULT_COLORS_JSON})",
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
    colors = load_character_colors(args.colors)

    vertices, faces = _load_ply_vertices_faces(args.template)
    vertices = apply_condition_deltas(vertices, condition)
    regions = _template_regions(vertices)

    skin_color = np.tile(np.array(colors["skin_color"], dtype=np.uint8), (len(vertices), 1))
    body_mesh = (vertices, faces, skin_color)

    all_meshes = [body_mesh]
    all_meshes.extend(build_face_parts(regions, colors))
    all_meshes.append(build_hair(regions, colors))
    all_meshes.extend(build_clothing(regions, colors))

    V, F, C = _merge_meshes(all_meshes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name
    _write_ply_colored(V, F, C, out_path)
    print(f"Stage 4 done. Character PLY (with face/hair/clothing) written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
