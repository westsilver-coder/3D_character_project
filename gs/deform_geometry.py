"""
Stage 3: 템플릿에 적용할 비율/스케일 조건만 계산.

Geometry를 직접 변형하지 않음. 3DGS 또는 human-prior 결과(예: point_cloud.npy)에서
"사람의 체형 특징을 요약한 수치(조건)"만 계산하여, Stage 4에서 단일 템플릿(sd_chibi_base.ply)에
적용할 수 있도록 정리한다. 최종 캐릭터 생성은 Stage 4에서 수행한다.

- PLY 템플릿 로드 없음. Blender 연동 없음. 얼굴/눈/코/입 미처리.
- 템플릿은 1개(sd_chibi_base)만 전제.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# -----------------------------------------------------------------------------
# sd_chibi_base 기준 델타 허용 범위 (Stage 4 적용 시 clamp)
# deformation_rules.py의 허용 범위 정신에 맞춤. JSON 직렬화 가능.
# -----------------------------------------------------------------------------
DELTA_HEIGHT_MIN = -0.15
DELTA_HEIGHT_MAX = 0.15
DELTA_SHOULDER_MIN = -0.10
DELTA_SHOULDER_MAX = 0.10
DELTA_THICKNESS_MIN = -0.10
DELTA_THICKNESS_MAX = 0.10
DELTA_ARM_MIN = -0.12
DELTA_ARM_MAX = 0.12
DELTA_LEG_MIN = -0.12
DELTA_LEG_MAX = 0.12

# 기준: 평균 인간 비율에 가까운 상수. 실사 비율 그대로 쓰지 않음.
REF_HEIGHT = 1.0            # bbox height 기준 (정규화 후 1 또는 단위 스케일 가정)
REF_SHOULDER_RATIO = 0.28   # shoulder_width / height
REF_THICKNESS_RATIO = 0.12  # body_thickness / height
REF_ARM_RATIO = 0.35        # arm proxy / height
REF_LEG_RATIO = 0.45        # leg proxy / height

# (사람 - 기준) 차이에 곱할 작은 계수 → 귀여움 유지, 조금씩만 반영
SENSITIVITY_HEIGHT = 0.3
SENSITIVITY_SHOULDER = 0.4
SENSITIVITY_THICKNESS = 0.4
SENSITIVITY_ARM = 0.35
SENSITIVITY_LEG = 0.35


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _bbox_extents(xyz: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Returns (height, shoulder_proxy, thickness_proxy, arm_proxy, leg_proxy) from bbox regions.
    Assumes Y-up: y min = foot, y max = head.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N, 3)")
    if len(xyz) < 10:
        raise ValueError("Too few points to estimate proportions")
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    y_min, y_max = float(np.min(y)), float(np.max(y))
    height = y_max - y_min
    if height <= 0:
        height = 1.0
    # Shoulder proxy: x extent in upper ~35% of height (head–shoulder band)
    y_upper = y_max - 0.35 * height
    upper_mask = y >= y_upper
    if np.any(upper_mask):
        shoulder_proxy = float(np.ptp(x[upper_mask]))
    else:
        shoulder_proxy = float(np.ptp(x))
    # Thickness: z extent (전체 또는 중간 50%)
    thickness_proxy = float(np.ptp(z))
    # Arm proxy: x extent of upper 50% (넓을수록 팔이 길 수 있음에 대한 단순 proxy)
    mid_y = np.median(y)
    arm_proxy = float(np.ptp(x[y >= mid_y])) if np.any(y >= mid_y) else float(np.ptp(x)) * 0.5
    # Leg proxy: lower 50% y extent
    leg_proxy = float(np.max(y[y <= mid_y]) - np.min(y)) if np.any(y <= mid_y) else height * 0.5
    return height, shoulder_proxy, thickness_proxy, arm_proxy, leg_proxy


def compute_condition_deltas(
    xyz: np.ndarray,
    *,
    delta_height_range: tuple[float, float] = (DELTA_HEIGHT_MIN, DELTA_HEIGHT_MAX),
    delta_shoulder_range: tuple[float, float] = (DELTA_SHOULDER_MIN, DELTA_SHOULDER_MAX),
    delta_thickness_range: tuple[float, float] = (DELTA_THICKNESS_MIN, DELTA_THICKNESS_MAX),
    delta_arm_range: tuple[float, float] = (DELTA_ARM_MIN, DELTA_ARM_MAX),
    delta_leg_range: tuple[float, float] = (DELTA_LEG_MIN, DELTA_LEG_MAX),
) -> dict[str, float]:
    """
    사람 point cloud에서 sd_chibi_base 기준 상대 변화량(델타)만 계산.

    입력:
        xyz: (N, 3) point cloud (Stage 2c 또는 init_3dgs 결과).
        delta_*_range: 각 델타 허용 구간. 기본값은 상단 상수 사용.

    출력:
        JSON 직렬화 가능한 dict. 모든 값은 clamp된 델타.
        - height_delta, shoulder_width_delta, body_thickness_delta,
          arm_length_delta, leg_length_delta

    Geometry 변형·PLY 로드·Blender·얼굴 요소 없음.
    """
    height, shoulder_proxy, thickness_proxy, arm_proxy, leg_proxy = _bbox_extents(xyz)
    if height <= 0:
        height = 1.0
    # 비율로 정규화 (height 기준)
    shoulder_ratio = shoulder_proxy / height
    thickness_ratio = thickness_proxy / height
    arm_ratio = arm_proxy / height
    leg_ratio = leg_proxy / height
    # (관측 - 기준) * 감도 → clamp. height_delta = 전체 키 스케일 차이 (REF_HEIGHT 대비)
    height_delta = _clamp(
        (height - REF_HEIGHT) * SENSITIVITY_HEIGHT,
        delta_height_range[0], delta_height_range[1],
    )
    shoulder_width_delta = _clamp(
        (shoulder_ratio - REF_SHOULDER_RATIO) * SENSITIVITY_SHOULDER,
        delta_shoulder_range[0], delta_shoulder_range[1],
    )
    body_thickness_delta = _clamp(
        (thickness_ratio - REF_THICKNESS_RATIO) * SENSITIVITY_THICKNESS,
        delta_thickness_range[0], delta_thickness_range[1],
    )
    arm_length_delta = _clamp(
        (arm_ratio - REF_ARM_RATIO) * SENSITIVITY_ARM,
        delta_arm_range[0], delta_arm_range[1],
    )
    leg_length_delta = _clamp(
        (leg_ratio - REF_LEG_RATIO) * SENSITIVITY_LEG,
        delta_leg_range[0], delta_leg_range[1],
    )
    return {
        "height_delta": float(height_delta),
        "shoulder_width_delta": float(shoulder_width_delta),
        "body_thickness_delta": float(body_thickness_delta),
        "arm_length_delta": float(arm_length_delta),
        "leg_length_delta": float(leg_length_delta),
    }


def run(
    point_cloud_path: str | Path | None = None,
    xyz: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Stage 3 진입: point cloud에서 조건 dict 생성.

    point_cloud_path 또는 xyz 중 하나 필수.
    - point_cloud_path: data/gs_output/point_cloud.npy 등
    - xyz: (N, 3) 배열 직접 전달

    반환: Stage 4에서 sd_chibi_base.ply에 적용할 비율/스케일 조건 dict (JSON 직렬화 가능).
    """
    if xyz is not None:
        pts = np.asarray(xyz, dtype=np.float64)
    elif point_cloud_path is not None:
        path = Path(point_cloud_path)
        if not path.exists():
            raise FileNotFoundError(f"[deform_geometry] Not found: {path}")
        pts = np.load(path)
    else:
        raise ValueError("[deform_geometry] Provide point_cloud_path or xyz")
    return compute_condition_deltas(pts)


# -----------------------------------------------------------------------------
# 사용 예시 (이 파일을 직접 실행할 때)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    default_pc = project_root / "data" / "gs_output" / "point_cloud.npy"
    if default_pc.exists():
        out = run(point_cloud_path=default_pc)
        print("Condition deltas (for Stage 4 / sd_chibi_base):")
        print(json.dumps(out, indent=2))
    else:
        # Dummy point cloud for test
        np.random.seed(42)
        xyz = np.random.randn(5000, 3).astype(np.float32) * 0.5
        xyz[:, 1] += 1.0
        out = run(xyz=xyz)
        print("Condition deltas (dummy point cloud):")
        print(json.dumps(out, indent=2))
        print("\nTo use real data, run init_3dgs first and pass data/gs_output/point_cloud.npy", file=sys.stderr)
