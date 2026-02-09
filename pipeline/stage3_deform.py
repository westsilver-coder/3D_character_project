"""
Stage 3: 템플릿에 적용할 비율/스케일 조건 정리.

최종 캐릭터 geometry를 만드는 단계가 아님.
deform_geometry를 사용해 Stage 2 출력(point_cloud)에서 조건 dict를 계산하고
JSON으로 저장. Stage 4에서 이 조건을 템플릿에 적용한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# pipeline/ 기준이므로 상위에서 gs 로드
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gs.deform_geometry import run as deform_run


DEFAULT_POINT_CLOUD = PROJECT_ROOT / "data" / "gs_output" / "point_cloud.npy"
DEFAULT_OUT_JSON = PROJECT_ROOT / "data" / "gs_output" / "condition_deltas.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Compute condition deltas from point cloud (no geometry output).")
    parser.add_argument(
        "--point-cloud",
        type=Path,
        default=DEFAULT_POINT_CLOUD,
        help=f"Path to point cloud .npy (default: {DEFAULT_POINT_CLOUD})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help=f"Output JSON path for condition deltas (default: {DEFAULT_OUT_JSON})",
    )
    args = parser.parse_args()

    if not args.point_cloud.exists():
        print(f"Error: Point cloud not found: {args.point_cloud}", file=sys.stderr)
        print("Run Stage 2 and init_3dgs first to generate point_cloud.npy.", file=sys.stderr)
        sys.exit(1)

    condition = deform_run(point_cloud_path=args.point_cloud)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(condition, f, indent=2, ensure_ascii=False)
    print(f"Stage 3 done. Condition deltas saved to: {args.out}", flush=True)


if __name__ == "__main__":
    main()
