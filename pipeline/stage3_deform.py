"""
Stage 3 (폐기): 비율/스케일 조정 단계.

프로젝트 방향 변경에 따라 geometry 변형 단계는 사용하지 않습니다.
- 실사 형상을 유지하고 appearance(렌더 스타일)만 Stage 4에서 변경합니다.
- 이 스크립트는 더미로 유지되며, 실행 시 안내만 출력하고 종료합니다.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 (deprecated): No geometry deformation.")
    parser.add_argument("--point-cloud", type=str, default="", help="Ignored")
    parser.add_argument("--out", type=str, default="", help="Ignored")
    args = parser.parse_args()
    print(
        "[Stage 3] Deprecated. Geometry is no longer modified; "
        "Stage 4 applies stylized rendering only (cell shading, color quantization).",
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
