"""
Stage 4: Stylized Rendering (Appearance-only).

Stage 2c에서 학습된 3DGS checkpoint를 입력으로 받아,
geometry는 그대로 두고 렌더링 스타일만 적용하여
"반실사 애니메이션 피규어/굿즈" 느낌의 결과를 출력한다.

- Diffuse 중심, 셀 셰이딩, 색상 양자화
- 출력: 스타일화된 이미지 (또는 턴테이블 영상)
- GPU/checkpoint 없을 때는 더미 실행으로 파이프라인 동작 확인 가능
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gs.render_character import (
    run as render_run,
    run_dummy,
    DEFAULT_CEL_BANDS,
    DEFAULT_COLOR_LEVELS,
    DEFAULT_SATURATION,
    DEFAULT_CONTRAST,
    OUTPUT_DIR as DEFAULT_OUTPUT_DIR,
)
from gs.train_3dgs import CHECKPOINT_DIR as DEFAULT_CHECKPOINT_DIR
from gs.train_3dgs import GS_OUTPUT_DIR as DEFAULT_GS_OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: 3DGS checkpoint → stylized render (cell shading + color quantization)."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="3DGS checkpoint directory (from Stage 2c)",
    )
    parser.add_argument(
        "--gs-output",
        type=Path,
        default=DEFAULT_GS_OUTPUT_DIR,
        help="data/gs_output (cameras.json for views)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for stylized images",
    )
    parser.add_argument(
        "--cel-bands",
        type=int,
        default=DEFAULT_CEL_BANDS,
        help="Cell shading bands (2~4, default %d)" % DEFAULT_CEL_BANDS,
    )
    parser.add_argument(
        "--color-levels",
        type=int,
        default=DEFAULT_COLOR_LEVELS,
        help="Color quantization levels (default %d)" % DEFAULT_COLOR_LEVELS,
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=DEFAULT_SATURATION,
        help="Saturation (default %.2f)" % DEFAULT_SATURATION,
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=DEFAULT_CONTRAST,
        help="Contrast (default %.2f)" % DEFAULT_CONTRAST,
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=8,
        help="Max number of camera views to render",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Dummy run: no checkpoint, only apply style to placeholder (Colab GPU 없을 때 파이프라인 확인용)",
    )
    parser.add_argument(
        "--no-dummy",
        action="store_true",
        help="Fail if no GPU or checkpoint (do not fall back to dummy)",
    )
    args = parser.parse_args()

    if args.dummy:
        paths = run_dummy(
            args.output_dir,
            cel_bands=args.cel_bands,
            color_levels=args.color_levels,
        )
        print("Stage 4 (dummy) done. Stylized placeholder saved:", paths, flush=True)
        return

    try:
        paths = render_run(
            checkpoint_dir=args.checkpoint_dir,
            gs_output_dir=args.gs_output,
            output_dir=args.output_dir,
            cel_bands=args.cel_bands,
            color_levels=args.color_levels,
            saturation=args.saturation,
            contrast=args.contrast,
            num_views=args.num_views,
            dummy_if_no_gpu=not args.no_dummy,
        )
        print("Stage 4 done. Stylized render output:", paths, flush=True)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        if not args.no_dummy:
            print("Tip: Run with --dummy to test the pipeline without a checkpoint.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
