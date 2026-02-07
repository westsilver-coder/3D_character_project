"""
Stage 1: 입력 이미지 전처리 (Computer Vision).
리사이즈(최대 1024px), 색상 정규화, 인물 마스크 생성 → 3D 복원용 전처리 이미지 세트.

COLMAP/3DGS 관점: 전경(인물) 실루엣이 안정적이어야 카메라 추정·포인트 정합이
안정된다. "예쁜 컷아웃"보다 foreground 마스크 일관성이 중요.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image as PILImage

# 프로젝트 루트를 path에 추가 (Colab/로컬 공통)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# 파라미터 (코드 상단에서 조절)
# -----------------------------------------------------------------------------
RAW_IMAGES_DIR = PROJECT_ROOT / "data" / "raw_images"
PROCESSED_IMAGES_DIR = PROJECT_ROOT / "data" / "processed_images"
MAX_SIZE_PX = 1024
BACKGROUND_RGB = (128, 128, 128)  # 마스크 외 배경 색 (단순 배경)
USE_PERSON_MASK = True             # 인물 마스크 적용 여부 (False면 리사이즈+색보정만)
COLOR_NORMALIZE = True             # 색상 정규화 (이미지 간 밝기/대비 맞춤)
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# SAM (Segment Anything) 옵션. prompt-free 배치 처리용.
SAM_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"  # vit_h / vit_l / vit_b (체크포인트와 일치해야 함)
SAM_ALPHA_THRESHOLD = 170   # 알파 threshold: 경계 노이즈 억제 (0~255)
SAM_USE_EROSION = True       # 경계 halo 감소를 위한 erosion (MinFilter)

# rembg 사용 여부 (인물 마스크용). 설치: pip install rembg[gpu] 또는 rembg
try:
    from rembg import remove as rembg_remove  # type: ignore[import-untyped]
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

# segment-anything (SAM). 설치: pip install segment-anything; 체크포인트는 별도 다운로드.
try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # type: ignore[import-untyped]
    HAS_SAM = True
except ImportError:
    HAS_SAM = False


def _list_image_paths(dir_path: Path) -> list[Path]:
    """지원 확장자의 이미지 파일 목록 (이름 순)."""
    out = []
    for ext in SUPPORTED_EXTENSIONS:
        out.extend(dir_path.glob(f"*{ext}"))
    return sorted(out, key=lambda p: p.name.lower())


def _resize_max_side(img: PILImage.Image, max_side: int) -> PILImage.Image:
    """긴 변이 max_side를 넘지 않도록 비율 유지 리사이즈."""
    from PIL import Image

    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), resample=Image.LANCZOS)


def _color_normalize_simple(img: PILImage.Image) -> PILImage.Image:
    """이미지 한 장에 대한 단순 색상 정규화 (밝기/대비 스트레칭)."""
    import numpy as np
    from PIL import Image

    arr = np.array(img)
    if arr.ndim == 2:
        lo, hi = np.percentile(arr, (2, 98))
        if hi > lo:
            arr = np.clip((arr.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    # RGB/RGBA
    out = arr.copy()
    for c in range(min(3, arr.shape[-1])):
        ch = arr[..., c]
        lo, hi = np.percentile(ch, (2, 98))
        if hi > lo:
            out[..., c] = np.clip((ch.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _create_sam_mask_generator(
    checkpoint_path: Path | None = None,
    model_type: str | None = None,
) -> Any:
    """
    SAM SamAutomaticMaskGenerator 생성 (배치에서 한 번만 호출).
    segment-anything 미설치 또는 체크포인트 없으면 None 반환.
    """
    if not HAS_SAM:
        return None
    path = Path(checkpoint_path or SAM_CHECKPOINT_PATH)
    if not path.is_file():
        return None
    model_type = model_type or SAM_MODEL_TYPE
    sam = sam_model_registry[model_type](checkpoint=str(path))
    return SamAutomaticMaskGenerator(sam)


def _apply_person_mask_sam(
    img: PILImage.Image,
    mask_generator: Any,
) -> PILImage.Image:
    """
    SAM 기반 인물 마스크: 배경을 BACKGROUND_RGB로 교체.
    prompt-free (SamAutomaticMaskGenerator). area 최대 마스크를 전경으로 사용.
    경계 노이즈 억제: 알파 threshold + optional erosion → COLMAP/3DGS 실루엣 안정성.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return img
    masks = mask_generator.generate(arr)
    if not masks:
        return img
    # area가 가장 큰 마스크를 "사람"으로 가정 (전신 촬영 시 인물이 가장 큰 영역)
    best = max(masks, key=lambda m: m["area"])
    seg = best["segmentation"]  # bool HxW
    alpha = np.where(seg, 255, 0).astype(np.uint8)
    # 경계 노이즈 억제: threshold (float 마스크일 때 대비; SAM은 bool이라 0/255 유지)
    if SAM_ALPHA_THRESHOLD > 0 and alpha.dtype == np.uint8:
        alpha = np.where(alpha >= SAM_ALPHA_THRESHOLD, 255, 0).astype(np.uint8)
    # erosion으로 경계 halo 감소 (COLMAP 매칭 시 배경 잔여 감소)
    if SAM_USE_EROSION:
        pil_alpha = Image.fromarray(alpha)
        pil_alpha = pil_alpha.filter(ImageFilter.MinFilter(size=3))
        alpha = np.array(pil_alpha)
    bg = Image.new("RGB", img.size, BACKGROUND_RGB)
    rgb = img.convert("RGB")
    alpha_pil = Image.fromarray(alpha)
    out = Image.composite(rgb, bg, alpha_pil)
    return out


def _apply_person_mask(img: PILImage.Image) -> PILImage.Image:
    """인물 마스크 적용 (rembg): 배경을 BACKGROUND_RGB로 교체. rembg 없으면 원본 반환."""
    from PIL import Image

    if not HAS_REMBG:
        return img
    # rembg: RGBA 반환 (배경 투명)
    rgba = rembg_remove(img, alpha_matting=False)
    if rgba.mode != "RGBA":
        return img
    rgb = rgba.convert("RGB")
    alpha = rgba.split()[-1]
    bg = Image.new("RGB", rgb.size, BACKGROUND_RGB)
    out = Image.composite(rgb, bg, alpha)
    return out


def run(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    max_size: int = MAX_SIZE_PX,
    use_sam: bool = False,
    use_person_mask: bool = USE_PERSON_MASK,
    color_normalize: bool = COLOR_NORMALIZE,
) -> list[Path]:
    """
    Stage 1 전처리 실행.
    raw_dir 이미지를 읽어 리사이즈 → (옵션) 색상 정규화 → (옵션) 인물 마스크 → processed_dir에 저장.
    마스크 우선순위: use_sam → SAM, else use_person_mask → rembg, else 마스크 없음.
    저장 파일 경로 목록 반환.
    """
    from PIL import Image

    raw_dir = raw_dir or RAW_IMAGES_DIR
    processed_dir = processed_dir or PROCESSED_IMAGES_DIR

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    paths = _list_image_paths(raw_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {raw_dir} (extensions: {SUPPORTED_EXTENSIONS})")

    # SAM 사용 시 배치용 generator 한 번만 생성 (COLMAP/3DGS 일관된 전경 실루엣)
    sam_generator: Any = None
    if use_sam:
        if not HAS_SAM:
            raise RuntimeError(
                "segment-anything is not installed. Install with: pip install segment-anything\n"
                "Download SAM checkpoint and set SAM_CHECKPOINT_PATH (e.g. sam_vit_h_4b8939.pth)."
            )
        sam_generator = _create_sam_mask_generator()
        if sam_generator is None:
            raise FileNotFoundError(
                f"SAM checkpoint not found at {SAM_CHECKPOINT_PATH}. "
                "Download from https://github.com/facebookresearch/segment-anything#model-checkpoints"
            )

    saved: list[Path] = []
    for i, src_path in enumerate(paths):
        img = Image.open(src_path).convert("RGB")
        img = _resize_max_side(img, max_size)
        if color_normalize:
            img = _color_normalize_simple(img)
        if use_sam and sam_generator is not None:
            img = _apply_person_mask_sam(img, sam_generator)
        elif use_person_mask:
            img = _apply_person_mask(img)
        # 3DGS 호환용 번호 붙인 파일명도 유지 (0000.png, 0001.png, ...)
        base = src_path.stem
        out_name = f"{i:04d}_{base}.png"
        out_path = processed_dir / out_name
        img.save(out_path, "PNG")
        saved.append(out_path)
    return saved


def main() -> None:
    """CLI 진입점. --use-sam → SAM, --no-mask → 마스크 없음, 기본 → rembg."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: 이미지 전처리 (리사이즈, 색보정, 인물 마스크)")
    parser.add_argument("--raw", type=Path, default=RAW_IMAGES_DIR, help="원본 이미지 디렉터리")
    parser.add_argument("--out", type=Path, default=PROCESSED_IMAGES_DIR, help="전처리 결과 저장 디렉터리")
    parser.add_argument("--max-size", type=int, default=MAX_SIZE_PX, help="긴 변 최대 픽셀")
    parser.add_argument("--no-mask", action="store_true", help="인물 마스크 비활성화 (SAM/rembg 모두 미사용)")
    parser.add_argument("--use-sam", action="store_true", help="인물 마스크에 SAM(Segment Anything) 사용 (rembg 대신)")
    parser.add_argument("--no-color-norm", action="store_true", help="색상 정규화 비활성화")
    args = parser.parse_args()

    # 마스크: --no-mask면 없음, --use-sam이면 SAM, 아니면 rembg (rembg 미설치 시 경고)
    use_sam = args.use_sam and not args.no_mask
    use_person_mask = not args.no_mask and not use_sam

    if use_sam and not HAS_SAM:
        print(
            "Error: --use-sam requires segment-anything.\n"
            "  pip install segment-anything\n"
            "  Download a checkpoint (e.g. sam_vit_h_4b8939.pth) and set SAM_CHECKPOINT_PATH in this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    if use_person_mask and not HAS_REMBG:
        print("Warning: rembg not installed. Run without person mask, or use --use-sam. Install: pip install rembg[gpu]")
        use_person_mask = False

    saved = run(
        raw_dir=args.raw,
        processed_dir=args.out,
        max_size=args.max_size,
        use_sam=use_sam,
        use_person_mask=use_person_mask,
        color_normalize=not args.no_color_norm,
    )
    print(f"Stage 1 done. Saved {len(saved)} images to {args.out}")


if __name__ == "__main__":
    main()
