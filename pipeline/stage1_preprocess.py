"""
Stage 1: Human-prior / 3DGS 입력용 이미지 전처리.

- 입력 이미지 그대로 비율 유지: max_side 기준 리사이즈만 (가로/세로 바꾸지 않음).
- 배경 제거: rembg 또는 YOLO-seg. 마스크 검증 실패 시 해당 프레임 스킵.
- crop/회전/letterbox 없음 → 입력 방향 그대로 출력.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image as PILImage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_IMAGES_DIR = PROJECT_ROOT / "data" / "raw_images"
PROCESSED_IMAGES_DIR = PROJECT_ROOT / "data" / "processed_images"
MAX_SIZE_PX = 1024
BACKGROUND_RGB = (128, 128, 128)
COLOR_NORMALIZE = True
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MAX_IMAGES = None

# rembg 마스크 검증
REMBG_FG_RATIO_MIN = 0.10
REMBG_BBOX_OVERLAP_MIN = 0.20

# YOLOv8-seg fallback
YOLO_SEG_MODEL = "yolov8n-seg.pt"
YOLO_CONF_THRESHOLD = 0.25
YOLO_PERSON_CLASS_ID = 0
YOLO_SEG_CLOSING_KERNEL_SIZE = 11
YOLO_FG_RATIO_MIN = 0.15
YOLO_FG_RATIO_MIN_STRICT = 0.25

try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


def _list_image_paths(dir_path: Path) -> list[Path]:
    out = []
    for ext in SUPPORTED_EXTENSIONS:
        out.extend(dir_path.glob(f"*{ext}"))
    return sorted(out, key=lambda p: p.name.lower())


def _resize_max_side(img: "PILImage.Image", max_side: int) -> "PILImage.Image":
    """긴 변을 max_side로. 비율 유지, 가로/세로 그대로."""
    from PIL import Image
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), resample=Image.LANCZOS)


def _color_normalize_simple(img: "PILImage.Image") -> "PILImage.Image":
    import numpy as np
    from PIL import Image
    arr = np.array(img)
    if arr.ndim == 2:
        lo, hi = np.percentile(arr, (2, 98))
        if hi > lo:
            arr = np.clip((arr.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    out = arr.copy()
    for c in range(min(3, arr.shape[-1])):
        ch = arr[..., c]
        lo, hi = np.percentile(ch, (2, 98))
        if hi > lo:
            out[..., c] = np.clip((ch.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _apply_rembg_mask(img: "PILImage.Image") -> tuple["PILImage.Image", Any]:
    """rembg로 전경 마스크(alpha) 추출, 원본 크기 기준 합성. 반환: (합성 RGB, alpha numpy H,W 0/255)."""
    from PIL import Image
    if not HAS_REMBG:
        return (img, None)
    rgba = rembg_remove(img, alpha_matting=False)
    if rgba.mode != "RGBA":
        return (img, None)
    alpha = rgba.split()[-1]
    if alpha.size != img.size:
        alpha = alpha.resize(img.size, resample=Image.NEAREST)
    import numpy as np
    alpha_np = np.array(alpha)
    bg = Image.new("RGB", img.size, BACKGROUND_RGB)
    out = Image.composite(img.convert("RGB") if img.mode != "RGB" else img, bg, alpha)
    return (out, alpha_np)


def _get_person_bboxes_from_seg(img: "PILImage.Image", yolo_seg_model: Any) -> list[tuple[int, int, int, int]]:
    """YOLO person bbox (x1,y1,x2,y2) for validation."""
    import numpy as np
    if not HAS_YOLO or yolo_seg_model is None:
        return []
    arr = np.array(img)
    if arr.ndim != 3:
        return []
    results = yolo_seg_model(arr, verbose=False)[0]
    if results.boxes is None:
        return []
    boxes = results.boxes
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    w, h = img.size
    out = []
    for i in range(len(cls)):
        if int(cls[i]) != YOLO_PERSON_CLASS_ID or conf[i] < YOLO_CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = int(xyxy[i, 0]), int(xyxy[i, 1]), int(xyxy[i, 2]), int(xyxy[i, 3])
        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
        out.append((x1, y1, x2, y2))
    return out


def _validate_mask(alpha: Any, img: "PILImage.Image", yolo_seg_model: Any) -> bool:
    """rembg 마스크 품질: fg 비율, bbox 내 overlap."""
    import numpy as np
    if alpha is None or alpha.size == 0:
        return False
    H, W = alpha.shape[0], alpha.shape[1]
    fg_area = (alpha > 0).sum()
    if H * W <= 0:
        return False
    if fg_area / (H * W) < REMBG_FG_RATIO_MIN:
        return False
    bboxes = _get_person_bboxes_from_seg(img, yolo_seg_model)
    if not bboxes:
        return True
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    bbox_area = (x2 - x1) * (y2 - y1)
    if bbox_area <= 0:
        return True
    crop = alpha[y1:y2, x1:x2]
    if (crop > 0).sum() / bbox_area < REMBG_BBOX_OVERLAP_MIN:
        return False
    return True


def _apply_yolo_seg_mask(
    img: "PILImage.Image",
    yolo_seg_model: Any,
    strict: bool = False,
) -> tuple["PILImage.Image", bool]:
    """YOLO person seg 마스크, closing. 반환 (합성 이미지, 검증 통과 여부)."""
    import numpy as np
    from PIL import Image
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return (img, False)
    H, W = arr.shape[0], arr.shape[1]
    if not HAS_YOLO or yolo_seg_model is None:
        return (img, False)
    results = yolo_seg_model(arr, retina_masks=True, verbose=False)[0]
    if results.masks is None or results.boxes is None:
        return (img, False)
    boxes = results.boxes
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    person_idx = [
        i for i in range(len(cls))
        if int(cls[i]) == YOLO_PERSON_CLASS_ID and float(conf[i]) >= YOLO_CONF_THRESHOLD
    ]
    if not person_idx:
        return (img, False)
    combined = np.zeros((H, W), dtype=np.uint8)
    for i in person_idx:
        m = results.masks.data[i]
        if hasattr(m, "cpu"):
            m = m.cpu().numpy()
        m = np.squeeze(m)
        if m.ndim != 2:
            continue
        if m.shape[0] != H or m.shape[1] != W:
            from PIL import Image as P
            pil_m = P.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
            pil_m = pil_m.resize((W, H), resample=P.NEAREST)
            m = (np.array(pil_m) > 127).astype(np.uint8)
        else:
            m = (m > 0.5).astype(np.uint8) if m.dtype != np.uint8 else (m > 0).astype(np.uint8)
        combined = np.maximum(combined, m)
    if combined.max() == 0:
        return (img, False)
    alpha = (combined * 255).astype(np.uint8)
    xyxy = boxes.xyxy.cpu().numpy()
    person_boxes = xyxy[person_idx]
    x1 = int(max(0, person_boxes[:, 0].min()))
    y1 = int(max(0, person_boxes[:, 1].min()))
    x2 = int(min(W, person_boxes[:, 2].max()))
    y2 = int(min(H, person_boxes[:, 3].max()))
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    fg_ratio = (alpha > 0).sum() / bbox_area
    threshold = YOLO_FG_RATIO_MIN_STRICT if strict else YOLO_FG_RATIO_MIN
    if fg_ratio < threshold:
        return (img, False)
    try:
        import cv2
        k = YOLO_SEG_CLOSING_KERNEL_SIZE
        if k >= 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    except ImportError:
        pass
    bg = Image.new("RGB", img.size, BACKGROUND_RGB)
    out = Image.composite(img.convert("RGB"), bg, Image.fromarray(alpha))
    return (out, True)


def run(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    max_size: int = MAX_SIZE_PX,
    mask_mode: str = "auto",
    color_normalize: bool = COLOR_NORMALIZE,
    max_images: int | None = None,
    strict_mask: bool = False,
) -> list[Path]:
    """
    이미지별: max_side 기준 리사이즈(비율 유지), 선택적 색보정, 배경 제거 후 저장.
    mask_mode: auto = rembg → 검증 실패 시 yolo fallback → 실패 시 스킵.
               rembg = rembg만. yolo = YOLO만. none = 마스크 없음.
    """
    from PIL import Image
    raw_dir = Path(raw_dir or RAW_IMAGES_DIR)
    processed_dir = Path(processed_dir or PROCESSED_IMAGES_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths = _list_image_paths(raw_dir)
    if not paths:
        raise FileNotFoundError(f"No images in {raw_dir} (ext: {SUPPORTED_EXTENSIONS})")
    limit = max_images if max_images is not None else MAX_IMAGES
    if limit is not None and limit >= 1:
        paths = paths[: int(limit)]
        print(f"[Stage1] Limiting to first {len(paths)} image(s)", flush=True)

    yolo_seg_model: Any = None
    if mask_mode in ("auto", "yolo"):
        if not HAS_YOLO:
            raise RuntimeError("mask_mode auto/yolo requires ultralytics. pip install ultralytics")
        yolo_seg_model = YOLO(YOLO_SEG_MODEL)
        print("[Stage1] YOLOv8-seg loaded.", flush=True)
    if mask_mode in ("auto", "rembg") and not HAS_REMBG:
        raise RuntimeError("mask_mode auto/rembg requires rembg. pip install rembg[gpu]")

    saved: list[Path] = []
    skipped = 0
    for i, src_path in enumerate(paths):
        print(f"[Stage1] {i+1}/{len(paths)}: {src_path.name}", flush=True)
        img = Image.open(src_path).convert("RGB")
        img = _resize_max_side(img, max_size)
        if color_normalize:
            img = _color_normalize_simple(img)
        if mask_mode == "none":
            out_path = processed_dir / f"{len(saved):04d}_{src_path.stem}.png"
            img.save(out_path, "PNG")
            saved.append(out_path)
            continue
        if mask_mode == "rembg":
            img, _ = _apply_rembg_mask(img)
            out_path = processed_dir / f"{len(saved):04d}_{src_path.stem}.png"
            img.save(out_path, "PNG")
            saved.append(out_path)
            continue
        if mask_mode == "yolo":
            img, valid = _apply_yolo_seg_mask(img, yolo_seg_model, strict=strict_mask)
            if not valid:
                print(f"[Stage1] Mask validation failed, skipping: {src_path.name}", flush=True)
                skipped += 1
                continue
            out_path = processed_dir / f"{len(saved):04d}_{src_path.stem}.png"
            img.save(out_path, "PNG")
            saved.append(out_path)
            continue
        assert mask_mode == "auto"
        img_rembg, alpha = _apply_rembg_mask(img)
        if alpha is not None and _validate_mask(alpha, img_rembg, yolo_seg_model):
            out_path = processed_dir / f"{len(saved):04d}_{src_path.stem}.png"
            img_rembg.save(out_path, "PNG")
            saved.append(out_path)
            continue
        img_yolo, valid = _apply_yolo_seg_mask(img, yolo_seg_model, strict=strict_mask)
        if not valid:
            print(f"[Stage1] rembg invalid + yolo fallback failed, skipping: {src_path.name}", flush=True)
            skipped += 1
            continue
        out_path = processed_dir / f"{len(saved):04d}_{src_path.stem}.png"
        img_yolo.save(out_path, "PNG")
        saved.append(out_path)
    if skipped > 0:
        print(f"[Stage1] Skipped {skipped} frame(s).", flush=True)
    print(f"[Stage1] Saved {len(saved)} images (aspect preserved, no crop/rotation).", flush=True)
    return saved


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1: Human/3DGS 입력용 전처리 (리사이즈+배경제거, 비율·방향 유지)")
    parser.add_argument("--raw", type=Path, default=RAW_IMAGES_DIR, help="원본 이미지 디렉터리")
    parser.add_argument("--out", type=Path, default=PROCESSED_IMAGES_DIR, help="출력 디렉터리")
    parser.add_argument("--max-size", type=int, default=MAX_SIZE_PX, help="긴 변 최대 픽셀 (비율 유지)")
    parser.add_argument(
        "--mask-mode",
        choices=("auto", "rembg", "yolo", "none"),
        default="auto",
        help="auto: rembg→검증 실패 시 yolo. rembg/yolo/none",
    )
    parser.add_argument("--strict-mask", action="store_true", help="yolo fg_ratio 0.25로 강화")
    parser.add_argument("--no-mask", action="store_true", help="마스크 비활성화")
    parser.add_argument("--no-color-norm", action="store_true", help="색상 정규화 비활성화")
    parser.add_argument("--max-images", type=int, default=None, metavar="N", help="처음 N장만 처리")
    args = parser.parse_args()
    mask_mode = "none" if args.no_mask else args.mask_mode
    if args.no_mask:
        print("[Stage1] --no-mask: mask disabled.", flush=True)
    saved = run(
        raw_dir=args.raw,
        processed_dir=args.out,
        max_size=args.max_size,
        mask_mode=mask_mode,
        color_normalize=not args.no_color_norm,
        max_images=args.max_images,
        strict_mask=args.strict_mask,
    )
    print(f"Stage 1 done. Saved {len(saved)} images to {args.out}", flush=True)


if __name__ == "__main__":
    main()
