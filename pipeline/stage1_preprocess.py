"""
Stage 1: Human-prior / 3D Gaussian Splatting 입력용 이미지 전처리.

목표: "multi-view world 정합성" — 모든 프레임에서
- 동일한 crop 영역 (global person bbox union)
- 동일한 사람 scale (person height = constant px)
- 사람 중심 정렬 (이미지 중앙)
- 배경 최소화, 인물 잘림 없음

→ 3DGS geometry collapse 방지.
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

# -----------------------------------------------------------------------------
# 파라미터
# -----------------------------------------------------------------------------
RAW_IMAGES_DIR = PROJECT_ROOT / "data" / "raw_images"
PROCESSED_IMAGES_DIR = PROJECT_ROOT / "data" / "processed_images"
MAX_SIZE_PX = 1024
BACKGROUND_RGB = (128, 128, 128)
COLOR_NORMALIZE = True
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MAX_IMAGES = None

# World-consistency: 동일 crop/scale/center
TARGET_PERSON_HEIGHT_PX = 800
OUT_CANVAS_WIDTH = 1024
OUT_CANVAS_HEIGHT = 1024
GLOBAL_CROP_MARGIN = 0.10  # union bbox 주변 10% 여유
BORDER_TOUCH_MARGIN_PX = 8  # bbox가 이미지 경계 이 안이면 discard (잘림)

# 마스크: rembg + yolo union, 검증
REMBG_FG_RATIO_MIN = 0.12
REMBG_BBOX_OVERLAP_MIN = 0.25
YOLO_SEG_MODEL = "yolov8n-seg.pt"
YOLO_CONF_THRESHOLD = 0.25
YOLO_PERSON_CLASS_ID = 0
YOLO_SEG_CLOSING_KERNEL_SIZE = 11
MASK_FG_RATIO_MIN = 0.18  # 최종 마스크 fg 비율 하한
MASK_BBOX_OVERLAP_MIN = 0.22  # bbox 내 fg 비율 하한

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


# -----------------------------------------------------------------------------
# YOLO: person bbox (and optional seg mask)
# -----------------------------------------------------------------------------
def _get_person_bboxes_yolo(img: "PILImage.Image", yolo_model: Any) -> list[tuple[int, int, int, int]]:
    """YOLO로 person(class 0) bbox 목록 반환. 픽셀 좌표 (x1,y1,x2,y2)."""
    import numpy as np
    if not HAS_YOLO or yolo_model is None:
        return []
    arr = np.array(img)
    if arr.ndim != 3:
        return []
    results = yolo_model(arr, verbose=False)[0]
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


def _bbox_touches_border(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    margin: int = BORDER_TOUCH_MARGIN_PX,
) -> bool:
    """bbox가 이미지 경계에 닿으면 True (인물 잘림)."""
    x1, y1, x2, y2 = bbox
    return (
        x1 < margin or y1 < margin
        or x2 > width - margin or y2 > height - margin
    )


def _union_bboxes(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    """여러 bbox의 union (x1,y1,x2,y2)."""
    if not bboxes:
        return None
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return (x1, y1, x2, y2)


def _normalized_bbox(bbox: tuple[int, int, int, int], w: int, h: int) -> tuple[float, float, float, float]:
    """(x1,y1,x2,y2) → (cx_norm, cy_norm, w_norm, h_norm)."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / max(w, 1)
    cy = (y1 + y2) / 2.0 / max(h, 1)
    bw = (x2 - x1) / max(w, 1)
    bh = (y2 - y1) / max(h, 1)
    return (cx, cy, bw, bh)


# -----------------------------------------------------------------------------
# Pass 1: Global person bbox union (normalized)
# -----------------------------------------------------------------------------
def _compute_global_crop_normalized(
    image_paths: list[Path],
    yolo_model: Any,
    max_side: int,
    margin: float = GLOBAL_CROP_MARGIN,
) -> tuple[float, float, float, float] | None:
    """
    모든 이미지에서 YOLO person bbox 추출 → normalized union + margin.
    반환: (left_norm, top_norm, right_norm, bottom_norm) in [0,1], or None if no person anywhere.
    """
    from PIL import Image
    import numpy as np
    all_left, all_right = [], []
    all_top, all_bottom = [], []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        img = _resize_max_side(img, max_side)
        w, h = img.size
        bboxes = _get_person_bboxes_yolo(img, yolo_model)
        if not bboxes:
            continue
        union = _union_bboxes(bboxes)
        if union is None:
            continue
        x1, y1, x2, y2 = union
        cx_n = (x1 + x2) / 2.0 / max(w, 1)
        cy_n = (y1 + y2) / 2.0 / max(h, 1)
        bw_n = (x2 - x1) / max(w, 1)
        bh_n = (y2 - y1) / max(h, 1)
        all_left.append(cx_n - bw_n / 2)
        all_right.append(cx_n + bw_n / 2)
        all_top.append(cy_n - bh_n / 2)
        all_bottom.append(cy_n + bh_n / 2)
    if not all_left:
        return None
    left = max(0.0, min(all_left) - margin)
    right = min(1.0, max(all_right) + margin)
    top = max(0.0, min(all_top) - margin)
    bottom = min(1.0, max(all_bottom) + margin)
    if left >= right or top >= bottom:
        return None
    return (left, top, right, bottom)


# -----------------------------------------------------------------------------
# Mask: rembg + yolo union, closing
# -----------------------------------------------------------------------------
def _rembg_alpha(img: "PILImage.Image") -> Any:
    """rembg alpha 채널만 (H,W) numpy uint8 0/255. 실패 시 None."""
    import numpy as np
    if not HAS_REMBG:
        return None
    from PIL import Image
    rgba = rembg_remove(img, alpha_matting=False)
    if rgba.mode != "RGBA":
        return None
    alpha = rgba.split()[-1]
    if alpha.size != img.size:
        alpha = alpha.resize(img.size, resample=Image.NEAREST)
    return np.array(alpha)


def _yolo_seg_mask(img: "PILImage.Image", yolo_model: Any) -> Any:
    """YOLO person segmentation mask (H,W) uint8 0/255. 없으면 None."""
    import numpy as np
    if not HAS_YOLO or yolo_model is None:
        return None
    arr = np.array(img)
    if arr.ndim != 3:
        return None
    H, W = arr.shape[0], arr.shape[1]
    results = yolo_model(arr, retina_masks=True, verbose=False)[0]
    if results.masks is None or results.boxes is None:
        return None
    boxes = results.boxes
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    person_idx = [
        i for i in range(len(cls))
        if int(cls[i]) == YOLO_PERSON_CLASS_ID and float(conf[i]) >= YOLO_CONF_THRESHOLD
    ]
    if not person_idx:
        return None
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
        return None
    return (combined * 255).astype(np.uint8)


def _mask_union_closing(rembg_alpha: Any, yolo_mask: Any, close_kernel: int = YOLO_SEG_CLOSING_KERNEL_SIZE) -> Any:
    """rembg + yolo union 후 closing. (H,W) uint8 0/255."""
    import numpy as np
    if rembg_alpha is None and yolo_mask is None:
        return None
    H, W = 0, 0
    out = None
    if rembg_alpha is not None:
        out = (rembg_alpha > 0).astype(np.uint8) * 255
        H, W = out.shape[0], out.shape[1]
    if yolo_mask is not None:
        if out is None:
            out = (yolo_mask > 0).astype(np.uint8) * 255
            H, W = out.shape[0], out.shape[1]
        else:
            out = np.maximum(out, (yolo_mask > 0).astype(np.uint8) * 255)
    if out is None:
        return None
    try:
        import cv2
        if close_kernel >= 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    except ImportError:
        pass
    return out


def _validate_mask_strict(
    alpha: Any,
    bbox_crop: tuple[int, int, int, int],
    crop_w: int,
    crop_h: int,
) -> bool:
    """fg_ratio 및 bbox 내 overlap 검증."""
    import numpy as np
    if alpha is None or alpha.size == 0:
        return False
    arr = np.asarray(alpha)
    if arr.ndim != 2:
        return False
    fg = (arr > 0).sum()
    total = arr.size
    if total <= 0:
        return False
    if fg / total < MASK_FG_RATIO_MIN:
        return False
    x1, y1, x2, y2 = bbox_crop
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(crop_w, x2), min(crop_h, y2)
    if x2 <= x1 or y2 <= y1:
        return False
    patch = arr[y1:y2, x1:x2]
    bbox_area = (x2 - x1) * (y2 - y1)
    overlap = (patch > 0).sum() / max(bbox_area, 1)
    return overlap >= MASK_BBOX_OVERLAP_MIN


# -----------------------------------------------------------------------------
# Pass 2: crop → scale (person height) → center on canvas
# -----------------------------------------------------------------------------
def _crop_normalized(
    img: "PILImage.Image",
    left_norm: float,
    top_norm: float,
    right_norm: float,
    bottom_norm: float,
) -> "PILImage.Image":
    """이미지에서 normalized 영역만 crop. 반환 크기는 이미지마다 다를 수 있음."""
    from PIL import Image
    w, h = img.size
    x1 = int(round(left_norm * w))
    y1 = int(round(top_norm * h))
    x2 = int(round(right_norm * w))
    y2 = int(round(bottom_norm * h))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def _center_on_canvas(
    img: "PILImage.Image",
    person_center_x: float,
    person_center_y: float,
    out_w: int,
    out_h: int,
    bg_rgb: tuple[int, int, int] = BACKGROUND_RGB,
) -> "PILImage.Image":
    """
    img 내 (person_center_x, person_center_y)가 출력 중앙에 오도록 함.
    - img가 canvas보다 크면: person 중심 기준 (out_w, out_h) 영역 crop.
    - img가 canvas보다 작으면: canvas에 배경 채운 뒤 img를 중앙에 paste.
    """
    from PIL import Image
    iw, ih = img.size
    if iw >= out_w and ih >= out_h:
        # Crop: center at (person_center_x, person_center_y), size (out_w, out_h)
        x0 = int(round(person_center_x - out_w / 2))
        y0 = int(round(person_center_y - out_h / 2))
        x0 = max(0, min(x0, iw - out_w))
        y0 = max(0, min(y0, ih - out_h))
        return img.crop((x0, y0, x0 + out_w, y0 + out_h))
    canvas = Image.new("RGB", (out_w, out_h), bg_rgb)
    px = int(round(out_w / 2 - person_center_x))
    py = int(round(out_h / 2 - person_center_y))
    canvas.paste(img, (px, py))
    return canvas


def run(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    max_size: int = MAX_SIZE_PX,
    mask_mode: str = "auto",
    color_normalize: bool = COLOR_NORMALIZE,
    max_images: int | None = None,
    strict_mask: bool = False,
    target_person_height: int = TARGET_PERSON_HEIGHT_PX,
    out_canvas_size: tuple[int, int] = (OUT_CANVAS_WIDTH, OUT_CANVAS_HEIGHT),
) -> list[Path]:
    """
    Two-pass:
    - Pass 1: 모든 이미지에서 YOLO person bbox → global normalized crop (union + margin).
    - Pass 2: 각 이미지에 동일 crop 적용 → person height로 scale → 중심 정렬 → 마스크(rembg+yolo union) → 검증 → 저장 or skip.

    mask_mode: auto = rembg+yolo union + 검증. rembg = rembg만. yolo = yolo만. none = 마스크 없음(정규화만).
    """
    import numpy as np
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

    need_yolo = mask_mode in ("auto", "yolo") or True  # global bbox always needs YOLO
    if not HAS_YOLO:
        raise RuntimeError("Stage 1 world-consistency requires ultralytics (YOLO). pip install ultralytics")
    yolo_model = YOLO(YOLO_SEG_MODEL)
    print("[Stage1] YOLOv8-seg loaded.", flush=True)
    if mask_mode in ("auto", "rembg") and not HAS_REMBG:
        raise RuntimeError("mask_mode auto/rembg requires rembg. pip install rembg[gpu]")

    # ----- Pass 1: global crop (normalized) -----
    print("[Stage1] Pass 1: computing global person bbox union...", flush=True)
    global_crop = _compute_global_crop_normalized(paths, yolo_model, max_size, margin=GLOBAL_CROP_MARGIN)
    if global_crop is None:
        raise ValueError(
            "[Stage1] No person detected in any image. Cannot compute global crop. "
            "Check raw_images and YOLO."
        )
    left_n, top_n, right_n, bottom_n = global_crop
    print(f"[Stage1] Global crop (normalized): left={left_n:.3f} top={top_n:.3f} right={right_n:.3f} bottom={bottom_n:.3f}", flush=True)

    out_w, out_h = out_canvas_size
    saved: list[Path] = []
    skipped_border = 0
    skipped_no_person = 0
    skipped_mask = 0

    for i, src_path in enumerate(paths):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[Stage1] Pass 2: {i+1}/{len(paths)} {src_path.name}", flush=True)
        try:
            img = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"[Stage1] Skip (open failed): {src_path.name} ({e})", flush=True)
            skipped_mask += 1
            continue
        img = _resize_max_side(img, max_size)
        w, h = img.size

        bboxes = _get_person_bboxes_yolo(img, yolo_model)
        if not bboxes:
            skipped_no_person += 1
            continue
        union = _union_bboxes(bboxes)
        if union is None:
            skipped_no_person += 1
            continue
        if _bbox_touches_border(union, w, h, BORDER_TOUCH_MARGIN_PX):
            skipped_border += 1
            continue

        # 동일 crop
        crop_img = _crop_normalized(img, left_n, top_n, right_n, bottom_n)
        cw, ch = crop_img.size
        if cw < 10 or ch < 10:
            skipped_mask += 1
            continue

        # Crop 좌표계에서 person bbox
        x1, y1, x2, y2 = union
        crop_x1 = int(round(left_n * w))
        crop_y1 = int(round(top_n * h))
        bbox_in_crop = (
            x1 - crop_x1, y1 - crop_y1,
            x2 - crop_x1, y2 - crop_y1,
        )
        bx1, by1, bx2, by2 = bbox_in_crop
        person_height_crop = max(1, by2 - by1)
        person_center_x_crop = (bx1 + bx2) / 2.0
        person_center_y_crop = (by1 + by2) / 2.0

        # Scale: person height = target_person_height
        scale = target_person_height / person_height_crop
        new_cw = max(1, int(round(cw * scale)))
        new_ch = max(1, int(round(ch * scale)))
        crop_img = crop_img.resize((new_cw, new_ch), resample=Image.LANCZOS)
        # bbox와 center를 scaled 좌표로
        person_center_x_scaled = person_center_x_crop * scale
        person_center_y_scaled = person_center_y_crop * scale
        bx1_s = int(bx1 * scale)
        by1_s = int(by1 * scale)
        bx2_s = int(bx2 * scale)
        by2_s = int(by2 * scale)
        bbox_crop_scaled = (bx1_s, by1_s, bx2_s, by2_s)

        # Mask: rembg and/or yolo (union when both), on scaled crop
        if mask_mode != "none":
            if mask_mode == "rembg":
                rembg_a, yolo_m = _rembg_alpha(crop_img), None
            elif mask_mode == "yolo":
                rembg_a, yolo_m = None, _yolo_seg_mask(crop_img, yolo_model)
            else:
                rembg_a, yolo_m = _rembg_alpha(crop_img), _yolo_seg_mask(crop_img, yolo_model)
            alpha = _mask_union_closing(rembg_a, yolo_m)
            if alpha is not None:
                if color_normalize:
                    crop_img = _color_normalize_simple(crop_img)
                bg = Image.new("RGB", crop_img.size, BACKGROUND_RGB)
                crop_img = Image.composite(
                    crop_img.convert("RGB") if crop_img.mode != "RGB" else crop_img,
                    bg,
                    Image.fromarray(alpha),
                )
                if not _validate_mask_strict(alpha, bbox_crop_scaled, new_cw, new_ch):
                    skipped_mask += 1
                    continue
            else:
                if mask_mode == "auto":
                    skipped_mask += 1
                    continue
        else:
            if color_normalize:
                crop_img = _color_normalize_simple(crop_img)

        # Center on canvas (person at image center)
        canvas = _center_on_canvas(
            crop_img,
            person_center_x_scaled,
            person_center_y_scaled,
            out_w,
            out_h,
            BACKGROUND_RGB,
        )

        out_path = processed_dir / f"{len(saved):04d}_{src_path.stem}.png"
        canvas.save(out_path, "PNG")
        saved.append(out_path)

    if skipped_border or skipped_no_person or skipped_mask:
        print(
            f"[Stage1] Skipped: border_touch={skipped_border}, no_person={skipped_no_person}, mask/invalid={skipped_mask}",
            flush=True,
        )
    print(f"[Stage1] Saved {len(saved)} images (world-consistent: same crop, scale, center).", flush=True)
    return saved


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Stage 1: Human/3DGS 입력용 전처리 (global bbox, same crop, scale+center normalization)"
    )
    parser.add_argument("--raw", type=Path, default=RAW_IMAGES_DIR, help="원본 이미지 디렉터리")
    parser.add_argument("--out", type=Path, default=PROCESSED_IMAGES_DIR, help="출력 디렉터리")
    parser.add_argument("--max-size", type=int, default=MAX_SIZE_PX, help="Pass 1/2 리사이즈 시 긴 변 최대 픽셀")
    parser.add_argument(
        "--mask-mode",
        choices=("auto", "rembg", "yolo", "none"),
        default="auto",
        help="auto: rembg+yolo union+검증. rembg/yolo: 해당만. none: 마스크 없음(정규화만)",
    )
    parser.add_argument("--strict-mask", action="store_true", help="(reserved) stricter fg ratio")
    parser.add_argument("--no-mask", action="store_true", help="마스크 비활성화 (mask-mode=none)")
    parser.add_argument("--no-color-norm", action="store_true", help="색상 정규화 비활성화")
    parser.add_argument("--max-images", type=int, default=None, metavar="N", help="처음 N장만 처리")
    parser.add_argument("--person-height", type=int, default=TARGET_PERSON_HEIGHT_PX, help="정규화 목표 사람 높이(px)")
    parser.add_argument("--canvas-size", type=int, nargs=2, default=[OUT_CANVAS_WIDTH, OUT_CANVAS_HEIGHT], metavar=("W", "H"), help="출력 캔버스 크기")
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
        target_person_height=args.person_height,
        out_canvas_size=tuple(args.canvas_size),
    )
    print(f"Stage 1 done. Saved {len(saved)} images to {args.out}", flush=True)


if __name__ == "__main__":
    main()
