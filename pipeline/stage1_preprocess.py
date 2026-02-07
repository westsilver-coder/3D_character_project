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
# 테스트용: 처음 N장만 처리 (None이면 전체). 코드 상단 또는 CLI --max-images 로 지정.
MAX_IMAGES = None  # 예: 1 또는 3 으로 설정하면 해당 장수만 처리

# YOLOv8 (ultralytics). YOLO+SAM 파이프라인에서 "어디를 볼지" 결정. semantic 판단은 YOLO만 수행.
# 배경이 크고 인물이 작은 구도도 지원: YOLO가 person bbox만 찾으면 그 crop만 SAM에 넣어서 처리.
YOLO_MODEL = "yolov8n.pt"   # yolov8n.pt / yolov8s.pt (가벼운 쪽이 빠름)
YOLO_CONF_THRESHOLD = 0.25  # person 검출 최소 confidence (작은 인물이면 더 낮춰 볼 수 있음, 예: 0.2)
YOLO_PERSON_CLASS_ID = 0    # COCO "person"
YOLO_SAM_MIN_CROP_SIDE = 32  # crop 한 변이 이보다 작으면 패딩으로 확장 (SAM 입력 최소 크기)
# 옵션 A: YOLO+SAM 시 bbox 패딩 비율(10~15%), morphological closing 커널 크기 (팔/다리/치마 안쪽 구멍 제거)
YOLO_SAM_BBOX_PADDING_RATIO = 0.15   # bbox 확장 (경계 잘림 방지)
YOLO_SAM_CLOSING_KERNEL_SIZE = 11    # MORPH_CLOSE 커널 (구멍 메우기, 홀수 권장)

# rembg 사용 여부 (인물 마스크용). 설치: pip install rembg[gpu] 또는 rembg
try:
    from rembg import remove as rembg_remove  # type: ignore[import-untyped]
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False

# segment-anything (SAM). 설치: pip install segment-anything; 체크포인트는 별도 다운로드.
try:
    from segment_anything import (  # type: ignore[import-untyped]
        SamAutomaticMaskGenerator,
        SamPredictor,
        sam_model_registry,
    )
    HAS_SAM = True
except ImportError:
    HAS_SAM = False

# ultralytics YOLOv8. 설치: pip install ultralytics
try:
    from ultralytics import YOLO  # type: ignore[import-untyped]
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


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


def _detect_person_bboxes(
    img: PILImage.Image,
    yolo_model: Any,
    conf_threshold: float = YOLO_CONF_THRESHOLD,
) -> list[tuple[int, int, int, int]]:
    """
    YOLOv8으로 "person" bbox 검출. class==person, conf>=threshold.
    반환: [(x1,y1,x2,y2), ...] 픽셀 좌표, 면적 큰 순 정렬 (메인 인물 우선).
    """
    import numpy as np

    if not HAS_YOLO or yolo_model is None:
        return []
    arr = np.array(img)
    if arr.ndim != 3:
        return []
    # ultralytics: model(img) returns Results; .boxes.xyxy (tensor), .boxes.conf, .boxes.cls
    results = yolo_model(arr, verbose=False)[0]
    boxes = results.boxes
    if boxes is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    out = []
    for i in range(len(cls)):
        if int(cls[i]) != YOLO_PERSON_CLASS_ID:
            continue
        if conf[i] < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, xyxy[i])
        # 클립 to image bounds
        w, h = img.size
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        out.append((x1, y1, x2, y2))
    # 면적 큰 순 (메인 인물 먼저)
    out.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    return out


def _load_yolo_model(model_name: str | None = None) -> Any:
    """YOLOv8 모델 로드 (배치에서 한 번만). person 검출용."""
    if not HAS_YOLO:
        return None
    name = model_name or YOLO_MODEL
    try:
        model = YOLO(name)
        return model
    except Exception:
        return None


def _create_sam_mask_generator(
    checkpoint_path: Path | None = None,
    model_type: str | None = None,
) -> Any:
    """
    SAM SamAutomaticMaskGenerator 생성 (배치에서 한 번만 호출).
    GPU에 로드하여 CPU에서의 극심한 지연을 방지.
    segment-anything 미설치 또는 체크포인트 없으면 None 반환.
    """
    if not HAS_SAM:
        return None
    import torch

    path = Path(checkpoint_path or SAM_CHECKPOINT_PATH)
    if not path.is_file():
        return None
    model_type = model_type or SAM_MODEL_TYPE
    sam = sam_model_registry[model_type](checkpoint=str(path))
    sam.to("cuda")
    print("SAM device:", next(sam.parameters()).device, flush=True)
    return SamAutomaticMaskGenerator(sam)


def _create_sam_predictor(
    checkpoint_path: Path | None = None,
    model_type: str | None = None,
) -> Any:
    """
    SAM SamPredictor 생성 (--use-yolo-sam 전용).
    box prompt 방식으로 단일 인물 마스크를 얻기 위해 사용.
    AutomaticMaskGenerator 대신 Predictor를 쓰면 "bbox 안의 하나의 객체"로 통째로 세그멘트되어
    팔/다리/치마 안쪽 구멍이 생기지 않음 (옵션 A).
    """
    if not HAS_SAM:
        return None
    import torch

    path = Path(checkpoint_path or SAM_CHECKPOINT_PATH)
    if not path.is_file():
        return None
    model_type = model_type or SAM_MODEL_TYPE
    sam = sam_model_registry[model_type](checkpoint=str(path))
    sam.to("cuda")
    print("SAM device (predictor):", next(sam.parameters()).device, flush=True)
    return SamPredictor(sam)


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
    import torch

    with torch.no_grad():
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


def _apply_person_mask_yolo_sam(
    img: PILImage.Image,
    yolo_model: Any,
    sam_predictor: Any,
) -> PILImage.Image:
    """
    YOLOv8 + SAM 인물 마스크 (옵션 A: SamPredictor + box prompt + closing).

    선택 이유: AutomaticMaskGenerator는 bbox 내부를 여러 segment로 나누어
    "area 최대 1개"만 쓰면 팔/다리/치마 안쪽이 배경으로 뚫리는 구멍이 발생.
    SamPredictor에 YOLO bbox를 box prompt로 주면 "이 상자 안의 하나의 객체"로
    통째로 세그멘트하므로 인물이 한 덩어리 mask가 됨. 이어서 morphological closing으로
    미세 구멍을 제거해 COLMAP/3DGS용 solid 실루엣을 보장.
    """
    import numpy as np
    from PIL import Image

    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return img
    H, W = arr.shape[0], arr.shape[1]

    bboxes = _detect_person_bboxes(img, yolo_model)
    if not bboxes:
        return img
    x1, y1, x2, y2 = bboxes[0]
    cw, ch = x2 - x1, y2 - y1
    # bbox 패딩 (10~15%): 경계 잘림 방지, SAM box prompt에 여유 부여
    pad_ratio = YOLO_SAM_BBOX_PADDING_RATIO
    pad_w = int(cw * pad_ratio)
    pad_h = int(ch * pad_ratio)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(W, x2 + pad_w)
    y2 = min(H, y2 + pad_h)
    # 최소 crop 크기 (매우 작은 bbox 대비)
    min_side = YOLO_SAM_MIN_CROP_SIDE
    if (x2 - x1) < min_side or (y2 - y1) < min_side:
        pad_w = max(0, (min_side - (x2 - x1) + 1) // 2)
        pad_h = max(0, (min_side - (y2 - y1) + 1) // 2)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(W, x2 + pad_w)
        y2 = min(H, y2 + pad_h)

    # SamPredictor: 전체 이미지에 set_image 후 box prompt로 단일 마스크 예측
    import torch

    sam_predictor.set_image(arr)
    box_xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)
    with torch.no_grad():
        masks, _scores, _logits = sam_predictor.predict(box=box_xyxy, multimask_output=False)
    if not masks or masks.size == 0:
        return img
    mask = masks[0]  # (H, W) bool
    alpha = np.where(mask, 255, 0).astype(np.uint8)

    # Morphological closing: 내부 구멍 제거 (dilate → erode) → 인물을 하나의 solid mask로
    try:
        import cv2  # type: ignore[import-untyped]

        k = YOLO_SAM_CLOSING_KERNEL_SIZE
        if k >= 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    except ImportError:
        pass  # cv2 없으면 closing 생략

    bg = Image.new("RGB", img.size, BACKGROUND_RGB)
    rgb = img.convert("RGB")
    out = Image.composite(rgb, bg, Image.fromarray(alpha))
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
    use_yolo_sam: bool = False,
    use_sam: bool = False,
    use_person_mask: bool = USE_PERSON_MASK,
    color_normalize: bool = COLOR_NORMALIZE,
    max_images: int | None = None,
) -> list[Path]:
    """
    Stage 1 전처리 실행.
    raw_dir 이미지를 읽어 리사이즈 → (옵션) 색상 정규화 → (옵션) 인물 마스크 → processed_dir에 저장.
    마스크 우선순위: use_yolo_sam → YOLO+SAM, else use_sam → SAM만, else use_person_mask → rembg.
    max_images: None이면 전체, N이면 처음 N장만 처리 (테스트용).
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

    # 테스트용: 처리할 이미지 수 제한 (코드 상단 MAX_IMAGES 또는 인자 max_images)
    limit = max_images if max_images is not None else MAX_IMAGES
    if limit is not None and limit >= 1:
        paths = paths[: int(limit)]
        print(f"[Stage1] Limiting to first {len(paths)} image(s) (max_images={limit})", flush=True)

    yolo_model: Any = None
    sam_predictor: Any = None  # --use-yolo-sam: box prompt용 (옵션 A)
    sam_generator: Any = None  # --use-sam: automatic mask generator용

    if use_yolo_sam:
        if not HAS_YOLO:
            raise RuntimeError(
                "--use-yolo-sam requires ultralytics (YOLOv8). Install: pip install ultralytics"
            )
        if not HAS_SAM:
            raise RuntimeError(
                "--use-yolo-sam requires segment-anything. Install: pip install segment-anything"
            )
        print("[Stage1] Loading YOLO (person detection)...", flush=True)
        yolo_model = _load_yolo_model()
        if yolo_model is None:
            raise RuntimeError(f"YOLO model could not be loaded: {YOLO_MODEL}")
        print("[Stage1] Loading SAM predictor (GPU, box prompt)...", flush=True)
        sam_predictor = _create_sam_predictor()
        if sam_predictor is None:
            raise FileNotFoundError(
                f"SAM checkpoint not found at {SAM_CHECKPOINT_PATH}. "
                "Download from https://github.com/facebookresearch/segment-anything#model-checkpoints"
            )
        print("[Stage1] YOLO+SAM ready (Option A: predictor + closing).", flush=True)
    elif use_sam:
        if not HAS_SAM:
            raise RuntimeError(
                "segment-anything is not installed. Install with: pip install segment-anything\n"
                "Download SAM checkpoint and set SAM_CHECKPOINT_PATH (e.g. sam_vit_h_4b8939.pth)."
            )
        print("[Stage1] Loading SAM model (GPU)...", flush=True)
        sam_generator = _create_sam_mask_generator()
        if sam_generator is None:
            raise FileNotFoundError(
                f"SAM checkpoint not found at {SAM_CHECKPOINT_PATH}. "
                "Download from https://github.com/facebookresearch/segment-anything#model-checkpoints"
            )
        print("[Stage1] SAM ready.", flush=True)

    saved: list[Path] = []
    for i, src_path in enumerate(paths):
        print(f"[Stage1] Processing {i+1}/{len(paths)}: {src_path.name}", flush=True)
        img = Image.open(src_path).convert("RGB")
        img = _resize_max_side(img, max_size)
        if color_normalize:
            img = _color_normalize_simple(img)
        if use_yolo_sam and yolo_model is not None and sam_predictor is not None:
            img = _apply_person_mask_yolo_sam(img, yolo_model, sam_predictor)
        elif use_sam and sam_generator is not None:
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
    """CLI 진입점. 마스크: --use-yolo-sam | --use-sam | --use-rembg | 기본 rembg. --no-mask로 비활성화."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: 이미지 전처리 (리사이즈, 색보정, 인물 마스크)")
    parser.add_argument("--raw", type=Path, default=RAW_IMAGES_DIR, help="원본 이미지 디렉터리")
    parser.add_argument("--out", type=Path, default=PROCESSED_IMAGES_DIR, help="전처리 결과 저장 디렉터리")
    parser.add_argument("--max-size", type=int, default=MAX_SIZE_PX, help="긴 변 최대 픽셀")
    parser.add_argument("--no-mask", action="store_true", help="인물 마스크 비활성화")
    parser.add_argument("--use-rembg", action="store_true", help="인물 마스크: rembg 사용 (기본값)")
    parser.add_argument("--use-sam", action="store_true", help="인물 마스크: SAM 단독 (가장 큰 segment 휴리스틱)")
    parser.add_argument("--use-yolo-sam", action="store_true", help="인물 마스크: YOLOv8 bbox + SAM (복잡한 배경 권장)")
    parser.add_argument("--no-color-norm", action="store_true", help="색상 정규화 비활성화")
    parser.add_argument("--max-images", type=int, default=None, metavar="N", help="테스트용: 처음 N장만 처리 (예: 1)")
    args = parser.parse_args()

    # 우선순위: --no-mask > --use-yolo-sam > --use-sam > --use-rembg 또는 기본 rembg
    no_mask = args.no_mask
    use_yolo_sam = args.use_yolo_sam and not no_mask
    use_sam = args.use_sam and not no_mask and not use_yolo_sam
    use_rembg = (args.use_rembg or (not no_mask and not use_yolo_sam and not use_sam))

    if use_yolo_sam and (not HAS_YOLO or not HAS_SAM):
        msg = "Error: --use-yolo-sam requires ultralytics and segment-anything.\n"
        if not HAS_YOLO:
            msg += "  pip install ultralytics\n"
        if not HAS_SAM:
            msg += "  pip install segment-anything (and SAM checkpoint)\n"
        print(msg, file=sys.stderr)
        sys.exit(1)

    if use_sam and not HAS_SAM:
        print(
            "Error: --use-sam requires segment-anything.\n"
            "  pip install segment-anything\n"
            "  Download a checkpoint (e.g. sam_vit_h_4b8939.pth) and set SAM_CHECKPOINT_PATH in this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    if use_rembg and not HAS_REMBG:
        print("Warning: rembg not installed. Use --use-sam or --use-yolo-sam, or install: pip install rembg[gpu]", file=sys.stderr)
        use_rembg = False
        if not no_mask and not use_sam and not use_yolo_sam:
            print("No mask method available. Proceeding without mask.", file=sys.stderr)

    saved = run(
        raw_dir=args.raw,
        processed_dir=args.out,
        max_size=args.max_size,
        use_yolo_sam=use_yolo_sam,
        use_sam=use_sam,
        use_person_mask=use_rembg,
        color_normalize=not args.no_color_norm,
        max_images=args.max_images,
    )
    print(f"Stage 1 done. Saved {len(saved)} images to {args.out}", flush=True)


if __name__ == "__main__":
    main()
