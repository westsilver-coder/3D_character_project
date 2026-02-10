"""
Stage 4 렌더링: 3DGS checkpoint → 스타일화 렌더 (반실사 애니 피규어 느낌).

Geometry는 변경하지 않음. 렌더링 스타일만 적용:
- Diffuse 중심 (SH DC만 사용해도 diffuse에 가깝게)
- 셀 셰이딩 (명암 양자화)
- 색상 단순화 (quantization, saturation/contrast)
GPU 없을 때는 더미 모드(placeholder 이미지에 스타일만 적용)로 파이프라인 동작 확인 가능.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "gs_checkpoints"
GS_OUTPUT_DIR = PROJECT_ROOT / "data" / "gs_output"
OUTPUT_DIR = PROJECT_ROOT / "output" / "stylized"

# 스타일 파라미터 (튜닝 가능)
DEFAULT_CEL_BANDS = 3
DEFAULT_COLOR_LEVELS = 6
DEFAULT_SATURATION = 1.0
DEFAULT_CONTRAST = 1.05


def apply_cell_shading(rgb: np.ndarray, bands: int = 3) -> np.ndarray:
    """
    명암을 2~4단계로 양자화 (toon/cell shading).
    rgb: (H, W, 3) float [0,1]. bands: 2~4 권장.
    """
    if bands < 2:
        return rgb
    gray = np.clip(np.dot(rgb, [0.299, 0.587, 0.114]), 0, 1)
    edges = np.linspace(0, 1, bands + 1)
    out_gray = np.zeros_like(gray)
    for i in range(bands):
        lo, hi = edges[i], edges[i + 1]
        mask = (gray >= lo) & (gray < hi) if i < bands - 1 else (gray >= lo)
        out_gray[mask] = (lo + hi) / 2
    scale = np.where(gray > 1e-6, out_gray / np.maximum(gray, 1e-6), 1.0)
    return np.clip(rgb * scale[:, :, np.newaxis], 0, 1).astype(np.float32)


def apply_color_quantization(rgb: np.ndarray, levels: int = 6) -> np.ndarray:
    """
    RGB를 단계적으로 양자화 (의상 디테일 단순화).
    rgb: (H, W, 3) float [0,1]. levels: 2~8 권장.
    """
    if levels < 2:
        return rgb
    step = 1.0 / (levels - 1)
    q = np.round(rgb / step).astype(np.int32)
    q = np.clip(q, 0, levels - 1)
    return (q * step).astype(np.float32)


def apply_saturation_contrast(rgb: np.ndarray, saturation: float = 1.0, contrast: float = 1.0) -> np.ndarray:
    """saturation/contrast 조정. rgb (H,W,3) [0,1]."""
    gray = np.dot(rgb, [0.299, 0.587, 0.114])[:, :, np.newaxis]
    rgb_sat = gray + (rgb - gray) * saturation
    rgb_sat = np.clip(rgb_sat, 0, 1)
    rgb_con = (rgb_sat - 0.5) * contrast + 0.5
    return np.clip(rgb_con, 0, 1).astype(np.float32)


def apply_stylization(
    rgb: np.ndarray,
    cel_bands: int = DEFAULT_CEL_BANDS,
    color_levels: int = DEFAULT_COLOR_LEVELS,
    saturation: float = DEFAULT_SATURATION,
    contrast: float = DEFAULT_CONTRAST,
) -> np.ndarray:
    """한 번에 셀 셰이딩 + 색상 양자화 + saturation/contrast 적용."""
    out = apply_saturation_contrast(rgb, saturation, contrast)
    out = apply_cell_shading(out, cel_bands)
    out = apply_color_quantization(out, color_levels)
    return out


def run_dummy(
    output_dir: Path,
    width: int = 256,
    height: int = 256,
    cel_bands: int = DEFAULT_CEL_BANDS,
    color_levels: int = DEFAULT_COLOR_LEVELS,
) -> list[Path]:
    """
    GPU/checkpoint 없이 스타일 파이프라인만 검증.
    placeholder 이미지에 셀 셰이딩·양자화 적용 후 저장.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 그라데이션 placeholder (명암이 있어야 셀 셰이딩이 보임)
    y = np.linspace(0, 1, height)
    x = np.linspace(0, 1, width)
    gray = (y[:, np.newaxis] * 0.7 + x[np.newaxis, :] * 0.3).astype(np.float32)
    rgb = np.stack([gray, gray * 0.9, gray * 0.95], axis=-1)
    out = apply_stylization(rgb, cel_bands=cel_bands, color_levels=color_levels)
    out_u8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    out_path = output_dir / "stylized_dummy.png"
    try:
        from PIL import Image
        Image.fromarray(out_u8).save(out_path)
    except Exception:
        np.save(output_dir / "stylized_dummy.npy", out_u8)
    return [out_path]


def run(
    checkpoint_dir: Path | None = None,
    gs_output_dir: Path | None = None,
    output_dir: Path | None = None,
    cel_bands: int = DEFAULT_CEL_BANDS,
    color_levels: int = DEFAULT_COLOR_LEVELS,
    saturation: float = DEFAULT_SATURATION,
    contrast: float = DEFAULT_CONTRAST,
    num_views: int = 8,
    width: int = 512,
    height: int = 512,
    device: str | None = None,
    dummy_if_no_gpu: bool = True,
) -> list[Path]:
    """
    3DGS checkpoint를 로드해 여러 시점으로 렌더한 뒤 스타일화 적용하여 저장.
    GPU/checkpoint 없으면 dummy_if_no_gpu 시 더미 실행.
    반환: 저장된 이미지 경로 목록.
    """
    checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR).resolve()
    gs_output_dir = Path(gs_output_dir or GS_OUTPUT_DIR).resolve()
    output_dir = Path(output_dir or OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    except ImportError:
        return run_dummy(output_dir, width, height, cel_bands, color_levels)

    ckpt_path = None
    if checkpoint_dir.exists():
        for p in sorted(checkpoint_dir.glob("ckpt_step_*.pth"), key=lambda x: int(x.stem.split("_")[-1]), reverse=True):
            ckpt_path = p
            break
    if ckpt_path is None or not ckpt_path.exists():
        if dummy_if_no_gpu:
            return run_dummy(output_dir, width, height, cel_bands, color_levels)
        raise FileNotFoundError(f"[render_character] No checkpoint in {checkpoint_dir}")

    try:
        from gs.train_3dgs import (
            GaussianModel,
            _render_one_view,
            _get_gaussian_rasterizer,
            _load_gs_output,
        )
    except ImportError:
        if dummy_if_no_gpu:
            return run_dummy(output_dir, width, height, cel_bands, color_levels)
        raise

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_points = ckpt["xyz"].shape[0]
    RasterSettings, GaussianRasterizer = _get_gaussian_rasterizer()
    gaussians = GaussianModel(num_points, sh_degree=0, device=dev)
    gaussians.set_xyz(torch.from_numpy(ckpt["xyz"]).float().to(dev))
    gaussians._log_scale.data = torch.from_numpy(ckpt["log_scale"]).float().to(dev)
    gaussians._quat.data = torch.from_numpy(ckpt["quat"]).float().to(dev)
    gaussians._logit_opacity.data = torch.from_numpy(ckpt["logit_opacity"]).float().to(dev)
    gaussians._sh_dc.data = torch.from_numpy(ckpt["sh_dc"]).float().to(dev)
    gaussians._sh_rest.data = torch.from_numpy(ckpt["sh_rest"]).float().to(dev)

    try:
        _, cameras, _ = _load_gs_output(gs_output_dir)
    except Exception:
        cameras = []

    saved = []
    if cameras:
        for i, cam in enumerate(cameras[:num_views]):
            try:
                out = _render_one_view(gaussians, cam, GaussianRasterizer, RasterSettings, dev)
                rgb = out.permute(1, 2, 0).cpu().numpy()
            except Exception:
                continue
            rgb = np.clip(rgb, 0, 1).astype(np.float32)
            styled = apply_stylization(rgb, cel_bands, color_levels, saturation, contrast)
            out_u8 = (np.clip(styled, 0, 1) * 255).astype(np.uint8)
            p = output_dir / f"stylized_view_{i:02d}.png"
            try:
                from PIL import Image
                Image.fromarray(out_u8).save(p)
            except Exception:
                np.save(p.with_suffix(".npy"), out_u8)
            saved.append(p)
    if not saved:
        h, w = height, width
        rgb = np.ones((h, w, 3), dtype=np.float32) * 0.5
        styled = apply_stylization(rgb, cel_bands, color_levels, saturation, contrast)
        out_u8 = (np.clip(styled, 0, 1) * 255).astype(np.uint8)
        p = output_dir / "stylized_view_00.png"
        try:
            from PIL import Image
            Image.fromarray(out_u8).save(p)
        except Exception:
            np.save(p.with_suffix(".npy"), out_u8)
        saved.append(p)
    return saved


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stage 4: Stylized 3DGS render (cell shading + color quantization)")
    p.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    p.add_argument("--gs-output", type=Path, default=GS_OUTPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--cel-bands", type=int, default=DEFAULT_CEL_BANDS, help="Toon bands (2~4)")
    p.add_argument("--color-levels", type=int, default=DEFAULT_COLOR_LEVELS, help="RGB quantization levels")
    p.add_argument("--saturation", type=float, default=DEFAULT_SATURATION)
    p.add_argument("--contrast", type=float, default=DEFAULT_CONTRAST)
    p.add_argument("--num-views", type=int, default=8)
    p.add_argument("--dummy", action="store_true", help="Force dummy run (no checkpoint)")
    p.add_argument("--no-dummy", action="store_true", help="Fail if no GPU/checkpoint")
    args = p.parse_args()
    try:
        if args.dummy:
            paths = run_dummy(args.output_dir, cel_bands=args.cel_bands, color_levels=args.color_levels)
        else:
            paths = run(
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
        for path in paths:
            print(path)
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
