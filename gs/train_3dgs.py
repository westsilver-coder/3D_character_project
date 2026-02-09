"""
Stage 2c: 3D Gaussian Splatting 학습.

init_3dgs.py 출력을 입력으로 받아 실제 3DGS 학습 수행.
- 입력: data/gs_output/point_cloud.npy, cameras.json, data/processed_images/ (GT)
- 기능: Gaussian 파라미터 학습, diff-gaussian-splatting 스타일 training loop
- COLMAP 미사용. world space = canonical human body space.
- 출력: 학습된 3DGS checkpoint (data/gs_output/ 또는 지정 경로)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GS_OUTPUT_DIR = PROJECT_ROOT / "data" / "gs_output"
PROCESSED_IMAGES_DIR = PROJECT_ROOT / "data" / "processed_images"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "gs_checkpoints"

# Training
MAX_STEPS = 30_000
SAVE_EVERY = 5_000
LR = 0.0025
SH_DEGREE = 0  # DC only
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _load_gs_output(data_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    """Load point_cloud.npy, cameras.json, image_list.txt from gs_output."""
    data_dir = Path(data_dir)
    pc_path = data_dir / "point_cloud.npy"
    cam_path = data_dir / "cameras.json"
    list_path = data_dir / "image_list.txt"
    if not pc_path.exists():
        raise FileNotFoundError(f"[train_3dgs] Not found: {pc_path}. Run gs/init_3dgs.py first.")
    if not cam_path.exists():
        raise FileNotFoundError(f"[train_3dgs] Not found: {cam_path}. Run gs/init_3dgs.py first.")
    points = np.load(pc_path)
    with open(cam_path, "r", encoding="utf-8") as f:
        cameras = json.load(f)["cameras"]
    if list_path.exists():
        with open(list_path, "r", encoding="utf-8") as f:
            image_list = [line.strip() for line in f if line.strip()]
    else:
        image_list = [c["image"] for c in cameras]
    return points, cameras, image_list


def _camera_to_view_proj(cam: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """K, R, t (world-to-camera) → viewmatrix (4x4), projmatrix (4x4) for rasterizer."""
    K = np.array(cam["K"], dtype=np.float32)
    R = np.array(cam["R"], dtype=np.float32)
    t = np.array(cam["t"], dtype=np.float32)
    w, h = int(cam["width"]), int(cam["height"])
    # view: world to camera (4x4)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3, 3] = t
    # proj: perspective from K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    znear, zfar = 0.01, 100.0
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 2 * fx / w
    proj[1, 1] = 2 * fy / h
    proj[0, 2] = 2 * (cx / w) - 1
    proj[1, 2] = 2 * (cy / h) - 1
    proj[2, 2] = -(zfar + znear) / (zfar - znear)
    proj[2, 3] = -(2 * zfar * znear) / (zfar - znear)
    proj[3, 2] = -1
    view_t = torch.from_numpy(view).to(device)
    proj_t = torch.from_numpy(proj).to(device)
    return view_t, proj_t


def _load_gt_image(image_dir: Path, name: str) -> torch.Tensor:
    """Load image as tensor (C,H,W) in [0,1]."""
    for ext in IMAGE_EXTENSIONS:
        p = image_dir / (Path(name).stem + ext)
        if p.exists():
            from PIL import Image
            img = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            return torch.from_numpy(img).permute(2, 0, 1)
    raise FileNotFoundError(f"[train_3dgs] GT image not found: {name} in {image_dir}")


def _get_gaussian_rasterizer():
    """Import diff-gaussian-splatting rasterizer. Raises if not installed."""
    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        return GaussianRasterizationSettings, GaussianRasterizer
    except ImportError:
        try:
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
            return GaussianRasterizationSettings, GaussianRasterizer
        except ImportError:
            raise RuntimeError(
                "[train_3dgs] Differentiable Gaussian rasterizer not found. "
                "Install from: https://github.com/graphdeco-inria/diff-gaussian-splatting "
                "(build diff-gaussian-rasterization submodule) or use a compatible package."
            ) from None


class GaussianModel(nn.Module):
    """Learnable 3D Gaussians: xyz, scale (log), rotation (quat), opacity (logit), SH."""

    def __init__(self, num_points: int, sh_degree: int = 0, device: torch.device = None):
        super().__init__()
        self.num_points = num_points
        self.sh_degree = sh_degree
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # (N, 3)
        self._xyz = nn.Parameter(torch.zeros(num_points, 3, device=self.device))
        # (N, 3) log scale
        self._log_scale = nn.Parameter(torch.zeros(num_points, 3, device=self.device))
        # (N, 4) quaternion, identity
        self._quat = nn.Parameter(torch.zeros(num_points, 4, device=self.device))
        self._quat.data[:, 0] = 1.0
        # (N, 1) logit opacity
        self._logit_opacity = nn.Parameter(torch.zeros(num_points, 1, device=self.device))
        # SH: (1 + 3) for degree 0 (DC), then 5, 7, ... up to sh_degree
        sh_dim = (self.sh_degree + 1) ** 2
        self._sh_dc = nn.Parameter(torch.zeros(num_points, 1, 3, device=self.device))
        self._sh_rest = nn.Parameter(torch.zeros(num_points, sh_dim - 1, 3, device=self.device))

    def set_xyz(self, xyz: torch.Tensor) -> None:
        self._xyz.data = xyz.to(self.device, dtype=torch.float32)

    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    def get_scales(self) -> torch.Tensor:
        return torch.exp(self._log_scale).clamp(min=1e-6)

    def get_rotations(self) -> torch.Tensor:
        q = self._quat / (self._quat.norm(dim=1, keepdim=True) + 1e-8)
        return q

    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._logit_opacity)

    def get_sh_dc(self) -> torch.Tensor:
        return self._sh_dc

    def get_sh_rest(self) -> torch.Tensor:
        return self._sh_rest


def _render_one_view(
    gaussians: GaussianModel,
    cam: dict[str, Any],
    rasterizer_fn,
    RasterSettings,
    device: torch.device,
) -> torch.Tensor:
    """Render one view; returns (3, H, W) RGB in [0,1]."""
    viewmatrix, projmatrix = _camera_to_view_proj(cam, device)
    w, h = int(cam["width"]), int(cam["height"])
    xyz = gaussians.get_xyz()
    scales = gaussians.get_scales()
    quats = gaussians.get_rotations()
    opacity = gaussians.get_opacity()
    sh_dc = gaussians.get_sh_dc()
    sh_rest = gaussians.get_sh_rest()

    # means2D / cov3D: rasterizer may recompute from means3D; pass zeros if API requires
    means2D = torch.zeros_like(xyz, device=device)
    cov3D = torch.zeros((xyz.shape[0], 6), device=device)
    shs = torch.cat([sh_dc, sh_rest], dim=1)

    K = np.array(cam["K"])
    tanfovx = (1.0 / w) * 2 * K[0, 0]
    tanfovy = (1.0 / h) * 2 * K[1, 1]
    campos = (-viewmatrix[:3, :3].T @ viewmatrix[:3, 3]).unsqueeze(0)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    # API follows graphdeco-inria/diff-gaussian-rasterization; adapt if using another fork
    settings = RasterSettings(
        image_height=h,
        image_width=w,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewmatrix.unsqueeze(0),
        projmatrix=projmatrix.unsqueeze(0),
        sh_degree=gaussians.sh_degree,
        campos=campos,
        prefiltered=False,
        debug=False,
    )
    rasterizer = rasterizer_fn(settings)
    out, _, _, _, _, _ = rasterizer(
        means3D=xyz.unsqueeze(0),
        means2D=means2D.unsqueeze(0),
        shs=shs.unsqueeze(0),
        colors_precomp=None,
        opacities=opacity.unsqueeze(0),
        scales=scales.unsqueeze(0),
        rotations=quats.unsqueeze(0),
        cov3Ds_precomp=cov3D.unsqueeze(0),
    )
    return out.squeeze(0)


def run(
    gs_output_dir: Path | None = None,
    image_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    max_steps: int = MAX_STEPS,
    save_every: int = SAVE_EVERY,
    lr: float = LR,
    device: str | None = None,
) -> Path:
    """
    init_3dgs 출력으로 3DGS 학습. COLMAP 미사용. world = canonical body space.
    """
    gs_output_dir = Path(gs_output_dir or GS_OUTPUT_DIR).resolve()
    image_dir = Path(image_dir or PROCESSED_IMAGES_DIR).resolve()
    checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR).resolve()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    points, cameras, image_list = _load_gs_output(gs_output_dir)
    if not image_list or not cameras:
        raise ValueError("[train_3dgs] No cameras or image list in gs_output.")
    RasterSettings, GaussianRasterizer = _get_gaussian_rasterizer()

    num_points = len(points)
    gaussians = GaussianModel(num_points, sh_degree=SH_DEGREE, device=device)
    gaussians.set_xyz(torch.from_numpy(points).float().to(device))
    # Small initial scale
    gaussians._log_scale.data = torch.log(torch.full_like(gaussians._log_scale, 0.01))
    gaussians._logit_opacity.data = torch.logit(torch.full((num_points, 1), 0.9, device=device))

    optimizer = torch.optim.Adam(
        [
            {"params": [gaussians._xyz], "lr": lr * 0.01},
            {"params": [gaussians._log_scale], "lr": lr},
            {"params": [gaussians._quat], "lr": lr * 0.001},
            {"params": [gaussians._logit_opacity], "lr": lr * 0.05},
            {"params": [gaussians._sh_dc, gaussians._sh_rest], "lr": lr * 0.001},
        ]
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    n_views = len(cameras)
    print(f"[train_3dgs] Training: {num_points} Gaussians, {n_views} views, device={device}", flush=True)

    for step in range(max_steps):
        optimizer.zero_grad()
        idx = step % n_views
        cam = cameras[idx]
        gt = _load_gt_image(image_dir, image_list[idx]).to(device)
        try:
            out = _render_one_view(gaussians, cam, GaussianRasterizer, RasterSettings, device)
        except Exception as e:
            print(f"[train_3dgs] Render failed at step {step}: {e}", flush=True)
            raise
        # Resize if needed
        if out.shape[1] != gt.shape[1] or out.shape[2] != gt.shape[2]:
            out = torch.nn.functional.interpolate(
                out.unsqueeze(0), size=(gt.shape[1], gt.shape[2]), mode="bilinear", align_corners=False
            ).squeeze(0)
        loss = (out - gt).abs().mean()
        loss.backward()
        optimizer.step()
        if (step + 1) % 500 == 0:
            print(f"[train_3dgs] step {step + 1}/{max_steps} loss={loss.item():.6f}", flush=True)
        if (step + 1) % save_every == 0 or step == max_steps - 1:
            ckpt = {
                "step": step + 1,
                "xyz": gaussians.get_xyz().detach().cpu().numpy(),
                "log_scale": gaussians._log_scale.detach().cpu().numpy(),
                "quat": gaussians.get_rotations().detach().cpu().numpy(),
                "logit_opacity": gaussians._logit_opacity.detach().cpu().numpy(),
                "sh_dc": gaussians.get_sh_dc().detach().cpu().numpy(),
                "sh_rest": gaussians.get_sh_rest().detach().cpu().numpy(),
            }
            path = checkpoint_dir / f"ckpt_step_{step + 1}.pth"
            torch.save(ckpt, path)
            print(f"[train_3dgs] Saved {path}", flush=True)

    print("[train_3dgs] Training done.", flush=True)
    return checkpoint_dir


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2c: 3DGS training (init_3dgs output → checkpoint)")
    parser.add_argument("--gs-output", type=Path, default=GS_OUTPUT_DIR, help="data/gs_output from init_3dgs")
    parser.add_argument("--images", type=Path, default=PROCESSED_IMAGES_DIR, help="GT images dir")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR, help="Where to save checkpoints")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    try:
        run(
            gs_output_dir=args.gs_output,
            image_dir=args.images,
            checkpoint_dir=args.checkpoint_dir,
            max_steps=args.max_steps,
            save_every=args.save_every,
            lr=args.lr,
            device=args.device,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
