"""
Stage 2c: 3D Gaussian Splatting 학습.

init_3dgs.py 출력을 입력으로 받아 실제 3DGS 학습 수행.
- 입력: data/gs_output/point_cloud.npy, cameras.json, data/processed_images/ (GT)
- 기능: Gaussian 파라미터 학습, diff-gaussian-splatting 스타일 training loop
- COLMAP 미사용. world space = canonical body space.
- 카메라 R,t도 학습(Option A). 초기값은 cameras.json.
- 출력: 학습된 3DGS checkpoint (+ 학습된 cameras)
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

# Initialization (critical for gradient flow and visible color)
# - Scale too small (e.g. 0.01) → splats invisible → no color gradient
# - sh_dc = 0 → black → no signal to learn from
INIT_SCALE = 0.12  # debug: 0.1~0.2 so splats clearly visible; 0.06 is minimal
INIT_OPACITY = 0.9  # high so contributions are non-zero
# DC color init: from GT mean (scene-specific) or neutral gray if no images
INIT_DC_FROM_GT_N_IMAGES = 8  # average mean RGB of first N GT images for sh_dc init
INIT_DC_FALLBACK = 0.5  # neutral gray when not using GT
# Per-point color init: sample first view GT at projected points (breaks gray symmetry)
INIT_DC_FROM_VIEW = True  # if True, init each Gaussian from first-view projection when valid

# Debug: save intermediate renders during training to verify Gaussians are visible
DEBUG_SAVE_EVERY = 500  # save render every N steps to checkpoint_dir/debug_renders/

# Learnable cameras (Option A): stability
CAMERA_LEARN = True
LR_CAMERA_RATIO = 0.08  # camera_lr = lr * LR_CAMERA_RATIO (e.g. 0.0025*0.08=2e-4). Kept smaller than Gaussian so pose refines gently.
CAMERA_FREEZE_STEPS = 1500  # first N steps: camera grads zeroed (Gaussian only). Then unfreeze for joint opt.
# Regularization (soft penalties; balance so learning is not blocked)
MIN_CAM_DIST = 1.0  # camera center must stay this far from origin (world scale ~ mesh height 1.0)
REG_DIST_WEIGHT = 0.1  # penalty when dist < MIN_CAM_DIST
REG_TZ_POS_WEIGHT = 1.0  # penalty for t_z > 0 (camera behind scene)
REG_L2_WEIGHT = 1e-5  # L2 toward initial pose (avoid drift)
# Logging
CAMERA_LOG_EVERY = 500  # log camera stats every N steps


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


def _get_projection_matrix(znear: float, zfar: float, fov_x_rad: float, fov_y_rad: float) -> np.ndarray:
    """
    graphdeco-inria style OpenGL projection matrix (see gaussian-splatting utils/graphics_utils.py).
    Transforms camera view space to NDC. Use with fovX = 2*atan(w/(2*fx)), fovY = 2*atan(h/(2*fy)).
    """
    tan_half_fov_y = np.tan(fov_y_rad / 2.0)
    tan_half_fov_x = np.tan(fov_x_rad / 2.0)
    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right
    P = np.zeros((4, 4), dtype=np.float32)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def _camera_to_view_proj(cam: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    K, R, t (world-to-camera) → viewmatrix (4x4), projmatrix (4x4) for diff-gaussian-rasterization.
    Matches graphdeco-inria convention: view = [R|t; 0 0 0 1], then .transpose(0,1) for C++ column-major.
    R,t from stage2: p_cam = R @ p_world + t.
    """
    K = np.array(cam["K"], dtype=np.float32)
    R = np.array(cam["R"], dtype=np.float32)
    t = np.array(cam["t"], dtype=np.float32)
    w, h = int(cam["width"]), int(cam["height"])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    znear, zfar = 0.01, 100.0

    # view: world-to-camera 4x4, [R|t; 0 0 0 1]
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3, 3] = t

    # diff-gaussian-rasterization C++ uses +Z as forward (t.x/t.z, t.y/t.z; in_frustum culls z<=0).
    # Stage2 cameras use OpenGL-style -Z forward (our z_cam is negative for points in front).
    # Flip view Z row so that "in front" in our convention becomes positive z for the rasterizer.
    view[2, :] = -view[2, :]

    # FOV from intrinsics (graphdeco focal2fov: fov = 2*atan(pixels/(2*focal)))
    fov_x_rad = 2.0 * np.arctan(w / (2.0 * fx))
    fov_y_rad = 2.0 * np.arctan(h / (2.0 * fy))
    proj = _get_projection_matrix(znear, zfar, fov_x_rad, fov_y_rad)

    # C++ auxiliary.h: transformPoint4x3 reads matrix column-major (row = matrix[0], matrix[4], matrix[8]).
    # Our view is row-major, so C++ effectively applies view^T. Pass view.T and proj.T so they apply view and proj.
    view_t = torch.from_numpy(view).float().to(device).transpose(0, 1)
    proj_t = torch.from_numpy(proj).float().to(device).transpose(0, 1)
    return view_t, proj_t


def _rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    6D continuous rotation (Zhou et al.) → 3x3 R via Gram–Schmidt. Differentiable, no singularities.
    d6: (6,) or (..., 6). Out: (3,3) or (..., 3, 3). L2 reg + ortho logging keep it stable; quaternion is an alternative if needed.
    """
    need_squeeze = d6.dim() == 1
    if need_squeeze:
        d6 = d6.unsqueeze(0)
    a, b = d6[..., :3], d6[..., 3:6]
    e1 = a / (a.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    b_orth = b - (e1 * b).sum(dim=-1, keepdim=True) * e1
    e2 = b_orth / (b_orth.norm(dim=-1, keepdim=True).clamp(min=1e-8))
    e3 = torch.linalg.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-2)
    if need_squeeze:
        R = R.squeeze(0)
    return R


def _matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """3x3 R → first two columns as 6D (for init)."""
    R = np.asarray(R, dtype=np.float32)
    return R[:, :2].flatten()


class LearnableCameras(nn.Module):
    """Per-view learnable R (6D) and t. K, w, h from fixed cam dict. Holds initial values for L2 reg."""
    def __init__(self, cameras: list[dict[str, Any]], device: torch.device):
        super().__init__()
        self.cameras = cameras
        self.device = device
        n = len(cameras)
        d6_list = []
        t_list = []
        for c in cameras:
            R = np.array(c["R"], dtype=np.float32)
            t = np.array(c["t"], dtype=np.float32)
            d6_list.append(_matrix_to_6d(R))
            t_list.append(t)
        d6_np = np.stack(d6_list)
        t_np = np.stack(t_list)
        self._cam_6d = nn.Parameter(torch.from_numpy(d6_np).float().to(device))
        self._cam_t = nn.Parameter(torch.from_numpy(t_np).float().to(device))
        self.register_buffer("_cam_6d_init", torch.from_numpy(d6_np).float().to(device))
        self.register_buffer("_cam_t_init", torch.from_numpy(t_np).float().to(device))

    def get_view_proj_and_campos(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """view (4x4), proj (4x4), campos (1,3) for rasterizer. view/proj column-major for C++."""
        cam = self.cameras[idx]
        d6 = self._cam_6d[idx]
        t = self._cam_t[idx]
        R = _rotation_6d_to_matrix(d6)
        view = torch.eye(4, device=self.device, dtype=torch.float32)
        view[:3, :3] = R
        view[:3, 3] = t
        view[2, :] = -view[2, :]
        view_t = view.T.unsqueeze(0)
        campos = (-R.T @ t).unsqueeze(0)
        K = np.array(cam["K"], dtype=np.float32)
        w, h = int(cam["width"]), int(cam["height"])
        fx, fy = float(K[0, 0]), float(K[1, 1])
        znear, zfar = 0.01, 100.0
        fov_x_rad = 2.0 * np.arctan(w / (2.0 * fx))
        fov_y_rad = 2.0 * np.arctan(h / (2.0 * fy))
        tan_half_fov_y = np.tan(fov_y_rad / 2.0)
        tan_half_fov_x = np.tan(fov_x_rad / 2.0)
        top = tan_half_fov_y * znear
        bottom = -top
        right = tan_half_fov_x * znear
        left = -right
        P = torch.zeros(4, 4, device=self.device, dtype=torch.float32)
        z_sign = 1.0
        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        proj_t = P.T.unsqueeze(0)
        return view_t, proj_t, campos

    def get_centers_depths_ortho(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For logging: centers (n,3), depths (n,) = -t_z, ortho_err (n,) = ||R^T R - I||_F per view."""
        n = self._cam_6d.shape[0]
        centers = []
        depths = []
        ortho_errs = []
        for i in range(n):
            d6 = self._cam_6d[i]
            t = self._cam_t[i]
            R = _rotation_6d_to_matrix(d6)
            c = -R.T @ t
            centers.append(c)
            depths.append(-t[2].item())
            ortho = (R.T @ R - torch.eye(3, device=self.device)).norm("fro").item()
            ortho_errs.append(ortho)
        return torch.stack(centers), torch.tensor(depths, device=self.device), torch.tensor(ortho_errs, device=self.device)

    def regularization_loss(
        self,
        min_dist: float,
        reg_dist_weight: float,
        reg_tz_pos_weight: float,
        reg_l2_weight: float,
    ) -> torch.Tensor:
        """Soft penalties: camera too close to origin, t_z > 0, L2 away from init."""
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        n = self._cam_6d.shape[0]
        for i in range(n):
            d6 = self._cam_6d[i]
            t = self._cam_t[i]
            R = _rotation_6d_to_matrix(d6)
            c = -R.T @ t
            dist = c.norm()
            if dist < min_dist:
                loss = loss + reg_dist_weight * (min_dist - dist) ** 2
            if t[2] > 0:
                loss = loss + reg_tz_pos_weight * (t[2] ** 2)
        loss = loss + reg_l2_weight * (
            (self._cam_6d - self._cam_6d_init).pow(2).sum() + (self._cam_t - self._cam_t_init).pow(2).sum()
        )
        return loss


def _load_gt_image(image_dir: Path, name: str) -> torch.Tensor:
    """Load image as tensor (C,H,W) in [0,1]."""
    for ext in IMAGE_EXTENSIONS:
        p = image_dir / (Path(name).stem + ext)
        if p.exists():
            from PIL import Image
            img = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            return torch.from_numpy(img).permute(2, 0, 1)
    raise FileNotFoundError(f"[train_3dgs] GT image not found: {name} in {image_dir}")


def _initial_dc_color_from_gt(
    image_dir: Path,
    image_list: list[str],
    device: torch.device,
    n_images: int = 8,
) -> torch.Tensor:
    """
    Average mean RGB of first N GT images → (1, 1, 3) for sh_dc init.
    Gives scene-specific initial color so DC is not black and gradients can flow.
    """
    n = min(n_images, len(image_list))
    if n == 0:
        return torch.full((1, 1, 3), INIT_DC_FALLBACK, device=device, dtype=torch.float32)
    acc = torch.zeros(3, device=device, dtype=torch.float32)
    for i in range(n):
        gt = _load_gt_image(image_dir, image_list[i]).to(device)
        acc += gt.mean(dim=(1, 2))
    mean_rgb = (acc / n).clamp(0.0, 1.0)
    # Ensure non-black: clamp to at least 0.2 so rasterizer always gets visible color
    if mean_rgb.max() < 0.1:
        mean_rgb = torch.full_like(mean_rgb, INIT_DC_FALLBACK)
    mean_rgb = mean_rgb.clamp(min=0.2)
    return mean_rgb.view(1, 1, 3)


def _sample_gt_colors_at_points(
    cam: dict[str, Any],
    gt_image: torch.Tensor,
    xyz_world: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Project world points into the first view and sample GT RGB. Returns (N, 3).
    Points behind camera or outside image get NaN; caller should replace with fallback.
    Convention: p_cam = R @ p + t, in front when z_cam < 0; project with fx,fy,cx,cy.
    """
    R = torch.from_numpy(np.array(cam["R"], dtype=np.float32)).to(device)
    t = torch.from_numpy(np.array(cam["t"], dtype=np.float32)).to(device)
    K = np.array(cam["K"], dtype=np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w, h = int(cam["width"]), int(cam["height"])
    # p_cam = R @ p + t  (3,), (N,3) -> (N,3)
    p_cam = (xyz_world @ R.T) + t.unsqueeze(0)
    z = p_cam[:, 2]
    # in front: z < 0 in our convention; avoid div by zero for points behind
    denom = torch.where(z < -0.01, (-z).clamp(min=1e-6), torch.ones_like(z, device=device))
    valid = (z < -0.01) & (denom < 1e4)
    u = fx * (p_cam[:, 0] / denom) + cx
    v = fy * (p_cam[:, 1] / denom) + cy
    # pixel coordinates: round and clamp
    u_idx = (u.round().long().clamp(0, w - 1))
    v_idx = (v.round().long().clamp(0, h - 1))
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h) & valid
    # sample: gt_image is (3, H, W)
    C, H, W = gt_image.shape
    colors = torch.full((xyz_world.shape[0], 3), float("nan"), device=device, dtype=gt_image.dtype)
    if in_bounds.any():
        v_idx_safe = v_idx.clamp(0, H - 1)
        u_idx_safe = u_idx.clamp(0, W - 1)
        sampled = gt_image[:, v_idx_safe, u_idx_safe].T  # (N, 3)
        colors = torch.where(in_bounds.unsqueeze(1), sampled, colors)
    return colors


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
        # Cap max scale so a few Gaussians don't explode into blobs (keeps shape readable).
        return torch.exp(self._log_scale).clamp(min=1e-6, max=0.15)

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
    log_diagnostics: bool = False,
    view_proj_campos: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Render one view; returns (3, H, W) RGB in [0,1]. view_proj_campos = (view_t, proj_t, campos) when using learnable cameras."""
    if view_proj_campos is not None:
        view_t, proj_t, campos = view_proj_campos
        viewmatrix = view_t.squeeze(0) if view_t.dim() == 3 else view_t
        projmatrix = proj_t.squeeze(0) if proj_t.dim() == 3 else proj_t
    else:
        viewmatrix, projmatrix = _camera_to_view_proj(cam, device)
        R = np.array(cam["R"], dtype=np.float32)
        t = np.array(cam["t"], dtype=np.float32)
        campos = torch.from_numpy((-R.T @ t).astype(np.float32)).to(device).unsqueeze(0)
    w, h = int(cam["width"]), int(cam["height"])
    xyz = gaussians.get_xyz()
    scales = gaussians.get_scales()
    quats = gaussians.get_rotations()
    # Rasterizer expects opacities (N, 1) only. Do not expand to (N, 1, 3) or alpha compositing breaks → black screen.
    opacity = gaussians.get_opacity()  # (N, 1)
    sh_dc = gaussians.get_sh_dc()
    sh_rest = gaussians.get_sh_rest()

    # means2D / cov3D: rasterizer may recompute from means3D; pass zeros if API requires
    # means2D = torch.zeros_like(xyz, device=device)
    #cov3D = torch.zeros((xyz.shape[0], 6), device=device)
    means2D = torch.zeros((xyz.shape[0], 2), device=device, dtype=xyz.dtype)

    # diff-gaussian-rasterization backward returns grad_colors_precomp as (N, 3).
    # Passing (N, 1, 3) causes "invalid gradient at index 3". Pass (N, 3) so gradient shape matches.
    if gaussians.sh_degree == 0:
        shs = None
        colors_precomp = gaussians.get_sh_dc().squeeze(1)  # (N, 1, 3) -> (N, 3)
    else:
        shs = torch.cat([sh_dc, sh_rest], dim=1)
        colors_precomp = None

    if log_diagnostics:
        # Camera-space z: p_cam = view @ p_world (viewmatrix is 4x4 world-to-camera)
        xyz_1 = torch.nn.functional.pad(xyz, (0, 1), value=1.0)  # (N, 4)
        xyz_cam = (viewmatrix @ xyz_1.T).T[:, :3]
        z_cam = xyz_cam[:, 2]
        n_total = xyz.shape[0]
        n_z_neg = (z_cam < 0).sum().item()
        n_z_pos = (z_cam > 0).sum().item()
        print(
            f"[render diag] xyz_cam z: min={z_cam.min().item():.4f} max={z_cam.max().item():.4f} "
            f"| z<0 (front in OpenGL): {n_z_neg}/{n_total} | z>0: {n_z_pos}/{n_total}"
        )
        print(
            f"[render diag] opacity: min={opacity.min().item():.4f} max={opacity.max().item():.4f} | "
            f"colors_precomp: min={colors_precomp.min().item():.4f} max={colors_precomp.max().item():.4f} | "
            f"scales: min={scales.min().item():.6f} max={scales.max().item():.4f}"
        )

    # tanfov from fixed K
    K = np.array(cam["K"])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    tanfovx = w / (2.0 * fx)
    tanfovy = h / (2.0 * fy)
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
        antialiasing=False,
        debug=False,
    )
    rasterizer = rasterizer_fn(settings)
    out = rasterizer(
        means3D=xyz,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,  # (N, 1) only
        scales=scales,
        rotations=quats,
    )

    if isinstance(out, tuple):
        out = out[0]
    out_img = out.squeeze(0)
    if log_diagnostics:
        print(
            f"[render diag] out RGB: min={out_img.min().item():.4f} max={out_img.max().item():.4f} mean={out_img.mean().item():.4f} shape={tuple(out_img.shape)}"
        )
    return out_img


def run(
    gs_output_dir: Path | None = None,
    image_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    max_steps: int = MAX_STEPS,
    save_every: int = SAVE_EVERY,
    lr: float = LR,
    device: str | None = None,
    camera_learn: bool = CAMERA_LEARN,
    camera_freeze_steps: int = CAMERA_FREEZE_STEPS,
    lr_camera_ratio: float = LR_CAMERA_RATIO,
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
    # Warn if all rotations are identity (e.g. ROMP failed in Stage 2) → color can stay gray
    R_list = [np.array(c["R"], dtype=np.float64) for c in cameras]
    I = np.eye(3, dtype=np.float64)
    all_identity = all(np.linalg.norm(R - I, "fro") < 0.15 for R in R_list)
    if all_identity:
        print(
            "[train_3dgs] WARNING: All camera rotations are near identity. "
            "Poses may be wrong → color can fail to learn (gray output). "
            "Consider re-running Stage 2 with ROMP/simple_romp installed for pose-aware cameras.",
            flush=True,
        )
    RasterSettings, GaussianRasterizer = _get_gaussian_rasterizer()

    num_points = len(points)
    gaussians = GaussianModel(num_points, sh_degree=SH_DEGREE, device=device)
    gaussians.set_xyz(torch.from_numpy(points).float().to(device))

    # Scale: larger init so splats are visible and color gradients can flow (official often use NN-distances; we use fixed)
    gaussians._log_scale.data.fill_(float(np.log(INIT_SCALE)))
    gaussians._logit_opacity.data = torch.logit(torch.full((num_points, 1), INIT_OPACITY, device=device))

    # DC color: init from GT mean (not black). Optionally per-point from first view to break gray symmetry.
    dc_init = _initial_dc_color_from_gt(
        image_dir, image_list, device, n_images=INIT_DC_FROM_GT_N_IMAGES
    )
    if INIT_DC_FROM_VIEW and image_list and cameras:
        first_gt = _load_gt_image(image_dir, image_list[0]).to(device)
        sampled = _sample_gt_colors_at_points(cameras[0], first_gt, gaussians.get_xyz(), device)
        fallback_rgb = dc_init.squeeze(0).squeeze(0)
        valid = torch.isfinite(sampled).all(dim=1)
        dc_data = torch.where(
            valid.unsqueeze(1),
            sampled.clamp(0.0, 1.0),
            fallback_rgb.unsqueeze(0).expand(num_points, 3),
        )
        gaussians._sh_dc.data = dc_data.unsqueeze(1).clone()
        n_valid = valid.sum().item()
        print(f"[train_3dgs] DC init: {n_valid}/{num_points} points from first-view projection, rest from mean.", flush=True)
    else:
        gaussians._sh_dc.data = dc_init.expand(num_points, 1, 3).clone()

    # Debug: log initial energy so we can confirm non-black
    with torch.no_grad():
        op_mean = gaussians.get_opacity().mean().item()
        scale_mean = gaussians.get_scales().mean().item()
        dc_mean = gaussians.get_sh_dc().mean().item()
    print(
        f"[train_3dgs] Init stats: opacity.mean()={op_mean:.4f}, scale.mean()={scale_mean:.4f}, sh_dc.mean()={dc_mean:.4f}",
        flush=True,
    )

    # Optimizer: Gaussians + optional learnable cameras
    param_groups = [
        {"params": [gaussians._xyz], "lr": lr * 0.01},
        {"params": [gaussians._log_scale], "lr": lr},
        {"params": [gaussians._quat], "lr": lr * 0.001},
        {"params": [gaussians._logit_opacity], "lr": lr * 0.05},
        {"params": [gaussians._sh_dc, gaussians._sh_rest], "lr": lr * 0.2},
    ]
    learnable_cameras: LearnableCameras | None = None
    lr_camera = lr * lr_camera_ratio
    if camera_learn:
        learnable_cameras = LearnableCameras(cameras, device)
        param_groups.append({"params": [learnable_cameras._cam_6d, learnable_cameras._cam_t], "lr": lr_camera})
        print(
            f"[train_3dgs] Learnable cameras (Option A). lr_camera={lr_camera:.2e} (ratio={lr_camera_ratio}), "
            f"freeze first {camera_freeze_steps} steps.",
            flush=True,
        )
    optimizer = torch.optim.Adam(param_groups)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    n_views = len(cameras)
    print(
        f"[train_3dgs] Training: {num_points} Gaussians, {n_views} views, device={device}",
        flush=True,
    )
    dc_rgb = gaussians.get_sh_dc().mean(dim=0).squeeze().tolist()
    print(
        f"[train_3dgs] Init: scale={INIT_SCALE}, opacity={INIT_OPACITY}, sh_dc mean RGB=[{dc_rgb[0]:.3f}, {dc_rgb[1]:.3f}, {dc_rgb[2]:.3f}]",
        flush=True,
    )

    for step in range(max_steps):
        optimizer.zero_grad()
        idx = step % n_views
        cam = cameras[idx]
        gt = _load_gt_image(image_dir, image_list[idx]).to(device)
        do_debug = (step + 1) % DEBUG_SAVE_EVERY == 0
        view_proj_campos = None
        if learnable_cameras is not None:
            view_proj_campos = learnable_cameras.get_view_proj_and_campos(idx)
        try:
            out = _render_one_view(
                gaussians, cam, GaussianRasterizer, RasterSettings, device, log_diagnostics=do_debug,
                view_proj_campos=view_proj_campos,
            )
        except Exception as e:
            print(f"[train_3dgs] Render failed at step {step}: {e}", flush=True)
            raise
        # Resize if needed
        if out.shape[1] != gt.shape[1] or out.shape[2] != gt.shape[2]:
            out = torch.nn.functional.interpolate(
                out.unsqueeze(0), size=(gt.shape[1], gt.shape[2]), mode="bilinear", align_corners=False
            ).squeeze(0)
        loss = (out - gt).abs().mean()
        if learnable_cameras is not None and step >= camera_freeze_steps:
            reg = learnable_cameras.regularization_loss(
                MIN_CAM_DIST, REG_DIST_WEIGHT, REG_TZ_POS_WEIGHT, REG_L2_WEIGHT
            )
            loss = loss + reg
        loss.backward()
        if learnable_cameras is not None and step < camera_freeze_steps:
            if learnable_cameras._cam_6d.grad is not None:
                learnable_cameras._cam_6d.grad.zero_()
            if learnable_cameras._cam_t.grad is not None:
                learnable_cameras._cam_t.grad.zero_()
        optimizer.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                op_m = gaussians.get_opacity().mean().item()
                sc_m = gaussians.get_scales().mean().item()
                dc_m = gaussians.get_sh_dc().mean().item()
            print(
                f"[train_3dgs] step {step + 1}/{max_steps} loss={loss.item():.6f} | opacity.mean={op_m:.4f} scale.mean={sc_m:.4f} sh_dc.mean={dc_m:.4f}",
                flush=True,
            )
        if learnable_cameras is not None and (step + 1) % CAMERA_LOG_EVERY == 0:
            with torch.no_grad():
                centers, depths, ortho = learnable_cameras.get_centers_depths_ortho()
                c_min = centers.min(dim=0).values.cpu().numpy()
                c_max = centers.max(dim=0).values.cpu().numpy()
                c_var = centers.var(dim=0).cpu().numpy()
                mean_depth = depths.mean().item()
                mean_ortho = ortho.mean().item()
                cam_frozen = " (cam frozen)" if step < camera_freeze_steps else ""
            print(
                f"[camera] step {step + 1} centers min=({c_min[0]:.3f},{c_min[1]:.3f},{c_min[2]:.3f}) "
                f"max=({c_max[0]:.3f},{c_max[1]:.3f},{c_max[2]:.3f}) var=({c_var[0]:.4f},{c_var[1]:.4f},{c_var[2]:.4f}) "
                f"mean_depth={mean_depth:.3f} ortho_err={mean_ortho:.6f}{cam_frozen}",
                flush=True,
            )
        # Debug: save intermediate render to verify Gaussians are visible
        if (step + 1) % DEBUG_SAVE_EVERY == 0:
            debug_dir = checkpoint_dir / "debug_renders"
            debug_dir.mkdir(parents=True, exist_ok=True)
            out_u8 = (out.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            try:
                from PIL import Image
                Image.fromarray(out_u8).save(debug_dir / f"step_{step + 1:06d}.png")
            except Exception as e:
                print(f"[train_3dgs] Debug save failed: {e}", flush=True)
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
            if learnable_cameras is not None:
                learned_cams = []
                for i in range(n_views):
                    with torch.no_grad():
                        d6 = learnable_cameras._cam_6d[i]
                        t = learnable_cameras._cam_t[i].cpu().numpy()
                        R = _rotation_6d_to_matrix(d6).cpu().numpy()
                    c = dict(cameras[i])
                    c["R"] = R.tolist()
                    c["t"] = t.tolist()
                    learned_cams.append(c)
                ckpt["cameras"] = learned_cams
                ckpt["cam_6d"] = learnable_cameras._cam_6d.detach().cpu().numpy()
                ckpt["cam_t"] = learnable_cameras._cam_t.detach().cpu().numpy()
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
    parser.add_argument("--no-camera-learn", action="store_true", help="Disable learnable cameras (use fixed cameras only)")
    parser.add_argument("--camera-freeze-steps", type=int, default=CAMERA_FREEZE_STEPS, help=f"Steps to freeze camera params (default {CAMERA_FREEZE_STEPS})")
    parser.add_argument("--lr-camera-ratio", type=float, default=LR_CAMERA_RATIO, help=f"Camera lr = lr * this (default {LR_CAMERA_RATIO})")
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
            camera_learn=not args.no_camera_learn,
            camera_freeze_steps=args.camera_freeze_steps,
            lr_camera_ratio=args.lr_camera_ratio,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
