# 카메라·입력 변동성 대응 업그레이드 옵션 평가

## 현재 문제 요약

- **Stage 1**: 규칙 기반 전처리 → 배경 잔존, 인물 잘림, 프레임 간 scale/중심 drift
- **Stage 2**: synthetic/estimated 카메라 → 3DGS가 blur collapse (형체 없는 안개)
- **카메라 depth·focal·world scale**은 이미 정규화 완료

목표: **입력 변동성에 robust**, 사용자 부담·순서 강제 최소화, 카메라가 완벽하지 않아도 수렴.

---

## 옵션 평가 (5가지 기준)

| 옵션 | 설명 | (1) 코드베이스 호환 | (2) 구현 난이도 | (3) Colab T4 | (4) Robustness 기대 | (5) MVP 현실성 |
|------|------|---------------------|-----------------|--------------|----------------------|----------------|
| **A** | **3DGS + Camera Joint Optimization** (Gaussian과 R,t 동시 학습) | ◎ 기존 train_3dgs 확장만 | ◎ 낮음 | ◎ 동일 메모리 | ◎ 높음 | ◎ 가장 높음 |
| **B** | DUSt3R / pose-free multi-view depth → 3DGS 초기값 | △ Stage2·init 교체 | △~○ 중간 | △ 모델 무거움 | ◎ 높음 | △ 중간 |
| **C** | BARF / Pose-Free NeRF 스타일 pose refinement | × NeRF 기반, 3DGS와 다름 | × 높음 | △ NeRF 비용 | ◎ 높음 | × 낮음 |
| **D** | Instant-NGP + camera optimization | × 렌더러 전면 교체 | × 높음 | △ 가능하나 별도 스택 | ◎ 높음 | × 낮음 |
| **E** | SMPL + differentiable renderer + multi-view 학습 | △ Stage2 연계 가능 | × 높음 | △ 무거움 | ◎ 높음 | × 연구 수준 |

---

## (1) 호환성 상세

- **A**: 현재 `train_3dgs.py`가 `cameras[idx]`로 고정 R,t를 쓰고 있음 → **같은 루프에서 R,t를 Parameter로 두고 optimizer에 넣으면 됨**. Stage 1/2/init_3dgs 출력 형식 유지.
- **B**: DUSt3R 출력(상대 포즈·depth)을 우리 K,R,t·point_cloud 형식으로 변환하는 어댑터 필요. Stage 2 대체 또는 보조.
- **C**: BARF는 NeRF + pose refinement. 우리는 3DGS이므로 “pose refinement만” 가져오면 사실상 A와 동일 아이디어.
- **D**: diff-gaussian-rasterization 제거하고 Instant-NGP 기반으로 전환 → 파이프라인 대규모 변경.
- **E**: SMPL mesh + 렌더러 + 멀티뷰 loss는 새 학습 그래프 설계 필요. MVP 범위를 넘어섬.

---

## (2) 구현 난이도

- **A**: 회전 파라미터화(6D continuous 또는 quaternion + 정규화), t 3차원, view matrix 구성만 추가. 기존 `_camera_to_view_proj`를 “learnable R,t → view”로 바꾸는 부분만 구현하면 됨.
- **B**: DUSt3R API·출력 포맷 이해, scale/좌표계 정렬, 실패 시 fallback 처리.
- **C/D/E**: 각각 NeRF/NGP/전용 학습 파이프라인 구현 또는 도입.

---

## (3) Colab T4

- **A**: 현재 3DGS와 동일 (뷰당 렌더 1회). 파라미터만 카메라 N개 × (6D+3) 추가 → T4에서 무리 없음.
- **B**: DUSt3R 추론이 추가되나, 한 번만 돌리면 됨. T4에서 가능.
- **C/D/E**: NeRF/NGP/대형 렌더러는 메모리·속도 부담이 상대적으로 큼.

---

## (4) Robustness

- **A**: 초기 카메라가 틀려도 **photometric loss로 R,t가 함께 갱신**되므로, Stage 1 drift·Stage 2 추정 오차를 학습 중에 보정 가능. “카메라를 완벽히 알 필요 없음”이 목표와 정확히 일치.
- **B**: DUSt3R이 잘 되면 초기값이 좋아져 수렴이 쉬워지지만, 실패 시 fallback이 필요. A와 병행하면 더 robust.
- **C/D/E**: 이론적으로는 pose refinement로 robust하나, 우리 스택과의 통합 비용이 큼.

---

## (5) MVP 현실성

- **A**: **한 모듈(train_3dgs)만 수정**, 기존 Stage 1→2→init 흐름 유지. 배포·실험 사이클이 짧음.
- **B**: 새 의존성·데이터 경로 도입. MVP 후 “초기값 개선” 단계로 적합.
- **C/D/E**: MVP 단계에서 채택하기엔 작업량·리스크가 큼.

---

# 선택 결론: **옵션 A (3DGS + Camera Joint Optimization)**

## 선택 이유 (기술적)

1. **문제와 직접 대응**  
   Blur collapse의 한 가지 핵심 원인은 **고정된 잘못된 카메라**이다.  
   R,t를 학습 가능하게 두면, 같은 L1 재구성 loss로 **geometry와 pose를 동시에 맞추는** 방향으로 gradient가 흐른다.

2. **최소 변경으로 효과**  
   - Stage 1: 지금처럼 두거나 단순화해도 됨 (규칙 기반 개선 제외 요청 반영).  
   - Stage 2: synthetic/estimated 모두 **초기값**으로만 쓰고, 정확할 필요 없음.  
   - Stage 3: **train_3dgs만** “고정 카메라” → “learnable 카메라”로 바꾸면 됨.

3. **수치적 안정성**  
   - 회전은 **6D continuous representation** (Zhou et al.)으로 두면 미분 가능하고 특이점이 없음.  
   - t는 3차원 그대로 두고, 초기값을 Stage 2 출력으로 두면 수렴 구간 안에 들어가기 쉬움.

4. **검증된 접근**  
   NeRF/3DGS 계열에서 “joint optimization of scene + cameras”는 여러 논문에서 쓰인 패턴이다. 우리는 3DGS + diff-gaussian-rasterization만 유지하면 되므로, 그 중 가장 가벼운 형태가 A다.

---

# Option A: 구체적 구조 설계

## 1. 파이프라인 역할 (변경 후)

```
[변경 없음]
Stage 1 (전처리)     → processed_images (가능하면 일정 scale/중심, 아니어도 됨)
Stage 2 (human prior) → canonical_mesh.ply, cameras.json (초기 카메라만 제공)
init_3dgs            → point_cloud.npy, cameras.json, image_list.txt

[변경]
train_3dgs           → Gaussian 파라미터 + **카메라 R,t (학습)** → 동일 L1 loss로 공동 최적화
```

- **Stage 1**: “규칙 기반으로 더 고치지 않는다”는 전제. 출력이 들쭉날쭉해도, **카메라를 학습**하는 쪽에서 보정을 기대.
- **Stage 2**: synthetic이든 estimated이든 **초기 pose**만 제공. 정확하지 않아도 됨.
- **train_3dgs**:  
  - 입력: 기존과 동일 (gs_output + processed_images).  
  - 내부: **N개 뷰에 대해 R_i, t_i를 Parameter로 두고**, 매 step 해당 뷰의 view matrix를 R_i, t_i로 만들고, 기존처럼 render → L1 loss → backward.  
  - 출력: 기존 checkpoint + **학습된 cameras.json** (또는 별도 cam_ckpt).

## 2. train_3dgs 변경 포인트

### 2.1 카메라 파라미터 저장소

- **회전**: 뷰당 **6D 벡터** (6D → 3×3 R 변환 사용). 또는 quaternion + normalize.
- **평행이동**: 뷰당 **t ∈ R^3**.
- **초기값**: 현재 `cameras.json`의 R, t를 그대로 넣음.  
- **고정 여부**: K (fx, fy, cx, cy)는 **고정** 권장. 해상도가 같다면 K 학습은 선택 사항.

### 2.2 뷰 행렬 생성 (미분 가능)

- 6D → R:  
  - 6D를 두 개 3차원 벡터 a, b로 나눔.  
  - Gram–Schmidt로 정규 직교 행렬 만들기 (R = [c1, c2, c3]).  
  - 이 연산은 미분 가능하므로 `_camera_to_view_proj`를 “numpy R,t” 대신 “torch 6D, t”를 받아서 view/proj를 **tensor**로 만드는 버전으로 확장.
- **Z-flip**: 기존과 동일 (rasterizer가 +Z forward 기대하므로 view[2,:] *= -1).

### 2.3 Optimizer

- 기존 parameter groups에 추가:  
  `{"params": [camera_6d, camera_t], "lr": lr_cam}` (예: lr_cam = 1e-4 ~ 5e-4).
- 카메라만 너무 크게 움직이지 않도록 **학습률을 Gaussian보다 작게** 두는 것이 안전.

### 2.4 학습 루프

- `idx = step % n_views` → 해당 뷰의 **learnable R(6D), t**로 view, proj 계산 → `_render_one_view(gaussians, view_t, proj_t, ...)` 형태로 렌더.
- 기존처럼 `(out - gt).abs().mean().backward()` → **Gaussian + 해당 뷰 카메라**까지 함께 갱신.

### 2.5 체크포인트

- 저장 시: `ckpt["cameras"]` 또는 `ckpt["camera_6d"]`, `ckpt["camera_t"]`를 함께 저장.
- 재학습/렌더 시: 이 값을 불러와서 사용 (render_character 등에서도 “학습된 카메라” 사용 가능).

## 3. 구현 시 유의사항

- **6D → R**:  
  - `a, b = 6d[:3], 6d[3:]`  
  - `c1 = a / (||a||+eps)`, `c2 = b - (b·c1)c1`, `c2 = c2 / (||c2||+eps)`, `c3 = c1×c2`  
  - `R = stack([c1,c2,c3], dim=1)` (3x3).  
  - det(R) = 1이 되도록 필요 시 c3 부호 조정 (reflection 방지).
- **초기 6D**: 기존 R(3,3)을 6D로 변환해 초기값으로 넣을 수 있음 (R의 첫 두 열을 펼치면 6D 한 가지 해).
- **K 고정**: width, height, fx, fy, cx, cy는 기존 cameras에서 읽어와서 **상수**로 사용.  
  → proj matrix는 기존처럼 numpy/torch 상수로 두고, **view만 learnable**로 두면 구현이 단순해짐.

## 4. 기대 효과

- Stage 1에서 scale/중심이 들쭉날쭉해도, **카메라가 학습**되면서 각 뷰가 “어디서 찍었는지”를 스스로 맞춰 감.
- Stage 2가 synthetic(원형 궤도)이어도, 실제 이미지가 그 궤도와 다르면 **R,t가 그에 맞게 이동**.
- ROMP 등이 없어 estimated가 전부 identity여도, **학습 초반에 rotation/translation이 갱신**되기 시작하면 색·형태가 정리될 가능성이 높음.

## 5. 다음 단계 (실제 코드 수정 시)

1. **`gs/train_3dgs.py`**  
   - `cameras`를 “고정 dict 리스트”가 아니라 **learnable tensor (6D + t per view)** 로 로드.  
   - 매 step view matrix를 이 tensor들로부터 미분 가능하게 계산.  
   - optimizer에 camera parameter group 추가.  
   - ckpt 저장/로드에 카메라 파라미터 포함.

2. **`_camera_to_view_proj`**  
   - “cam dict” 대신 (R_6d_tensor, t_tensor, K, w, h)를 받는 버전 추가하거나,  
   - cam dict에서 R,t를 읽되 **이미 tensor이고 requires_grad=True**인 경로를 하나 두어서, 그 tensor로 view를 만드는 분기 추가.

3. **`gs/render_character.py`**  
   - 학습된 checkpoint에서 카메라를 불러와 사용할 수 있도록 옵션 추가 (기존 cameras.json 대신 ckpt 내 cameras 사용).

이렇게 하면 **Stage 1 규칙 기반 수정 없이**, “카메라도 함께 학습하는 구조”로 전환할 수 있고, 입력 변동성에 대한 robustness를 가장 낮은 비용으로 높일 수 있다.
