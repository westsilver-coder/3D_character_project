# Stage 2: Human-Prior 기반 3D 복원 (COLMAP 제거)

## 1. 설계 배경

- **COLMAP 한계**: rigid SfM 기반이라 사람(non-rigid) 중심 장면에서 camera=1 등 구조적 실패가 빈번함.
- **목표**: 실사 인물 → 캐릭터화된 3D 인물(미니어처/게임 캐릭터). 포토그래메트리 정확도는 불필요, **deformation / stylization 가능한 구조**가 중요.
- **해결**: COLMAP을 제거하고 **인체 prior(SMPL 등)** 를 기준으로 한 Stage 2~3 재설계.

---

## 2. Geometry 기준 (COLMAP 대신 무엇이 기준인가)

| 구분 | COLMAP 파이프라인 | Human-Prior 파이프라인 (현재) |
|------|-------------------|------------------------------|
| **세계 좌표계** | SfM이 추정한 임의의 3D 좌표계 | **Canonical body space**: 인체 모델(SMPL)의 T-pose, 원점 = 몸 중심(pelvis 근처) |
| **기하 기준** | Sparse 3D points + triangulation | **Canonical mesh**: 평균 body shape(β) + T-pose(θ=0) SMPL 메시. 이후 3DGS는 이 메시 표면/근처에서 초기화 가능 |
| **카메라 정보** | SfM으로 추정된 R, t, K (실패 시 camera=1) | **Synthetic orbit**: 원형 궤도 perspective 카메라. canonical body 원점을 바라봄. “촬영 시점”을 canonical body 기준으로 표현 |
| **일관성** | 여러 뷰의 feature 일치로 결정 | **인체 prior로 뷰 간 일관성**: 모든 이미지가 “같은 사람”의 서로 다른 각도로 정의됨 |

정리하면, **geometry의 기준은 “Canonical human mesh (SMPL T-pose + 평균 shape)”** 이고, **각 이미지의 카메라는 이 canonical space에서의 시점(extrinsic + intrinsic)** 으로 저장된다. 3DGS는 이 카메라와 (선택) canonical 메시 기반 초기점으로 학습한다.

---

## 3. Stage 2 입출력 (Human-Prior 버전)

### 입력
- **Stage 1 출력**: `data/processed_images/` (전처리된 전신 인물 이미지)

### 출력 (기본: `data/human_prior/`)
- **`canonical_mesh.ply`**: SMPL T-pose 메시 (vertices ~6890). `SMPL_NEUTRAL.pkl`에서 직접 생성. Stage 3 Gaussian 초기화용.
- **`cameras.json`**: Synthetic orbit 카메라 (K, R, t). 3DGS 학습용.
- **`image_list.txt`**: 사용된 이미지 파일명 목록 (순서 일치).

### 처리 흐름
1. **SMPL 로드**: `SMPL_NEUTRAL.pkl` 직접 로드 → canonical T-pose 메시 → PLY 저장.
2. **Synthetic 카메라**: 입력 이미지 목록·해상도로 원형 궤도 카메라 생성 → cameras.json, image_list.txt.
3. ROMP/COLMAP 없음. Stage 2 = human shape prior 생성.

---

## 4. 3D Gaussian Splatting 초기화 연결

- **카메라**: `data/human_prior/cameras.json` + `image_list.txt` → 3DGS가 각 학습 이미지의 viewpoint로 사용.
- **초기 3D 점/가우시안**:
  - **권장**: `canonical_mesh.ply` 표면(및 약간 바깥)에서 점 샘플링 → 이 위치를 초기 Gaussian 중심으로 사용. (정확한 포토그래메트리 불필요하므로 대략적 표면이면 충분.)
  - **대안**: SfM 없이 랜덤/균일 초기화 후 카메라만 고정하고 3DGS만 학습하는 방식도 가능.
- **학습**: 기존 3DGS와 동일하게, 주어진 카메라와 이미지로 re-render → photo-metric loss로 최적화. COLMAP sparse points는 사용하지 않음.

요약: **Human-prior Stage 2의 출력(canonical mesh + cameras.json)이 3DGS의 “geometry·view 기준”을 대체**한다.

---

## 5. 디렉터리 구조 (Stage 유지, COLMAP만 제거)

- **Stage 1**: 변경 없음. `data/processed_images/` 생성.
- **Stage 2**: `pipeline/stage2_reconstruct.py` — COLMAP 호출 제거, human-prior 전용으로 교체. 출력은 `data/human_prior/`.
- **Stage 3 이후**: `data/colmap/` 대신 `data/human_prior/` (및 3DGS 체크포인트)를 참조하도록만 변경.

`data/colmap/`, `database.db`, `sparse/` 는 더 이상 생성·사용하지 않음.
