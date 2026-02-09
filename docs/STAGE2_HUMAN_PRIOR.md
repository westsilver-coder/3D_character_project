# Stage 2: Human-Prior 기반 3D 복원 (COLMAP 제거)

Stage 2는 **2a(prior) → 2b(초기화) → 2c(학습)** 로 명확히 분리된다.  
기존에 “train_3dgs”가 초기화만 담당했던 부분은 **init_3dgs.py(2b)** 로, 실제 3DGS 학습은 **train_3dgs.py(2c)** 로 정리된다.

---

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

## 3. Stage 2 단계별 구분 (2a / 2b / 2c)

### Stage 2a — Human shape prior 생성
- **입력**: Stage 1 출력 `data/processed_images/` (이미지 목록·해상도만 사용)
- **출력**: `data/human_prior/` — canonical_mesh.ply (~6890 verts), cameras.json, image_list.txt
- **처리**: SMPL_NEUTRAL.pkl 로드 → T-pose mesh. Synthetic orbit 카메라 생성. ROMP/COLMAP 없음.
- **구현**: `pipeline/stage2_reconstruct.py`

### Stage 2b — 3DGS 초기화 (학습·렌더링 없음)
- **입력**: `data/human_prior/` (canonical_mesh.ply, cameras.json, image_list.txt)
- **출력**: `data/gs_output/` — point_cloud.npy, cameras.json, image_list.txt
- **처리**: mesh 표면 샘플링 → 초기 Gaussian 중심. **optimizer, loss, 렌더링 미포함**
- **구현**: `gs/init_3dgs.py`

### Stage 2c — 3DGS 학습
- **입력**: `data/gs_output/point_cloud.npy`, `cameras.json`, `data/processed_images/` (GT 이미지)
- **출력**: 학습된 3DGS checkpoint (예: data/gs_checkpoints/)
- **처리**: diff-gaussian-splatting 스타일 training loop. COLMAP 미사용. world = canonical body space.
- **구현**: `gs/train_3dgs.py`

---

## 4. 3DGS 초기화·학습 연결 (2b → 2c)

- **2b (init_3dgs)**: `canonical_mesh.ply` 표면 샘플링 → point_cloud.npy. cameras.json, image_list.txt export. **학습·optimizer·loss·렌더링은 포함하지 않음.**
- **2c (train_3dgs)**: point_cloud.npy + cameras.json + GT 이미지로 실제 3DGS 학습. 카메라는 이미 K, R, t로 주어지며 COLMAP 미사용. world space = canonical human body space. 학습된 checkpoint 저장.

요약: **2a(prior) → 2b(초기화 전용) → 2c(학습)** 순서로, Human-prior가 3DGS의 geometry·view 기준을 대체한다.

---

## 5. 디렉터리 구조 (Stage 유지, COLMAP 제거)

- **Stage 1**: 변경 없음. `data/processed_images/` 생성.
- **Stage 2a**: `pipeline/stage2_reconstruct.py` — 출력 `data/human_prior/`.
- **Stage 2b**: `gs/init_3dgs.py` — 초기화 전용. 출력 `data/gs_output/`.
- **Stage 2c**: `gs/train_3dgs.py` — 학습. checkpoint 예: `data/gs_checkpoints/`.
- **Stage 3 이후**: `data/colmap/` 대신 `data/human_prior/`, `data/gs_output/`, checkpoint 참조.

`data/colmap/`, `database.db`, `sparse/` 는 사용하지 않음.
