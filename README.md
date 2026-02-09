# 실사 전신 인물 → 템플릿 기반 애니 캐릭터 생성 시스템

**3D Vision / 3DGS 기반** · Python · Google Colab (T4 GPU)

---

## 한 줄 요약

실사 전신 이미지를 입력으로 **사람의 구조적 정보(키, 체형, 실루엣, 의상 색 등)** 만 추출하고, **고정된 애니 스타일 캐릭터 템플릿(마네킹)** 위에 이를 매핑하여 **귀엽고 일관된 애니/피규어 스타일** 3D 캐릭터를 생성하는 파이프라인. (실사 이미지를 직접 캐릭터처럼 변형하는 방식이 아님.)

---

## 전체 파이프라인 흐름

```
[입력] 전신 인물 이미지 20~40장 (다양한 시점)
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Stage 1  │  이미지 전처리 (리사이즈, 색상 정규화, 인물 마스크)        │
│  CV       │  → 3D 복원용 전처리 이미지 세트                            │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Stage 2a │  Human shape prior (canonical mesh + synthetic cameras)   │
│  Prior    │  → data/human_prior/                                       │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Stage 2b │  3DGS 초기화 (init_3dgs.py) — point_cloud, cameras export │
│  Init     │  → data/gs_output/                                         │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Stage 2c │  3DGS 학습 (train_3dgs.py) — GT 이미지로 Gaussian 학습    │
│  Train    │  → 3DGS checkpoint                                         │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Stage 3  │  템플릿에 적용할 비율/스케일 조건 정리 (데포르메 규칙 활용)│
│  Conditions│ → 조건 데이터 (최종 캐릭터 geometry 직접 생성 아님)       │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Stage 4  │  템플릿(PLY) 로드 + Stage 3 조건 적용                      │
│  Final    │  → 렌더 가능한 캐릭터 geometry/appearance (파이프라인 최종) │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
[파이프라인 출력] 애니/피규어 스타일 3D 캐릭터 (PLY 등)

  ※ 렌더링은 pipeline 단계 아님. 필요 시 gs/render_character.py 로 선택 수행.
```

**핵심:** 파이프라인은 Stage 4까지. Stage 4 출력 = 렌더 가능한 캐릭터. 템플릿 `character/templates/sd_chibi_base.ply` 고정. Stage 5 없음. 렌더링은 gs/render_character.py 선택 수행.

**설계 원칙:** Stage 3은 조건(dict)만 정리, Stage 4가 최종 캐릭터 생성. Stage 5 없음. 렌더링은 선택.

---

## 프로젝트 목표

| 항목 | 내용 |
|------|------|
| **입력** | 동일 인물 전신 이미지 20~40장, 다양한 시점 |
| **핵심 처리** | 사람 구조 정보 추출(2a~2c) → 비율/조건 정리(3) → **템플릿 위에 매핑(4)** |
| **출력** | 귀엽고 일관된 애니/피규어 스타일 3D 캐릭터 (템플릿 기반) |
| **환경** | Python, Google Colab, T4 · 고해상도/실시간은 목표 아님 |

---

## 구현 순서 권장

파이프라인이 순서대로 돌아가려면 아래 순서로 구현하는 것을 권장한다.

| 순서 | 구현 대상 | 이유 |
|------|-----------|------|
| 1 | `character/deformation_rules.py`, `character/style_presets.py` | 규칙·프리셋 정의만 있으면 됨. 의존성 없고 이후 Stage의 “설정 계약”이 됨. |
| 2 | `pipeline/stage1_preprocess.py` | OpenCV/PIL 등으로 구현 가능. Stage 2 입력을 만드는 단계라 먼저 완성해야 함. |
| 3 | Stage 2a: `pipeline/stage2_reconstruct.py` | Human shape prior (canonical mesh + synthetic cameras). |
| 4 | Stage 2b: `gs/init_3dgs.py` | 3DGS 초기화 전용. point_cloud.npy, cameras.json 생성. 학습/렌더링 없음. |
| 5 | Stage 2c: `gs/train_3dgs.py` | init_3dgs 출력 + GT 이미지로 실제 3DGS 학습 → checkpoint 저장. |
| 6 | `gs/deform_geometry.py` + `pipeline/stage3_deform.py` | Stage 2/3 조건: 템플릿에 적용할 비율/스케일 정리. |
| 7 | `pipeline/stage4_style.py` | **템플릿 로드 + Stage 3 조건 적용** → 렌더 가능한 캐릭터 geometry/appearance (파이프라인 최종). |

**정리:** 파이프라인 = Stage 1 → 2a→2b→2c → Stage 3(조건 정리) → Stage 4(최종 캐릭터 생성). 렌더링은 **선택** → `gs/render_character.py`.

---

## 입력 이미지 가이드

### 어떤 이미지를 넣어야 하나?

| 항목 | 권장 사항 |
|------|-----------|
| **위치** | `data/raw_images/` 에 넣는다. (Stage 1이 여기서 읽어 `data/processed_images/` 로 출력) |
| **장수** | **20~40장** 정도. 너무 적으면 3DGS 복원 품질이 떨어짐. |
| **구도** | **전신이 프레임에 들어오는** 사진. 머리부터 발끝까지 보이도록. |
| **시점** | **한 사람을 360도에 가깝게 돌려가며 촬영**. 앞·옆·뒤·대각선 등 다양한 각도가 있으면 3D 복원이 안정적. |
| **해상도** | 원본은 그대로 두어도 됨. Stage 1에서 최대 1024px 등으로 리사이즈함. |
| **촬영** | 같은 조명·같은 배경이 ideal이지만, 꼭 동일할 필요는 없음. 인물 마스크(Stage 1)로 배경은 줄일 수 있음. |

### 옷(의상)도 3D 미니어처로 만들고 싶을 때

**네. 입은 옷까지 3D 미니어처로 만들려면, 넣는 이미지 전부 “동일한 의상”을 입은 상태여야 합니다.**

- 3DGS는 **촬영된 그대로**를 3D로 복원한다.  
  → 모든 각도에서 **같은 옷**이 보여야, 옷 형태·주름·색이 일관되게 3D로 합쳐진다.
- 장면마다 옷이 바뀌면(예: 한 장은 티셔츠, 다른 장은 코트) 복원 결과가 뒤섞이고, 옷이 깨져 보이거나 일관성이 없어질 수 있다.

**정리:**

- **동일 인물 + 동일 의상 + 여러 각도(20~40장)** → **몸과 옷이 함께** 3D로 복원되고, 이후 데포르메·스타일을 적용하면 **옷 입은 캐릭터 미니어처**가 된다.
- `data/raw_images/` 에 그 조건을 만족하는 이미지만 넣어두면 된다.

---

## Stage별 상세: 목적 · 입출력 · 구현

### Stage 1 — 입력 이미지 전처리 (Computer Vision)

| 구분 | 내용 |
|------|------|
| **목적** | 3D 복원에 적합한 일관된 이미지 세트 만들기 |
| **입력** | 동일 인물 전신 이미지 20~40장 (다양한 시점) |
| **처리** | 이미지 리사이즈(최대 1024px), 색상 정규화, 인물 마스크(배경 단순화) |
| **출력** | `data/processed_images/` — 3D 복원용 전처리 이미지 세트 |
| **구현** | `pipeline/stage1_preprocess.py` |

---

### Stage 2a — Human shape prior 생성 (COLMAP/ROMP 미사용)

| 구분 | 내용 |
|------|------|
| **목적** | SMPL T-pose 메시 + synthetic orbit 카메라로 human shape prior 생성. Stage 2b에서 mesh 표면을 Gaussian 초기화에 사용 |
| **입력** | Stage 1 전처리 이미지 디렉터리 (이미지 목록·해상도만 사용) |
| **처리** | `SMPL_NEUTRAL.pkl` 직접 로드 → canonical T-pose mesh. 카메라는 원형 궤도(synthetic) |
| **출력** | `data/human_prior/` — canonical_mesh.ply (~6890 verts), cameras.json, image_list.txt |
| **SMPL 모델** | `data/smpl/SMPL_NEUTRAL.pkl` 또는 환경변수 `SMPL_MODEL_PATH` |
| **구현** | `pipeline/stage2_reconstruct.py` |

---

### Stage 2b — 3DGS 초기화 (학습/렌더링 없음)

| 구분 | 내용 |
|------|------|
| **목적** | Human-prior 기반 3DGS **초기화 전용**. canonical_mesh.ply 표면 샘플링 → 초기 Gaussian 중심 생성 |
| **입력** | `data/human_prior/` (canonical_mesh.ply, cameras.json, image_list.txt) |
| **처리** | mesh 표면 샘플링 → point_cloud.npy. cameras.json, image_list.txt export. **학습·optimizer·loss·렌더링 미포함** |
| **출력** | `data/gs_output/` — point_cloud.npy, cameras.json, image_list.txt |
| **구현** | `gs/init_3dgs.py` |

---

### Stage 2c — 3DGS 학습

| 구분 | 내용 |
|------|------|
| **목적** | init_3dgs 출력 + GT 이미지로 **실제 3D Gaussian Splatting 학습** (diff-gaussian-splatting 스타일) |
| **입력** | `data/gs_output/point_cloud.npy`, `cameras.json`, `data/processed_images/` (GT 이미지) |
| **처리** | Gaussian 파라미터(position, scale, rotation, opacity, SH) 학습. COLMAP 미사용. world = canonical body space |
| **출력** | 학습된 3DGS checkpoint (예: `data/gs_checkpoints/`) |
| **구현** | `gs/train_3dgs.py` |

---

### Stage 3 — 템플릿에 적용할 조건 정리 (역할 축소)

| 구분 | 내용 |
|------|------|
| **목적** | 최종 캐릭터를 만드는 단계가 아니라, **템플릿에 적용할 비율/스케일 조건**을 정리하는 단계 |
| **입력** | Stage 2c 체크포인트(condition 정보), 데포르메 규칙(JSON/dict) |
| **처리** | 데포르메 규칙을 활용해 비율·스케일 조건 정리. (머리/팔/다리 등 스케일) — Stage 4에서 템플릿에 적용 |
| **출력** | 조건 데이터 (템플릿 매핑용). 최종 캐릭터 geometry는 Stage 4에서 생성 |
| **구현** | `gs/deform_geometry.py`, `pipeline/stage3_deform.py`, `character/deformation_rules.py` |

---

### Stage 4 — 최종 캐릭터 생성 (파이프라인 마지막 단계)

| 구분 | 내용 |
|------|------|
| **목적** | 고정 템플릿(sd_chibi_base.ply) 로드 → Stage 3 조건 적용 → **렌더 가능한** 애니/피규어 스타일 캐릭터 geometry/appearance 생성 |
| **입력** | 템플릿 PLY(`character/templates/sd_chibi_base.ply`), Stage 3 조건(dict/JSON) |
| **처리** | 템플릿 로드 → 비율/스케일 델타 적용. 실사 눈/코/입 미사용. |
| **출력** | 렌더 가능한 캐릭터 geometry/appearance (예: `output/characters/` 에 PLY 등) |
| **구현** | `pipeline/stage4_style.py` |

---

### 렌더링 (파이프라인 단계 아님, 선택 사항)

| 구분 | 내용 |
|------|------|
| **역할** | Stage 4 이후, 필요 시에만 수행. 턴테이블 mp4·회전 뷰 등 시각화. |
| **입력** | Stage 4 출력(렌더 가능한 캐릭터) |
| **구현** | `gs/render_character.py` |

---

## 프로젝트 디렉터리 구조

```
project/
├── data/
│   ├── raw_images/          # 원본 전신 인물 이미지 (20~40장)
│   ├── processed_images/    # Stage 1 전처리 결과
│   ├── human_prior/         # Stage 2 출력 (no COLMAP): canonical_mesh.ply, cameras.json
│   ├── gs_output/           # Stage 2b: init_3dgs 출력 (point_cloud, cameras)
│   └── gs_checkpoints/      # Stage 2c: train_3dgs checkpoint
│
├── docs/
│   ├── STAGE2_HUMAN_PRIOR.md  # Stage 2 설계: 2a/2b/2c, geometry 기준
│   └── PROJECT_DIRECTION.md   # 프로젝트 목표 재정의, 수정 대상 최소 목록, Stage 4 로드맵
│
├── gs/                      # 3DGS·조건·렌더
│   ├── init_3dgs.py         # Stage 2b: 3DGS 초기화 전용
│   ├── train_3dgs.py        # Stage 2c: 3DGS 학습
│   ├── deform_geometry.py   # Stage 3: 조건(dict) 계산
│   └── render_character.py  # 렌더링 (선택, pipeline 단계 아님)
│
├── character/               # 규칙·프리셋 정의
│   ├── deformation_rules.py # 데포르메 규칙 로드/검증 (JSON·dict)
│   └── style_presets.py     # Stage 4용 스타일 프리셋
│
├── pipeline/                # Stage 진입점 (순차 실행)
│   ├── stage1_preprocess.py # Stage 1
│   ├── stage2_reconstruct.py # Stage 2a: human shape prior (no COLMAP)
│   ├── stage3_deform.py     # Stage 3
│   └── stage4_style.py      # Stage 4
│
├── output/
│   ├── characters/          # 최종 캐릭터 결과
│   └── videos/              # 턴테이블 mp4 등
│
└── README.md
```

- **pipeline/** : Stage 1~4 순서 실행. **Stage 5는 없음** (렌더링은 gs/render_character.py 선택 실행).
- **gs/** : 3DGS 학습·조건 계산·렌더링(선택) 구현.
- **character/** : 데포르메 규칙·템플릿·프리셋 등 설정.

---

## 사용자 정의 데포르메 설정

데포르메 규칙은 **JSON** 또는 **Python dict**로 정의한다.

```json
{
  "head_scale": 1.4,
  "arm_length": 0.8,
  "leg_length": 0.85,
  "torso_scale": 0.9,
  "style": "2.5D_character"
}
```

`character/deformation_rules.py`에서 로드·검증 후 Stage 3에서 사용.

---

## MVP 성공 조건

- [ ] 전신 인물 사진 → 3DGS 복원 성공
- [ ] 데포르메 규칙이 실제 기하 구조에 반영됨
- [ ] 실사가 아닌 **캐릭터화된** 3D 인물 생성
- [ ] 결과를 회전하여 시각적으로 확인 가능

---

## 설계 원칙

- **CV / 3D Vision 중심** — 2D→3D, 기하 변형, 렌더링에 집중
- **Geometry vs Appearance 분리** — Stage 3(기하) / Stage 4(외형) 명확히 구분
- **재현 가능한 파이프라인 우선** — 완벽한 품질보다 “끝까지 동작하는 흐름”
- **파라미터 조절** — 모든 주요 파라미터는 코드 상단에서 설정 가능

---

## 최종 결과 정의

> “실사 전신 인물을 촬영하면, 사용자가 정의한 캐릭터 규칙에 따라 3D 캐릭터로 변환해주는 3D 비전 시스템”
