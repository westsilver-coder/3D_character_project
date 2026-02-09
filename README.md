# 실사 전신 인물 → 반실사 애니 피규어 스타일 3D 미니어처

**3D Vision / 3DGS 기반** · Python · Google Colab (T4 GPU)

---

## 한 줄 요약

실사 전신 이미지를 입력으로 **3DGS로 실사 비율의 3D 복원**을 한 뒤, **geometry는 그대로 두고 렌더링 스타일만** 적용하여 **일본 애니메이션 피규어/굿즈 같은 반실사(stylized realism) 3D 미니어처**를 생성하는 파이프라인. (캐릭터화/SD·치비 템플릿이 아님.)

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
│  Train    │  → 3DGS checkpoint (실사 형상 + 의상 복원)                  │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Stage 4  │  Stylized Rendering (appearance만 변경)                    │
│  Style    │  checkpoint → 셀 셰이딩·색상 양자화 → 스타일화 이미지/영상  │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
[파이프라인 출력] 반실사 애니 피규어 스타일 렌더 결과 (이미지/턴테이블)

  ※ Colab GPU 있을 때 고품질 렌더, 없을 때는 --dummy 로 파이프라인·파라미터 확인 가능.
```

**핵심:** geometry는 Stage 2c까지로 확정. Stage 4는 **렌더 스타일만** 적용(셀 셰이딩, 색 단순화). Blender/PLY 템플릿/프로시저얼 얼굴 없음. 결과는 눈으로 확인 가능한 렌더링 출력.

---

## 프로젝트 목표

| 항목 | 내용 |
|------|------|
| **입력** | 동일 인물 전신 이미지 20~40장, 다양한 시점 |
| **핵심 처리** | 3DGS 복원(2a~2c) → **Stage 4에서 appearance만 스타일화** (실사 형상 유지) |
| **출력** | 반실사 애니 피규어/굿즈 느낌의 3D 미니어처 (렌더 이미지·영상) |
| **환경** | Python, Google Colab (T4). GPU 없을 때는 더미 실행으로 튜닝·파이프라인 확인 |

---

## 구현 순서 권장

파이프라인이 순서대로 돌아가려면 아래 순서로 구현하는 것을 권장한다.

| 순서 | 구현 대상 | 이유 |
|------|-----------|------|
| 1 | `pipeline/stage1_preprocess.py` | Stage 2 입력용 전처리 이미지. |
| 2 | Stage 2a: `pipeline/stage2_reconstruct.py` | Human shape prior (canonical mesh + synthetic cameras). |
| 3 | Stage 2b: `gs/init_3dgs.py` | 3DGS 초기화. point_cloud.npy, cameras.json. |
| 4 | Stage 2c: `gs/train_3dgs.py` | 3DGS 학습 → checkpoint (실사 형상 복원). |
| 5 | `gs/render_character.py` + `pipeline/stage4_style.py` | **Stage 4:** checkpoint → 셀 셰이딩·색상 양자화 → 스타일화 렌더 출력. |

**정리:** 파이프라인 = Stage 1 → 2a → 2b → 2c → Stage 4(스타일화 렌더). 결과는 렌더 이미지를 눈으로 확인하며 `--cel-bands`, `--color-levels` 등으로 튜닝 가능.

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

- **동일 인물 + 동일 의상 + 여러 각도(20~40장)** → **몸과 옷이 함께** 3D로 복원되고, Stage 4에서 스타일화 렌더를 적용하면 **옷 입은 반실사 미니어처**가 된다.
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

### Stage 4 — Stylized Rendering (파이프라인 최종 단계)

| 구분 | 내용 |
|------|------|
| **목적** | Stage 2c 3DGS checkpoint를 **그대로** 사용해, **렌더 스타일만** 적용하여 반실사 애니 피규어 느낌의 결과 생성 |
| **입력** | 3DGS checkpoint (`data/gs_checkpoints/`), `data/gs_output/cameras.json` |
| **처리** | Diffuse 중심 렌더 → 셀 셰이딩(명암 양자화) → 색상 단순화(quantization, saturation/contrast). geometry 수정 없음. |
| **출력** | 스타일화된 렌더 이미지 (예: `output/stylized/`). GPU 없을 때는 `--dummy` 로 placeholder에 스타일 적용해 파이프라인·파라미터 확인 |
| **구현** | `pipeline/stage4_style.py` (내부에서 `gs/render_character.py` 호출) |
| **튜닝** | `--cel-bands`, `--color-levels`, `--saturation`, `--contrast` 로 눈으로 확인하며 조정 가능 |

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
│   └── STAGE2_HUMAN_PRIOR.md  # Stage 2 설계: 2a/2b/2c, geometry 기준
│
├── gs/                      # 3DGS·렌더
│   ├── init_3dgs.py         # Stage 2b: 3DGS 초기화 전용
│   ├── train_3dgs.py        # Stage 2c: 3DGS 학습
│   └── render_character.py  # Stage 4: 스타일화 렌더 (셀 셰이딩·색상 양자화)
│
├── pipeline/                # Stage 진입점 (순차 실행)
│   ├── stage1_preprocess.py # Stage 1
│   ├── stage2_reconstruct.py # Stage 2a: human shape prior (no COLMAP)
│   └── stage4_style.py      # Stage 4
│
├── output/
│   ├── stylized/             # Stage 4 스타일화 렌더 이미지
│   └── videos/               # 턴테이블 mp4 등 (선택)
│
└── README.md
```

- **pipeline/** : Stage 1 → 2a → 2b → 2c → Stage 4.
- **gs/** : 3DGS 초기화·학습·Stage 4 스타일화 렌더.
- **output/stylized/** : Stage 4 출력 (스타일화된 렌더 이미지).

---

## Stage 4 스타일 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--cel-bands` | 3 | 셀 셰이딩 단계 수 (2~4). 클수록 명암 계단이 많음. |
| `--color-levels` | 6 | RGB 양자화 단계. 작을수록 플라스틱 피규어 느낌. |
| `--saturation` | 1.0 | 채도. |
| `--contrast` | 1.05 | 대비. |

렌더 결과를 눈으로 보면서 위 파라미터를 조정해 원하는 반실사 톤을 맞출 수 있다.

---

## MVP 성공 조건

- [ ] 전신 인물 사진 → 3DGS 복원 성공 (Stage 1~2c)
- [ ] Stage 4 스타일화 렌더 출력 (셀 셰이딩·색상 단순화)
- [ ] **실사 형상 유지** + 애니 피규어/굿즈 느낌의 시각 결과
- [ ] GPU 없을 때 `--dummy` 로 파이프라인·파라미터 확인 가능

---

## 설계 원칙

- **Geometry 유지, Appearance만 변경** — Stage 4는 렌더 스타일만 적용. PLY/템플릿 생성 없음.
- **Blender 미사용** — 모든 작업은 Python + 3DGS + 렌더 코드로 처리.
- **재현 가능한 파이프라인** — 끝까지 동작하는 흐름 우선. Colab GPU 있을 때 고품질, 없을 때 더미 허용.
- **파라미터 조절** — --cel-bands, --color-levels 등으로 눈으로 확인하며 튜닝

---

## 최종 결과 정의

> “실사 전신 인물을 촬영하면, 3DGS로 실사 비율의 3D를 복원하고, 렌더 스타일만 바꿔 **반실사 애니메이션 피규어/굿즈 스타일 3D 미니어처**를 만드는 3D 비전 시스템”
