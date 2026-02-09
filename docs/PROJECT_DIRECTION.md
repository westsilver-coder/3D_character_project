# 프로젝트 방향 정리 및 수정·로드맵

## 1. 반드시 수정이 필요한 파일 (최소 목록)

| 파일 | 기존 역할 | 변경된 역할 |
|------|-----------|-------------|
| **README.md** | 실사 복원 → 데포르메 → 스타일 적용으로 “캐릭터화된 3D 인물” 생성 목표 기술 | 목표를 “템플릿(마네킹) 위에 사람 정보(체형·색·의상)를 입혀 귀엽고 일관된 애니/피규어 스타일 3D 캐릭터 생성”으로 정정. Stage 3/4 역할 변경 반영. |
| **pipeline/stage3_deform.py** | 3DGS geometry를 데포르메 규칙으로 변형해 “캐릭터 비율 3DGS” 출력 | 템플릿에 적용할 **비율/스케일 조건**을 정리하는 단계로 역할 축소 (최종 캐릭터 geometry 직접 생성 아님). |
| **pipeline/stage4_style.py** | 기하 유지한 채 Gaussian color/SH 등으로 외형 스타일만 적용 | **선택한 캐릭터 템플릿(PLY) 로드** 후, Stage 2/3에서 얻은 사람 정보(체형·색·의상)를 **템플릿 위에 매핑**해 애니메이션 스타일 캐릭터 geometry/appearance 생성. |
| **character/style_presets.py** | Gaussian·렌더링용 스타일 프리셋(2.5D, 셀쉐이딩, 파스텔 등) 제공 | **템플릿 선택**(SD chibi / 동숲 / 룩업 등)을 명시적 옵션으로 추가하고, 각 프리셋에 대응하는 템플릿 경로·매핑 옵션 제공. 기존 셀/파스텔 등은 템플릿 렌더 시 적용할 appearance 옵션으로 유지 가능. |
| **character/deformation_rules.py** | 데포르메 규칙(비율·디테일) 정의, Stage 3에서 “캐릭터 3DGS” 생성에 사용 | 규칙은 **템플릿에 적용할 비율/스케일 조건**으로만 사용됨을 docstring(또는 주석)으로 명시. 스키마/기본값은 유지. |
| **gs/deform_geometry.py** | 3DGS geometry를 데포르메 규칙에 따라 변형 | 출력이 “최종 캐릭터 3DGS”가 아니라 **Stage 4에서 템플릿에 적용할 조건(비율/스케일 등)** 로 쓰임을 docstring으로 명시. |

- **수정하지 않는 것**: Stage 1, Stage 2a/2b/2c, Stage 5, `init_3dgs.py`, `train_3dgs.py`, `stage2_reconstruct.py` 등 기존 파이프라인 코드는 구조 유지.  
- **추가 권장**: `character/templates/` 에 있는 템플릿(예: `sd_chibi_base.ply`)을 Stage 4에서 참조할 수 있도록 경로/이름 규약을 README 또는 이 문서에 한 줄 정도 명시.

---

## 2. 설계 원칙 (문서에 반영할 내용)

- **실사 눈/코/입**은 직접 사용하지 않음. 얼굴 요소는 **템플릿 스타일에 맞게 새로 생성**.
- **귀여움·일관성** 최우선. 언캐니 밸리가 나올 수 있는 실사 디테일은 제거.
- **템플릿 선택**(SD / 동숲 / 룩업 등)은 **명시적 옵션**이어야 함.
- Stage 2c 출력(3DGS checkpoint)은 “최종 캐릭터”가 아니라 **템플릿에 적용할 condition 정보**로 사용.

---

## 3. 앞으로 구현할 핵심 단계 (Stage 4 중심 로드맵)

```
[현재까지]
  Stage 1 → 2a → 2b → 2c → (Stage 3: 조건 정리) → [Stage 4: 미구현] → Stage 5

[Stage 4 구현 순서]
```

1. **템플릿 로드 및 규격 정의**
   - `character/templates/*.ply` (예: `sd_chibi_base.ply`) 로드.
   - 템플릿 메쉬의 좌표계·비율·실루엣 규격을 문서/코드로 정의 (Blender 제작 규격과 일치).

2. **Stage 2/3 출력과의 데이터 계약 정리**
   - Stage 2c checkpoint / Stage 3에서 내보내는 “조건” 형식 정의 (체형 비율, 스케일, 색/의상 정보 등).
   - 이 조건이 템플릿의 어떤 요소(본/메쉬 영역, UV, 머티리얼 등)에 대응하는지 매핑 테이블(또는 설정) 설계.

3. **매핑 모듈 (Stage 4 핵심)**
   - 입력: (1) 선택된 템플릿 PLY, (2) Stage 2/3 조건(비율·스케일·색·의상 등).
   - 처리: 템플릿 geometry에 비율/스케일 적용, 색/의상 정보를 템플릿 표면(또는 머티리얼/텍스처)에 매핑.
   - 출력: 애니메이션 스타일 캐릭터용 geometry(+ appearance) 표현. (3DGS로 유지할지, 메쉬+텍스처로 할지는 구현 선택.)

4. **템플릿 선택 옵션 연동**
   - `character/style_presets.py`(또는 동등 설정)에서 템플릿 이름(예: `sd_chibi`, `동숲`, `lookup`)과 `character/templates/*.ply` 경로 연결.
   - Stage 4 진입 시 “어떤 템플릿 쓸지”를 인자/설정으로 받도록 연결.

5. **Stage 4 파이프라인 진입점**
   - `pipeline/stage4_style.py`: (1) 템플릿 선택 읽기, (2) 템플릿 로드, (3) Stage 2/3 조건 로드, (4) 매핑 모듈 호출, (5) 결과 저장(및 필요 시 Stage 5 입력 형식으로 내보내기).

6. **Stage 5와 연결**
   - Stage 4 출력이 기존처럼 “스타일이 입혀진 3D 캐릭터”이므로, Stage 5(턴테이블·피규어 카메라·미니어처 렌더)는 기존 역할 유지. 입출력 포맷만 Stage 4 출력과 맞추면 됨.

---

## 4. 요약

- **최소 수정**: README, `stage3_deform.py`, `stage4_style.py`, `style_presets.py`, `deformation_rules.py`, `deform_geometry.py` — 각각 **역할 설명·목표 정정** 및 Stage 4는 “템플릿 로드 + 사람 정보 매핑”으로 재정의.
- **구조 유지**: Stage 1, 2a/2b/2c, 5와 기존 디렉터리 구조는 붕괴 없이 유지.
- **로드맵**: 템플릿 규격 정의 → Stage 2/3 조건 계약 → 매핑 모듈 구현 → 템플릿 선택 연동 → `stage4_style.py` 구현 → Stage 5 연동.
