"""
데포르메 규칙 정의 (JSON / Python dict).

Stage 3에서 템플릿에 적용할 비율/스케일 조건을 정리할 때 참조.
룩업 피규어(애니 피규어) 스타일: 비율·디테일 수준 설정.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# 규칙 스키마: Stage 3 데포르메에서 참조하는 키들
# 모든 파라미터는 코드 상단에서 조절 가능 (프로젝트 요구사항)
# -----------------------------------------------------------------------------

# 룩업 피규어 스타일 기본값 (애니메이션 피규어 느낌)
DEFAULT_HEAD_SCALE = 1.35
DEFAULT_ARM_LENGTH = 0.88
DEFAULT_LEG_LENGTH = 0.88
DEFAULT_TORSO_SCALE = 0.92
DEFAULT_HAND_SCALE = 0.85
DEFAULT_FOOT_SCALE = 0.88
# 디테일 수준: "figurine" = 피규어 수준만 (얼굴/머리/의상 디테일 과하지 않음)
DEFAULT_DETAIL_LEVEL = "figurine"

# 허용 범위 (검증용)
SCALE_MIN = 0.5
SCALE_MAX = 2.0
DETAIL_LEVELS = ("figurine", "simplified", "realistic")


def get_default_rules() -> dict[str, Any]:
    """
    기본 데포르메 규칙: 룩업 피규어(애니메이션 피규어) 스타일.
    실사 디테일이 아닌 피규어 수준의 비율·디테일만 적용.
    """
    return {
        "name": "lookup_figure",
        "description": "룩업 피규어 / 애니메이션 피규어 스타일. 얼굴·머리·의상 디테일은 피규어 수준.",
        # ---- 비율 (기하 변형) ----
        "head_scale": DEFAULT_HEAD_SCALE,
        "arm_length": DEFAULT_ARM_LENGTH,
        "leg_length": DEFAULT_LEG_LENGTH,
        "torso_scale": DEFAULT_TORSO_SCALE,
        "hand_scale": DEFAULT_HAND_SCALE,
        "foot_scale": DEFAULT_FOOT_SCALE,
        # ---- 디테일 수준 (Stage 3/4 참고용: 실사 디테일 유지 여부) ----
        "detail_level": DEFAULT_DETAIL_LEVEL,
        # ---- 스타일 프리셋 이름 (Stage 4에서 참조) ----
        "style": "2.5D_character",
    }


def get_preset(preset_name: str) -> dict[str, Any]:
    """
    프리셋 이름으로 데포르메 규칙 반환.
    - lookup_figure: 룩업/애니 피규어 (기본)
    - chibi: 더 짧은 팔다리, 더 큰 머리
    - nendoroid_style: 넨도로이드 느낌 (머리 크게, 몸 통통)
    """
    presets: dict[str, dict[str, Any]] = {
        "lookup_figure": get_default_rules(),
        "chibi": {
            "name": "chibi",
            "description": "챠비 스타일. 머리 더 크게, 팔다리 더 짧게.",
            "head_scale": 1.5,
            "arm_length": 0.78,
            "leg_length": 0.78,
            "torso_scale": 0.88,
            "hand_scale": 0.82,
            "foot_scale": 0.85,
            "detail_level": "figurine",
            "style": "2.5D_character",
        },
        "nendoroid_style": {
            "name": "nendoroid_style",
            "description": "넨도로이드 스타일. 큰 머리, 통통한 몸, 짧은 팔다리.",
            "head_scale": 1.45,
            "arm_length": 0.82,
            "leg_length": 0.82,
            "torso_scale": 0.95,
            "hand_scale": 0.8,
            "foot_scale": 0.82,
            "detail_level": "figurine",
            "style": "2.5D_character",
        },
        "simplified": {
            "name": "simplified",
            "description": "단순화 비율만 적용, 디테일은 실사에 가깝게 유지 가능.",
            "head_scale": 1.15,
            "arm_length": 0.92,
            "leg_length": 0.92,
            "torso_scale": 0.95,
            "hand_scale": 0.9,
            "foot_scale": 0.9,
            "detail_level": "simplified",
            "style": "2.5D_character",
        },
    }
    if preset_name not in presets:
        raise ValueError(
            f"Unknown preset: '{preset_name}'. Choose from: {list(presets.keys())}"
        )
    return dict(presets[preset_name])


def _validate_scale(value: float, key: str) -> None:
    if not (SCALE_MIN <= value <= SCALE_MAX):
        raise ValueError(
            f"'{key}' must be in [{SCALE_MIN}, {SCALE_MAX}], got {value}"
        )


def validate_rules(rules: dict[str, Any]) -> dict[str, Any]:
    """
    규칙 dict 검증 및 기본값 채우기.
    필수 키가 없으면 기본값(lookup_figure)으로 채움.
    """
    default = get_default_rules()
    out = {**default, **rules}

    for key in ("head_scale", "arm_length", "leg_length", "torso_scale", "hand_scale", "foot_scale"):
        val = out.get(key, default[key])
        if not isinstance(val, (int, float)):
            raise TypeError(f"'{key}' must be a number, got {type(val).__name__}")
        _validate_scale(float(val), key)

    detail = out.get("detail_level", default["detail_level"])
    if detail not in DETAIL_LEVELS:
        raise ValueError(
            f"'detail_level' must be one of {DETAIL_LEVELS}, got '{detail}'"
        )
    out["detail_level"] = detail

    return out


def load_from_json(path: str | Path) -> dict[str, Any]:
    """JSON 파일에서 데포르메 규칙 로드 후 검증."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return validate_rules(data)


def save_to_json(rules: dict[str, Any], path: str | Path) -> None:
    """데포르메 규칙을 JSON 파일로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(validate_rules(rules), f, ensure_ascii=False, indent=2)


def is_figurine_detail(rules: dict[str, Any]) -> bool:
    """피규어 수준 디테일만 적용할지 여부 (실사 디테일 억제)."""
    return rules.get("detail_level", DEFAULT_DETAIL_LEVEL) == "figurine"


# -----------------------------------------------------------------------------
# 사용 예시 (이 파일을 직접 실행할 때)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    rules = get_default_rules()
    print("Default (lookup_figure) rules:", json.dumps(rules, ensure_ascii=False, indent=2))
    print("is_figurine_detail:", is_figurine_detail(rules))

    # 프리셋 예시
    print("\nPreset 'chibi':", get_preset("chibi")["head_scale"])
