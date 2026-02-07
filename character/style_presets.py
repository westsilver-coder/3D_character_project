"""
캐릭터 스타일 프리셋 (2.5D, 셀 셰이딩, 파스텔 등).
Stage 4에서 기하는 유지한 채 외형만 변경할 때 사용.
Gaussian color, SH 계수 조정, 렌더링 후처리 파라미터를 프리셋으로 제공.
"""

from __future__ import annotations

from typing import Any

# -----------------------------------------------------------------------------
# 스타일 파라미터: Stage 4 / gs 렌더에서 참조
# 모든 값은 코드 상단에서 조절 가능 (프로젝트 요구사항)
# -----------------------------------------------------------------------------

# 색감 (0~2 정도, 1=원본)
DEFAULT_SATURATION = 1.0
DEFAULT_CONTRAST = 1.0
# 셀 셰이딩: 단계 수 (0=비활성, 2~4=토온/2.5D)
DEFAULT_CEL_BANDS = 3
# 파스텔: 0=없음, 1=강함
DEFAULT_PASTEL_STRENGTH = 0.0


def get_preset(name: str) -> dict[str, Any]:
    """
    스타일 프리셋 이름으로 설정 dict 반환.
    deformation_rules의 "style" 키와 연동 (예: "2.5D_character").
    """
    presets: dict[str, dict[str, Any]] = {
        "2.5D_character": {
            "name": "2.5D_character",
            "description": "2.5D 게임 캐릭터 느낌. 단순한 색감 + 부드러운 셀 셰이딩.",
            "shading_mode": "cel",
            "cel_bands": 3,
            "saturation": 0.95,
            "contrast": 1.05,
            "pastel_strength": 0.0,
            "simplify_palette": True,
            "sh_scale": 0.7,
            "post_process": {"outline": False, "bloom": False},
        },
        "cel_shading": {
            "name": "cel_shading",
            "description": "선명한 셀 셰이딩 (토온/애니 느낌).",
            "shading_mode": "cel",
            "cel_bands": 4,
            "saturation": 1.0,
            "contrast": 1.15,
            "pastel_strength": 0.0,
            "simplify_palette": True,
            "sh_scale": 0.5,
            "post_process": {"outline": True, "bloom": False},
        },
        "pastel": {
            "name": "pastel",
            "description": "파스텔 톤. 부드럽고 채도 낮은 색감.",
            "shading_mode": "soft",
            "cel_bands": 0,
            "saturation": 0.75,
            "contrast": 0.95,
            "pastel_strength": 0.6,
            "simplify_palette": True,
            "sh_scale": 0.85,
            "post_process": {"outline": False, "bloom": False},
        },
        "simplified": {
            "name": "simplified",
            "description": "단순화된 색감. 피규어/미니어처에 맞게 색 수 줄임.",
            "shading_mode": "soft",
            "cel_bands": 0,
            "saturation": 0.9,
            "contrast": 1.0,
            "pastel_strength": 0.2,
            "simplify_palette": True,
            "sh_scale": 0.8,
            "post_process": {"outline": False, "bloom": False},
        },
        "neutral": {
            "name": "neutral",
            "description": "스타일 최소 적용. 원본에 가깝게.",
            "shading_mode": "soft",
            "cel_bands": 0,
            "saturation": 1.0,
            "contrast": 1.0,
            "pastel_strength": 0.0,
            "simplify_palette": False,
            "sh_scale": 1.0,
            "post_process": {"outline": False, "bloom": False},
        },
    }
    if name not in presets:
        raise ValueError(
            f"Unknown style preset: '{name}'. Choose from: {list(presets.keys())}"
        )
    return dict(presets[name])


def get_default_style() -> dict[str, Any]:
    """기본 스타일: deformation_rules 기본값과 맞춘 2.5D_character."""
    return get_preset("2.5D_character")


def list_presets() -> list[str]:
    """사용 가능한 스타일 프리셋 이름 목록."""
    return [
        "2.5D_character",
        "cel_shading",
        "pastel",
        "simplified",
        "neutral",
    ]


def validate_style(style: dict[str, Any]) -> dict[str, Any]:
    """
    스타일 dict 검증 및 기본값 채우기.
    필수 키가 없으면 2.5D_character 기준으로 채움.
    """
    default = get_default_style()
    out = {**default, **style}

    if out.get("cel_bands", 0) < 0:
        out["cel_bands"] = 0
    if not 0 <= out.get("saturation", 1.0) <= 2.0:
        out["saturation"] = default["saturation"]
    if not 0 <= out.get("pastel_strength", 0.0) <= 1.0:
        out["pastel_strength"] = default["pastel_strength"]

    return out


# -----------------------------------------------------------------------------
# 사용 예시 (이 파일을 직접 실행할 때)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    preset = get_default_style()
    print("Default style (2.5D_character):")
    print(json.dumps(preset, ensure_ascii=False, indent=2))
    print("\nAvailable presets:", list_presets())
