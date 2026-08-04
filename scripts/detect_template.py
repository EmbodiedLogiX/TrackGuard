from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODEL_BY_TEMPLATE = {
    "qwen3_vl_nothink": "checkpoints/Qwen3-VL-8B-Instruct",
    "qwen3_vl": "checkpoints/Qwen3-VL-8B-Instruct",
    "qwen2_vl": "checkpoints/Qwen2.5-VL-3B-Instruct",
}

TEMPLATE_PREFERENCE = ("qwen3_vl_nothink", "qwen3_vl", "qwen2_vl")

PRESET_MODELS = {
    "qwen25_3b": ("checkpoints/Qwen2.5-VL-3B-Instruct", "qwen2_vl"),
    "qwen3_2b": ("checkpoints/Qwen3-VL-2B-Instruct", "qwen3_vl_nothink"),
    "qwen3_8b": ("checkpoints/Qwen3-VL-8B-Instruct", "qwen3_vl_nothink"),
    "gemma3_4b": ("checkpoints/gemma-3-4b-it", "gemma3"),
}


def list_templates() -> set[str]:
    try:
        from llamafactory.data.template import TEMPLATES
        return set(TEMPLATES.keys())
    except Exception:
        pass
    try:
        from llamafactory.data import template as tpl_mod
        return set(getattr(tpl_mod, "TEMPLATES", {}).keys())
    except Exception as e:
        print(f"cannot import LLaMA-Factory: {e}", file=sys.stderr)
        return set()


def pick_template(available: set[str]) -> str:
    for name in TEMPLATE_PREFERENCE:
        if name in available:
            return name
    vl_like = sorted(n for n in available if "qwen" in n and "vl" in n)
    if vl_like:
        return vl_like[0]
    raise SystemExit(
        "no Qwen-VL template found. try: pip install -U llamafactory transformers\n"
        f"templates containing 'qwen': {sorted(n for n in available if 'qwen' in n)}"
    )


def resolve(preset: str | None = None) -> tuple[str, str]:
    import os
    preset = preset or os.environ.get("MODEL_PRESET", "")
    if preset in PRESET_MODELS:
        model_path, template = PRESET_MODELS[preset]
        available = list_templates()
        if template not in available:
            template = pick_template(available)
        return template, model_path

    available = list_templates()
    template = pick_template(available)
    model_path = MODEL_BY_TEMPLATE.get(template, PRESET_MODELS["qwen25_3b"][0])
    if not Path(model_path).exists():
        for _, (p, _) in PRESET_MODELS.items():
            if Path(p).exists():
                model_path = p
                break
    return template, model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recommend", action="store_true")
    parser.add_argument("--preset", type=str, default="")
    args = parser.parse_args()
    template, model_path = resolve(args.preset or None)
    if args.recommend:
        print(f"{template},{model_path}")
    else:
        print(template)


if __name__ == "__main__":
    main()
