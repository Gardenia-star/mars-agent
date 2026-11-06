# llm_web.py
# -*- coding: utf-8 -*-
"""
Web 端多模态诊断报告生成。

优先走 llm_api_client（OpenAI 兼容接口：DeepSeek / OpenAI / Qwen-VL / GLM-4V ...
也支持本地 Ollama 的 /v1），未配置时才回退到旧的 `ollama` Python 包。
"""

import io
import os
import sys
from typing import Any, Dict

import numpy as np
from PIL import Image

# 让 web_interface/ 能 import 到上一级的共享客户端
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from llm_api_client import (  # noqa: F401
        LLMCallError,
        LLMConfigError,
        analyze_image,
        config_problem,
        describe,
        load_config,
    )

    _API_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover
    analyze_image = None
    _API_IMPORT_ERROR = _e

from algo_web_bridge import CLASS_CN


def backend_summary() -> str:
    """给 UI 显示的一句话后端说明。"""
    if analyze_image is None:
        return f"未加载 API 客户端（{_API_IMPORT_ERROR}）"
    cfg = load_config()
    if config_problem(cfg):
        return f"未配置（{cfg.get('provider')}）→ 请在项目根目录的 .env 里填写 API Key"
    return describe(cfg)


def build_prompt(result: Dict[str, Any]) -> str:
    """
    把算法输出的结果整理成给多模态模型的文字提示。
    """
    cls_code = result.get("cls_result", "N")
    state_cn = CLASS_CN.get(cls_code, "未知")
    gray = result.get("gray_mean", 0)
    contrast = result.get("contrast", 0)
    sat = result.get("saturation", 0)
    vc = result.get("crust_speed", 0)
    vn = result.get("node_speed", 0)

    prompt = (
        "你是一名铝电解槽现场的资深工艺诊断专家，请结合这张火眼( FireEye )区域的图像，以及给出的特征指标，"
        "输出一份结构化、简洁清晰的中文分析报告。\n\n"
        "【已知数值特征】\n"
        f"1. 平均灰度 μ ≈ {gray:.1f}\n"
        f"2. 对比度 σ ≈ {contrast:.1f}\n"
        f"3. 饱和度 S ≈ {sat:.3f}\n"
        f"4. 结壳速率 V_crust ≈ {vc:.3f}\n"
        f"5. 角点速率 V_node ≈ {vn:.3f}\n"
        f"6. 模型判定过热度状态：{state_cn}（代码 {cls_code} ）\n\n"
        "【请按以下格式输出分析报告】（不要废话）：\n"
        "一、图像视觉特征简要点评\n"
        "  - 从亮度、颜色、纹理分布三方面，用 2~4 条要点描述画面特征。\n\n"
        "二、过热度状态判断\n"
        "  - 明确说明当前属于偏高/偏低/正常中的哪一种，并给出 1~2 句理由。\n\n"
        "三、工艺风险提示\n"
        "  - 用 2~3 条项目符号列出可能的工艺风险（如结壳不稳、闷槽风险、阳极效应诱因等）。\n\n"
        "四、操作建议\n"
        "  - 用 3~5 条具体、可执行的调节建议（例如微调槽电压、调整加料节奏、安排人工巡检等），尽量量化区间。\n\n"
        "所有内容请使用中文回答，分段清晰，适合直接复制到班组汇报 PPT。"
    )
    return prompt


def pil_from_bgr(frame_bgr: np.ndarray) -> Image.Image:
    """BGR numpy → PIL RGB"""
    rgb = frame_bgr[:, :, ::-1]
    return Image.fromarray(rgb)


def _jpeg_bytes(frame_bgr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    pil_from_bgr(frame_bgr).save(buf, format="JPEG")
    return buf.getvalue()


def _via_ollama_pkg(frame_bgr: np.ndarray, prompt: str, model: str) -> str:
    try:
        import ollama
    except ImportError as e:
        raise LLMCallError(
            "未配置云端大模型 API，且本机没有安装 ollama Python 包。\n"
            "请任选其一：\n"
            "  1) 在 config/llm_api.json 里配置云端 API（推荐，例如 DeepSeek）；\n"
            "  2) pip install ollama 并本机运行 `ollama serve` + `ollama pull llava`。"
        ) from e

    resp = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": _jpeg_bytes(frame_bgr)},
                ],
            }
        ],
    )
    return resp["message"]["content"]


def llava_analyze_frame(frame_bgr: np.ndarray, result: Dict[str, Any], model: str = "llava") -> str:
    """
    对当前 ROI 图像 + 数值特征生成诊断报告。

    函数名保留以兼容原调用方（streamlit_app.py 里叫 “AI 诊断”）。
    """
    prompt = build_prompt(result)

    if analyze_image is not None:
        cfg = load_config()
        problem = config_problem(cfg)
        if problem:
            raise LLMConfigError(problem)
        return analyze_image(_jpeg_bytes(frame_bgr), prompt, config=cfg)

    return _via_ollama_pkg(frame_bgr, prompt, model)
