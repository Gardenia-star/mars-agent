# llm_service.py
# -*- coding: utf-8 -*-
"""
PyQt5 桌面端的大模型分析线程。

优先走 llm_api_client —— 统一的 OpenAI 兼容多模态接口，可对接
DeepSeek(deepseek-flash) / OpenAI(gpt-4o) / 通义千问(qwen-vl) / 智谱(glm-4v) 等云端 API，
也支持把 provider 设为 ollama 走本地部署。

仅当大模型 API 完全未配置时，才回退到旧的 `ollama` Python 包。
"""

import io
import os
import sys

from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal

# 让 qt_interface/ 能 import 到上一级的共享客户端
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

# 专家级提示词（告诉模型它是工业专家，以及如何按颜色判断温度）
PROMPT = (
    "你是一名铝电解槽工况诊断专家。请分析这张火眼（FireEye）区域的图片。\n"
    "已知规则：\n"
    "1. 颜色偏红、亮白表示温度极高（Hot/过热），可能导致结壳速率低。\n"
    "2. 颜色暗沉、发黑表示温度偏低（Cold/过冷）。\n"
    "3. 颜色橘红均匀表示正常（Normal）。\n"
    "\n"
    "请按照以下格式输出分析报告（不要废话）：\n"
    "【视觉特征】：描述颜色分布、火焰形态、晶核边界清晰度。\n"
    "【工况判断】：判断是过热、过冷还是正常，并给出置信理由。\n"
    "【操作建议】：针对该状态给出具体的电压或加料调整建议。\n"
    "所有的回答都用中文回答"
)


def backend_summary() -> str:
    """给 UI 显示的一句话后端说明。"""
    if analyze_image is None:
        return f"未加载 API 客户端（{_API_IMPORT_ERROR}）"
    cfg = load_config()
    problem = config_problem(cfg)
    if problem:
        return f"未配置（{cfg.get('provider')}）→ 请在项目根目录的 .env 里填写 API Key"
    return describe(cfg)


class LLMWorker(QThread):
    """
    独立的线程用于调用大模型，防止卡死 UI
    """

    finished_signal = pyqtSignal(str)  # 成功信号，返回分析文本
    error_signal = pyqtSignal(str)     # 失败信号

    def __init__(self, roi_qimage, model_name="llava"):
        super().__init__()
        self.roi_qimage = roi_qimage
        self.model_name = model_name
        self.backend = backend_summary()

    def qimage_to_bytes(self, qimg):
        """将 QImage 转换为 Bytes 供大模型读取"""
        try:
            # 1. QImage -> PIL Image
            buffer = qimg.bits().asstring(qimg.byteCount())
            if qimg.format() == 13:  # Format_RGB888
                pil_img = Image.frombuffer("RGB", (qimg.width(), qimg.height()), buffer, "raw", "RGB", 0, 1)
            elif qimg.format() == 4:  # Format_RGB32
                pil_img = Image.frombuffer("RGBA", (qimg.width(), qimg.height()), buffer, "raw", "BGRA", 0, 1).convert("RGB")
            else:
                # 通用兜底：先保存成 buffer 再读取
                byte_array = io.BytesIO()
                qimg.save(byte_array, format="JPEG")
                return byte_array.getvalue()

            # 2. PIL Image -> Bytes
            byte_io = io.BytesIO()
            pil_img.save(byte_io, format='JPEG')
            return byte_io.getvalue()
        except Exception as e:
            print(f"图片转换失败: {e}")
            return None

    # ---------- 回退路径：旧的本地 ollama Python 包 ----------
    def _analyze_via_ollama_pkg(self, img_bytes):
        try:
            import ollama
        except ImportError as e:
            raise LLMCallError(
                "未配置云端大模型 API，且本机没有安装 ollama Python 包。\n"
                "请任选其一：\n"
                "  1) 在 config/llm_api.json 里配置云端 API（推荐，例如 DeepSeek）；\n"
                "  2) pip install ollama 并本机运行 `ollama serve` + `ollama pull llava`。"
            ) from e

        response = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': PROMPT, 'images': [img_bytes]}],
        )
        return response['message']['content']

    def run(self):
        try:
            img_bytes = self.qimage_to_bytes(self.roi_qimage)
            if not img_bytes:
                self.error_signal.emit("图片格式转换失败")
                return

            # 首选：统一的多模态 API 客户端（云端 API，或本地 Ollama 的 /v1）
            if analyze_image is not None:
                cfg = load_config()
                problem = config_problem(cfg)
                if problem:
                    # 配置不全（例如还没填 API Key）→ 直接给出可操作的报错，不静默回退
                    raise LLMConfigError(problem)
                self.finished_signal.emit(analyze_image(img_bytes, PROMPT, config=cfg))
                return

            # 仅当共享客户端不可用时，才回退到旧的本地 ollama 包
            self.finished_signal.emit(self._analyze_via_ollama_pkg(img_bytes))

        except (LLMConfigError, LLMCallError) as e:
            self.error_signal.emit(str(e))
        except Exception as e:
            self.error_signal.emit(f"大模型调用失败: {e}")
