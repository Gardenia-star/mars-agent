# llm_api_client.py
# -*- coding: utf-8 -*-
"""
MARS-Agent 多模态大模型客户端 —— 统一的 OpenAI 兼容 /chat/completions 接口。

一份代码同时支持：
  * 云端多模态 API：DeepSeek(deepseek-flash) / OpenAI(gpt-4o) / 通义千问(qwen-vl)
    / 智谱(glm-4v) / Kimi / SiliconFlow 等任何 OpenAI 兼容端点；
  * 本地 Ollama：它的 /v1 同样是 OpenAI 兼容接口（模型如 llava）。

因此不再强制依赖 `ollama` 这个 Python 包，也不再要求本地部署。

配置优先级：环境变量  >  config/llm_api.json  >  内置默认值。
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "llm_api.json")
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
ENV_PREFIX = "MARS_LLM_"

# ---- 常见厂商的 OpenAI 兼容端点预设（把 provider 设成这些名字即可） ----
PRESETS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",          # 目前唯一支持视觉的 DeepSeek 模型
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "qwen": {                               # 阿里云百炼（中国北京）兼容模式
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "qwen-intl": {                          # 阿里云百炼 国际站（新加坡）
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "zhipu": {                              # 智谱 GLM-4V
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4v-plus",
        "api_key_env": "ZHIPUAI_API_KEY",
    },
    "moonshot": {                           # Kimi 视觉预览版
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k-vision-preview",
        "api_key_env": "MOONSHOT_API_KEY",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "api_key_env": "SILICONFLOW_API_KEY",
    },
    "ollama": {                             # 本地 Ollama 的 OpenAI 兼容层
        "base_url": "http://localhost:11434/v1",
        "model": "llava",
        "api_key_env": "",
    },
    "custom": {                             # 完全自定义，请自行填 base_url / model
        "base_url": "",
        "model": "",
        "api_key_env": "MARS_LLM_API_KEY",
    },
}

DEFAULTS: Dict[str, Any] = {
    "provider": "deepseek",
    "enabled": True,
    "timeout": 180,          # 秒
    "max_tokens": 2048,
    "temperature": 0.3,
    "detail": "",            # ""=不发送 detail 字段（兼容性最好）；可填 low/high/original/auto
    "system_prompt": "你是一名长期驻厂的资深铝电解工艺工程师。",
}

# 默认的专家级中文提示词（原本躺在 algo_fireeye.py 里从未被调用）
DEFAULT_PROMPT = """你是一名长期驻厂的铝电解工艺工程师，熟悉电解质过热度识别、火眼视频诊断。

现在给你一帧来自“火眼监控”的 ROI 图像，图像已通过目标检测裁剪，主要包含电解质表面和火焰区域。
请根据图像进行诊断分析，并**只用简体中文回答**，严格按以下结构输出（标题保留）：

【一、图像整体描述】
- 用 2~3 句话描述当前火眼画面的整体情况（亮度、颜色分布、是否有明显结壳块、黑色冷料等）。

【二、静态特征评估】
结合亮度和纹理，对下列指标给出定性判断（偏低/正常/偏高）并解释原因（颜色、纹理、细节清晰度）：
- 平均灰度 μ_x（电解质表面整体明暗）
- 对比度 σ_x（灰度变化，是否“花”“杂乱”）
- 饱和度 s_x（彩色成分，尤其红/黄色强度）

【三、动态特征与过热度判断】
根据火焰形态、亮区位置、边缘清晰度，判断当前过热度状态（偏低/正常/偏高），并说明：
- 依据了哪些局部特征；
- 结壳速率 V_crust 可能偏高、正常还是偏低；
- 角点速率 V_node 的可能变化趋势（结壳是否在破裂/剥落）。

【四、风险等级与工艺建议】
- 给出风险等级（低/中/高）；
- 提出不超过 3 条可落地的工艺建议（是否微调槽电压及幅度区间、加料节奏/捞渣/阳极检查、是否加强监测）。

【五、需要持续观察的现象】
- 列出 1~2 个未来 5~10 分钟需重点关注的现象。

要求：全程使用专业但易懂的简体中文；不要出现“作为 AI 模型”“我只能”之类措辞；
若信息不足请说明假设条件再给出慎重建议。"""


class LLMConfigError(RuntimeError):
    """配置缺失或不合法。"""


class LLMCallError(RuntimeError):
    """接口调用失败。"""


def _parse_env_file(path: str) -> Dict[str, str]:
    """
    极简 .env 解析（不依赖 python-dotenv）：
    支持 KEY=VALUE、# 注释、export 前缀、单/双引号、行尾注释。
    """
    data: Dict[str, str] = {}
    if not os.path.isfile(path):
        return data
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]
                elif " #" in val:                      # 去掉行尾注释
                    val = val.split(" #", 1)[0].strip()
                if key:
                    data[key] = val
    except Exception as e:
        print(f"[llm_api] 解析 {path} 失败：{e}")
    return data


def load_env_file(path: Optional[str] = None) -> Dict[str, str]:
    """
    把 .env 的内容注入 os.environ。

    已存在的真实环境变量**优先**（不会被 .env 覆盖），
    因此 `export DASHSCOPE_API_KEY=xxx` 永远比 .env 里的值更强。
    返回本次从文件里读到的键值对（便于调试）。
    """
    data = _parse_env_file(path or ENV_FILE)
    for k, v in data.items():
        if k not in os.environ:
            os.environ[k] = v
    return data


def _read_config_file() -> Dict[str, Any]:
    if not os.path.isfile(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:  # 配置写坏了也不要让整个程序起不来
        print(f"[llm_api] 读取 {CONFIG_PATH} 失败：{e}")
        return {}


def _env_overrides() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    mapping = {
        "PROVIDER": "provider",
        "BASE_URL": "base_url",
        "MODEL": "model",
        "API_KEY": "api_key",
        "TIMEOUT": "timeout",
        "MAX_TOKENS": "max_tokens",
        "DETAIL": "detail",
        "TEMPERATURE": "temperature",
    }
    for suffix, key in mapping.items():
        val = os.environ.get(ENV_PREFIX + suffix)
        if val not in (None, ""):
            out[key] = val
    en = os.environ.get(ENV_PREFIX + "ENABLED")
    if en not in (None, ""):
        out["enabled"] = en.strip().lower() in ("1", "true", "yes", "on")
    return out


def load_config() -> Dict[str, Any]:
    """合并：默认值 <- provider 预设 <- 配置文件 <- .env <- 真实环境变量。"""
    load_env_file()
    file_cfg = _read_config_file()
    provider = str(
        os.environ.get(ENV_PREFIX + "PROVIDER")
        or file_cfg.get("provider")
        or DEFAULTS["provider"]
    ).strip().lower()

    cfg: Dict[str, Any] = dict(DEFAULTS)
    cfg["provider"] = provider
    cfg.update(PRESETS.get(provider, {}))
    cfg.update({k: v for k, v in file_cfg.items() if v is not None})
    cfg.update(_env_overrides())

    # 数字类型收敛
    for key, caster in (("timeout", int), ("max_tokens", int), ("temperature", float)):
        try:
            cfg[key] = caster(cfg.get(key))
        except (TypeError, ValueError):
            cfg[key] = DEFAULTS[key]
    return cfg


def resolve_api_key(cfg: Optional[Dict[str, Any]] = None) -> str:
    """按 显式配置 -> 环境变量 -> 常见厂商环境变量 的顺序找 key。"""
    load_env_file()
    cfg = cfg or load_config()
    key = str(cfg.get("api_key") or "").strip()
    if key:
        return key
    for name in (ENV_PREFIX + "API_KEY", cfg.get("api_key_env") or ""):
        if name:
            val = os.environ.get(name, "").strip()
            if val:
                return val
    return ""


def config_problem(cfg: Optional[Dict[str, Any]] = None) -> str:
    """返回人类可读的配置问题；空串表示配置可用。"""
    cfg = cfg or load_config()
    if not cfg.get("enabled"):
        return "配置中 enabled=false，大模型分析已关闭。"
    if not str(cfg.get("base_url") or "").strip():
        return f"provider='{cfg.get('provider')}' 没有 base_url，请在 config/llm_api.json 里填写。"
    if not str(cfg.get("model") or "").strip():
        return f"provider='{cfg.get('provider')}' 没有 model，请在 config/llm_api.json 里填写。"
    # 本地 Ollama 不需要 key
    if cfg.get("provider") == "ollama":
        return ""
    if not resolve_api_key(cfg):
        return (
            "未找到 API Key。请任选一种方式配置：\n"
            f"  1) 写入 {ENV_FILE}\n"
            f"     （推荐）内容形如： {cfg.get('api_key_env') or 'MARS_LLM_API_KEY'}=sk-xxxxxxxx\n"
            f"  2) 环境变量 export {cfg.get('api_key_env') or 'MARS_LLM_API_KEY'}=<你的Key>\n"
            f"  3) 写入 {CONFIG_PATH} 的 \"api_key\" 字段"
        )
    return ""


def describe(cfg: Optional[Dict[str, Any]] = None) -> str:
    cfg = cfg or load_config()
    return f"{cfg.get('provider')} | {cfg.get('model')} | {cfg.get('base_url')}"


def encode_image_bytes(data: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def analyze_image(
    image_bytes: bytes,
    prompt: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    mime: str = "image/jpeg",
) -> str:
    """
    调用多模态大模型分析一张图片。

    :param image_bytes: JPEG/PNG 等图片的原始字节
    :param prompt: 提示词，默认使用 DEFAULT_PROMPT（专家诊断模板）
    :param config: 可选，直接传入已加载的配置
    :return: 模型返回的中文文本
    """
    cfg = config or load_config()
    problem = config_problem(cfg)
    if problem:
        raise LLMConfigError(problem)

    try:
        import requests
    except ImportError as e:  # pragma: no cover
        raise LLMCallError("缺少 requests 库，请先执行: pip install requests") from e

    if not image_bytes:
        raise LLMCallError("没有可用的图像数据。")

    url = str(cfg["base_url"]).rstrip("/") + "/chat/completions"
    image_block: Dict[str, Any] = {"url": encode_image_bytes(image_bytes, mime)}
    if str(cfg.get("detail") or "").strip():
        image_block["detail"] = str(cfg["detail"]).strip()

    payload: Dict[str, Any] = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": cfg.get("system_prompt") or ""},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or DEFAULT_PROMPT},
                    {"type": "image_url", "image_url": image_block},
                ],
            },
        ],
        "max_tokens": cfg["max_tokens"],
        "temperature": cfg["temperature"],
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    key = resolve_api_key(cfg)
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=cfg["timeout"])
    except Exception as e:
        raise LLMCallError(
            f"无法连接大模型接口 {url}：{e}\n"
            "请检查网络、base_url 是否正确；若用本地 Ollama，请确认 `ollama serve` 已启动。"
        ) from e

    if resp.status_code != 200:
        raise LLMCallError(
            f"接口返回 HTTP {resp.status_code}。\n响应内容：{resp.text[:800]}\n"
            "常见原因：API Key 无效/欠费、模型名不支持视觉、base_url 写错。"
        )

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise LLMCallError(f"无法解析接口响应：{resp.text[:800]}") from e

    # 少数实现会把 content 返回成分段列表
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict):
                parts.append(str(blk.get("text") or blk.get("content") or ""))
            else:
                parts.append(str(blk))
        content = "".join(parts)

    return str(content or "").strip()


__all__ = [
    "analyze_image",
    "load_config",
    "load_env_file",
    "resolve_api_key",
    "config_problem",
    "describe",
    "encode_image_bytes",
    "config_path",
    "env_file",
    "LLMConfigError",
    "LLMCallError",
    "DEFAULT_PROMPT",
    "PRESETS",
]

config_path = CONFIG_PATH
env_file = ENV_FILE
