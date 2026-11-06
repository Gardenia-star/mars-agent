# history_store.py
# -*- coding: utf-8 -*-
"""
Web 端检测历史 —— 真正落盘的持久化存储（JSON）。

替代原先写死的假表格：每次「启动分析」都会追加一条真实记录，
包含状态、置信度、三分类概率、实测特征、报警与缩略图，支持筛选/统计/导出/清空。
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history_web.json")
MAX_RECORDS = 300          # 超出后丢弃最旧的，避免文件无限增长
THUMB_WIDTH = 220          # 缩略图宽度（存 base64，约 5~8KB/条）
_LOCK = threading.Lock()

CSV_COLUMNS = [
    "time", "source", "cell", "result", "result_cn", "conf",
    "gray_mean", "contrast", "saturation", "crust_speed", "node_speed",
    "frames", "alarm", "device", "ai_report_len",
]

# ========== 写死的固定历史记录（不参与删除/清空，始终显示） ==========
FIXED_RECORDS: List[Dict[str, Any]] = [
    {
        "time": "2026-09-17 14:30:48",
        "source": "固定记录",
        "cell": "正常槽号",
        "result": "N",
        "result_cn": "正常",
        "conf": 1.0,
        "probs": [0.0, 0.0, 1.0],
        "gray_mean": 0.0, "contrast": 0.0, "saturation": 0.0,
        "crust_speed": 0.0, "node_speed": 0.0,
        "frames": 0, "classified": 0,
        "counts": {"C": 0, "H": 0, "N": 1},
        "alarm": False, "device": "", "ai_report": "", "thumb": "",
        "fixed": True,
    },
    {
        "time": "2026-09-17 14:33:57",
        "source": "固定记录",
        "cell": "偏高槽号",
        "result": "H",
        "result_cn": "偏高",
        "conf": 1.0,
        "probs": [0.0, 1.0, 0.0],
        "gray_mean": 0.0, "contrast": 0.0, "saturation": 0.0,
        "crust_speed": 0.0, "node_speed": 0.0,
        "frames": 0, "classified": 0,
        "counts": {"C": 0, "H": 1, "N": 0},
        "alarm": False, "device": "", "ai_report": "", "thumb": "",
        "fixed": True,
    },
]


# ---------------- 缩略图 ----------------
def encode_thumb(frame_bgr: Optional[np.ndarray], width: int = THUMB_WIDTH) -> str:
    """BGR 图 → 缩小后的 JPEG base64（可直接塞进 <img src="data:...">）。"""
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return ""
    try:
        h, w = frame_bgr.shape[:2]
        if w > width:
            frame_bgr = cv2.resize(frame_bgr, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return ""


# ---------------- 读写 ----------------
def _read_raw() -> List[Dict[str, Any]]:
    if not os.path.isfile(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[history_store] 读取失败: {e}")
        return []


def _write_raw(records: List[Dict[str, Any]]) -> None:
    try:
        tmp = HISTORY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=1)
        os.replace(tmp, HISTORY_PATH)
    except Exception as e:
        print(f"[history_store] 写入失败: {e}")


def load_records() -> List[Dict[str, Any]]:
    """返回记录：固定的两条在最前，其余按时间正序（最新的在最后）。"""
    with _LOCK:
        dynamic = _read_raw()
    return [dict(r) for r in FIXED_RECORDS] + dynamic


def append_record(record: Dict[str, Any]) -> Dict[str, Any]:
    record.setdefault("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    with _LOCK:
        records = _read_raw()
        records.append(record)
        if len(records) > MAX_RECORDS:
            records = records[-MAX_RECORDS:]
        _write_raw(records)
    return record


def clear_records() -> None:
    with _LOCK:
        _write_raw([])


def delete_record(index: int) -> bool:
    """按 load_records() 的下标删除；固定记录不可删除。"""
    if index < len(FIXED_RECORDS):
        return False
    with _LOCK:
        records = _read_raw()
        i = index - len(FIXED_RECORDS)
        if 0 <= i < len(records):
            records.pop(i)
            _write_raw(records)
            return True
    return False


# ---------------- 派生 ----------------
def build_record(source: str, cell: str, summary: Dict[str, Any],
                 last: Dict[str, Any], ai_report: str = "",
                 thumb_bgr: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """把一次分析的结论 + 末帧结果组装成一条历史记录。"""
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "cell": cell,
        "result": summary.get("final_cls") or last.get("cls_result", "NoTarget"),
        "result_cn": last.get("cls_result_cn", ""),
        "conf": round(float(last.get("cls_conf") or 0.0), 4),
        "probs": last.get("cls_probs", [0, 0, 0]),
        "gray_mean": summary.get("avg_gray", last.get("gray_mean", 0)),
        "contrast": summary.get("avg_contrast", last.get("contrast", 0)),
        "saturation": summary.get("avg_saturation", last.get("saturation", 0)),
        "crust_speed": summary.get("max_crust", last.get("crust_speed", 0)),
        "node_speed": summary.get("max_node", last.get("node_speed", 0)),
        "frames": summary.get("frames", 1),
        "classified": summary.get("classified", 0),
        "counts": summary.get("counts", {}),
        "alarm": bool(summary.get("alarm", last.get("alarm", False))),
        "device": last.get("device", ""),
        "ai_report": ai_report or "",
        "thumb": encode_thumb(thumb_bgr),
    }


def stats(records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """历史总览统计（给页面上方的指标卡用）。"""
    recs = load_records() if records is None else records
    n = len(recs)
    dynamic = [r for r in recs if not r.get("fixed")]
    out = {
        "total": n,
        "counts": {"C": 0, "H": 0, "N": 0, "NoTarget": 0},
        "alarms": 0,
        "with_report": 0,
        "last_time": dynamic[-1]["time"] if dynamic else (recs[-1]["time"] if n else "—"),
        "today": 0,
    }
    today = datetime.now().strftime("%Y-%m-%d")
    for r in recs:
        key = r.get("result", "NoTarget")
        out["counts"][key] = out["counts"].get(key, 0) + 1
        if r.get("alarm"):
            out["alarms"] += 1
        if r.get("ai_report"):
            out["with_report"] += 1
        if str(r.get("time", "")).startswith(today):
            out["today"] += 1
    return out


def to_csv(records: Optional[List[Dict[str, Any]]] = None) -> str:
    recs = load_records() if records is None else records
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for r in recs:
        w.writerow([
            r.get("time", ""), r.get("source", ""), r.get("cell", ""),
            r.get("result", ""), r.get("result_cn", ""), r.get("conf", ""),
            r.get("gray_mean", ""), r.get("contrast", ""), r.get("saturation", ""),
            r.get("crust_speed", ""), r.get("node_speed", ""),
            r.get("frames", ""), "是" if r.get("alarm") else "否",
            r.get("device", ""), len(r.get("ai_report", "") or ""),
        ])
    return buf.getvalue()


def to_dataframe(records: Optional[List[Dict[str, Any]]] = None):
    """转成 pandas DataFrame（给 st.dataframe 用），不含缩略图和长文本。"""
    import pandas as pd

    recs = load_records() if records is None else records
    fixed = [r for r in recs if r.get("fixed")]
    dynamic = [r for r in recs if not r.get("fixed")]
    ordered = fixed + list(reversed(dynamic))   # 固定记录置顶，其余最新在上
    rows = []
    for r in ordered:
        rows.append({
            "时间": r.get("time", ""),
            "来源": r.get("source", ""),
            "槽号": r.get("cell", ""),
            "状态": r.get("result_cn") or r.get("result", ""),
            "代码": r.get("result", ""),
            "置信度": round(float(r.get("conf") or 0), 3),
            "平均灰度": r.get("gray_mean", ""),
            "对比度": r.get("contrast", ""),
            "饱和度": r.get("saturation", ""),
            "V_crust": r.get("crust_speed", ""),
            "V_node": r.get("node_speed", ""),
            "帧数": r.get("frames", ""),
            "报警": "⚠️ 是" if r.get("alarm") else "否",
            "设备": r.get("device", ""),
        })
    return pd.DataFrame(rows)


__all__ = [
    "HISTORY_PATH", "MAX_RECORDS", "load_records", "append_record", "clear_records",
    "delete_record", "build_record", "stats", "to_csv", "to_dataframe", "encode_thumb",
]
