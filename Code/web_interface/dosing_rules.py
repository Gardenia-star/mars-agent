# dosing_rules.py
# -*- coding: utf-8 -*-
"""
氟化铝（AlF3）加料建议 —— 规则推算模块

⚠️ 重要声明
    本模块输出的是**基于视觉过热度判定的工艺规则推算值，不是实测加料量**。
    真实 AlF3 加料量取决于分子比化验、电解温度、氧化铝浓度、电流效率、槽型与班组台账等多源数据。
    接入真实台账/数据库后，应以实测值替代这里的推算。

推算思路（与铝电解常规工艺逻辑一致）：
    过热度偏高(H) → 电解质温度偏高 / 分子比偏高 → 需要**增加 AlF3 加料**以降分子比、降温
    过热度偏低(C) → 电解质偏冷 / 分子比偏低 → 需要**减少或暂停 AlF3 加料**，防止进一步降温
    正常(N)       → 维持现有加料节奏
    幅度由「稳定判定置信度」与「异常帧占比」共同决定。
"""

from __future__ import annotations

from typing import Any, Dict, List

# 参考区间：仅作对照展示，**均为行业常见区间，非本系统实测**
REFERENCE_RANGES = [
    ("分子比 (NaF/AlF3)", "2.20 ~ 2.40", "常规区间"),
    ("电解温度", "940 ~ 965 °C", "常规区间"),
    ("过热度", "8 ~ 15 °C", "常规区间"),
    ("AlF3 加料", "由槽型与班组台账确定", "规则推算"),
]

DISCLAIMER = ("本页结论为**规则推算**：由火眼视觉过热度判定推导加料方向与幅度，"
              "并非实测加料量。实际调整请以分子比化验、槽温与班组台账为准。")

ACTION_TABLE = [
    (60, "显著增加 AlF3 加料", "#c03030", "电解质过热明显，需加大氟化铝投放以降分子比、抑制温度上行"),
    (25, "适度增加 AlF3 加料", "#e08a5a", "有过热倾向，建议小幅提高加料量并加密观察"),
    (-25, "维持当前加料节奏", "#3ddc84", "过热度处于正常区间，按现行节奏投料即可"),
    (-60, "适度减少 AlF3 加料", "#4d8dff", "电解质偏冷，建议暂缓加料，避免温度继续下行"),
    (-999, "暂停 / 显著减少 AlF3 加料", "#3d6bff", "过冷明显，需防止进一步降温与结壳加剧"),
]


def _pick_action(score: float):
    for threshold, label, color, reason in ACTION_TABLE:
        if score >= threshold:
            return label, color, reason
    return ACTION_TABLE[-1][1], ACTION_TABLE[-1][2], ACTION_TABLE[-1][3]


def analyze_dosing(results: List[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    基于一段分析结果给加料建议。

    :param results: FireEyeAnalyzer.analyze_frame 的结果序列（按时间顺序）
    :param summary: algo_web_bridge.summarize(results) 的结论
    :return: 结构化建议字典
    """
    if not results:
        return {"ready": False}

    counts = (summary or {}).get("counts", {}) or {}
    n_h, n_c, n_n = counts.get("H", 0), counts.get("C", 0), counts.get("N", 0)
    total = max(1, n_h + n_c + n_n)
    frac_h, frac_c, frac_n = n_h / total, n_c / total, n_n / total

    last = results[-1]
    cls = last.get("cls_result", "NoTarget")
    conf = float(last.get("cls_conf") or 0.0)

    # ---- 加料评分：正=需增加，负=需减少，量纲约 ±100 ----
    score = 0.0
    if cls == "H":
        score += (55 + 45 * conf) * frac_h
    elif cls == "C":
        score -= (55 + 45 * conf) * frac_c
    # 即便是正常态，但过程中异常帧较多，也给出轻微偏移
    score += 12 * frac_h - 12 * frac_c
    score = max(-100.0, min(100.0, score))

    label, color, reason = _pick_action(score)
    delta_pct = round(score / 100.0 * 15.0)          # 建议幅度按 ±15% 封顶

    # ---- 依据：列出真正参与判断的实测指标 ----
    basis: List[str] = []
    basis.append(f"稳定判定：{last.get('cls_result_cn', cls)}（置信度 {conf*100:.1f}%）")
    basis.append(f"采样帧分布：偏高 {n_h} / 正常 {n_n} / 偏低 {n_c}（共 {total} 帧有效判定）")
    basis.append(f"平均灰度 μ = {last.get('gray_mean', 0)}，对比度 σ = {last.get('contrast', 0)}")
    basis.append(f"结壳速率 V_crust 峰值 = {(summary or {}).get('max_crust', 0)}，"
                 f"角点速率 V_node 峰值 = {(summary or {}).get('max_node', 0)}")
    if (summary or {}).get("alarm"):
        basis.append("⚠️ 本次分析触发了持续异常报警（H/C 持续 >10 s 且平均置信度 >90%）")

    # ---- 工艺建议 ----
    advice: List[str] = []
    if score >= 25:
        advice.append(f"AlF3 加料量上调约 {abs(delta_pct)}%，分 2~3 次投加，避免一次过量造成局部过冷。")
        advice.append("同步关注槽电压：过热常伴随极距偏小，必要时小幅上调 5~15 mV。")
        advice.append("2 小时内复测火眼画面，确认亮区未继续扩大。")
    elif score <= -25:
        advice.append(f"AlF3 加料量下调约 {abs(delta_pct)}%，或本班次暂缓一次投加。")
        advice.append("检查是否因加料过量导致降温，必要时确认氧化铝下料是否均匀。")
        advice.append("关注结壳是否增厚、火眼亮区是否收缩，防止走向闷槽。")
    else:
        advice.append("维持当前 AlF3 加料节奏，按常规周期投料。")
        advice.append("保持加料均匀性，避免忽多忽少引起分子比波动。")
        advice.append("按班组要求做常规巡检即可。")

    if (summary or {}).get("alarm"):
        advice.append("本次已触发持续报警，建议立即人工复核该槽，并记录处理措施。")

    # ---- 建议强度随采样帧的演变（真数据推导） ----
    series = []
    for i, r in enumerate(results):
        c = r.get("cls_result")
        f = float(r.get("cls_conf") or 0.0)
        v = 0.0
        if c == "H":
            v = (55 + 45 * f)
        elif c == "C":
            v = -(55 + 45 * f)
        series.append(round(v, 1))

    return {
        "ready": True,
        "score": round(score, 1),
        "action": label,
        "color": color,
        "reason": reason,
        "direction": "增加" if score > 0 else ("减少" if score < 0 else "维持"),
        "delta_pct": delta_pct,
        "delta_text": (f"{delta_pct:+d}%" if delta_pct else "0%"),
        "severity": ("高" if abs(score) >= 60 else "中" if abs(score) >= 25 else "低"),
        "frac": {"H": round(frac_h, 3), "N": round(frac_n, 3), "C": round(frac_c, 3)},
        "basis": basis,
        "advice": advice,
        "series": series,
        "reference": REFERENCE_RANGES,
        "disclaimer": DISCLAIMER,
    }


__all__ = ["analyze_dosing", "REFERENCE_RANGES", "DISCLAIMER", "ACTION_TABLE"]
