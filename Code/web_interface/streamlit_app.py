# streamlit_app.py
# -*- coding: utf-8 -*-
"""
MARS-Agent 火眼诊断系统 —— Web 端（科技风驾驶舱）

v6 相对 v5 的变化：
  * 导航从左侧栏移到**顶部**，五个入口：性能监控 / 过程诊断 / 加药分析 / 历史管理 / 系统设置
  * 顶部标题改为「基于YOLO火眼识别与多模态大模型铝电解诊断系统」，放大并置于左上角，删除 MARS-AGENT 字样
  * 整站改为科技风主题：深空底色 + 网格 + 霓虹青描边 + 等宽数字 + 面板四角标记
  * 新增「性能监控」页（系统/推理性能）与「加药分析」页（AlF3 规则推算）
"""

import base64
import os
import sys
from datetime import datetime

import cv2
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo_web_bridge import (  # noqa: E402
    CLASS_CN, FireEyeAnalyzer, get_device, load_bgr_from_bytes,
    load_frames_from_bytes, models_ready, summarize,
)
import dosing_rules  # noqa: E402
import history_store  # noqa: E402

# 注意：except 里的 `as _e` 在 except 块结束后会被 Python 删除，
# 不能在 lambda 里闭包引用它（否则调用时 NameError）。这里先存成模块级变量。
_LLM_IMPORT_ERROR = None
try:
    from llm_web import backend_summary as llm_backend_summary, llava_analyze_frame
except Exception as _exc:  # pragma: no cover
    _LLM_IMPORT_ERROR = _exc
    llm_backend_summary = lambda: f"未加载（{_LLM_IMPORT_ERROR}）"   # noqa: E731
    llava_analyze_frame = None

DEFAULT_SAMPLE_FRAMES = 24
PAGES = [("perf", "性能监控"), ("diagnose", "过程诊断"), ("dosing", "加药分析"),
         ("history", "历史管理"), ("settings", "系统设置")]

STATE_COLOR = {"C": "#4d8dff", "H": "#c03030", "N": "#3ddc84", "NoTarget": "#8b97c9"}
STATE_TEXT = {"C": "温度偏低 COLD", "H": "严重过热 HOT", "N": "工况正常 NORMAL", "NoTarget": "等待目标 …"}
CYAN, DIM, MONO = "#5a7bff", "#8b97c9", "var(--mono)"

st.set_page_config(page_title="基于YOLO火眼识别与多模态大模型铝电解诊断系统",
                   page_icon="🔥", layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# 科技风主题
# ============================================================
st.markdown("""
<style>
:root{
    /* 参考图配色：深藏青底 + 靛蓝块 + 白字 + 红色点缀 */
    --bg0:#000410; --bg1:#000830; --bg2:#001038;
    --panel-a:#0c1140; --panel-b:#070b28;
    --line:#1b2a6b; --line-soft:#152057;
    --accent:#5a7bff; --accent-hi:#8ba4ff; --accent-deep:#2a37c9;
    --indigo:#101060;
    --text:#eef2ff; --muted:#8b97c9;
    --ok:#3ddc84; --warn:#ffb020; --bad:#c03030;
    --mono:"JetBrains Mono","Cascadia Mono",Consolas,"DejaVu Sans Mono",ui-monospace,monospace;
    --cn:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC","Source Han Sans SC",sans-serif;
}
header[data-testid="stHeader"]{display:none;}
#MainMenu,footer{visibility:hidden;}

html,body{height:100vh;overflow:hidden;}
[data-testid="stAppViewContainer"]{
    height:100vh;overflow:hidden;
    background:
        radial-gradient(1200px 620px at 8% -14%, rgba(28,40,150,.38), transparent 62%),
        radial-gradient(900px 520px at 102% -6%, rgba(16,16,96,.50), transparent 58%),
        linear-gradient(180deg,#000830 0%,#000410 100%);
    font-family:var(--cn); color:var(--text);
}
[data-testid="stAppViewContainer"]::before{
    content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.45;
    background-image:
        linear-gradient(rgba(90,123,255,.055) 1px,transparent 1px),
        linear-gradient(90deg,rgba(90,123,255,.055) 1px,transparent 1px);
    background-size:46px 46px;
    -webkit-mask-image:radial-gradient(circle at 50% 0%,#000 26%,transparent 84%);
    mask-image:radial-gradient(circle at 50% 0%,#000 26%,transparent 84%);
}
[data-testid="stMain"],[data-testid="stMainBlockContainer"],.block-container{
    height:100vh!important;overflow:hidden;position:relative;z-index:1;
}
.block-container{max-width:100%!important;padding:.45rem .95rem .35rem .95rem!important;}
[data-testid="stVerticalBlock"]{gap:.32rem;}
[data-testid="stHorizontalBlock"]{gap:.5rem;}

/* ---------- 标题：左上大字 + 左侧竖向色条 ---------- */
.app-title-wrap{display:flex;align-items:center;gap:15px;padding:2px 0 2px 2px;}
.app-title-bar{
    width:6px;height:48px;border-radius:3px;flex:0 0 auto;
    background:linear-gradient(180deg,var(--accent-hi) 0%,var(--accent-deep) 65%,var(--indigo) 100%);
    box-shadow:0 0 20px rgba(90,123,255,.80),0 0 44px rgba(42,55,201,.45);
}
.app-title{
    font-family:var(--cn);font-size:34px;font-weight:900;letter-spacing:2px;line-height:1.16;
    color:#f4f7ff;text-shadow:0 0 18px rgba(90,123,255,.45);
}
.app-title .accent{color:var(--accent-hi);}
.hdr-right{
    text-align:right;font-family:var(--mono);font-size:21px;font-weight:700;
    color:#b9c6ff;letter-spacing:1.2px;line-height:1.25;padding-top:10px;
    text-shadow:0 0 14px rgba(90,123,255,.45);
}
.hr-tech{display:none;}

/* ---------- 顶部导航：下划线标签（tab 风格） ---------- */
.st-key-tabbar{border-bottom:1px solid var(--line-soft);margin:6px 0 7px 0;}
.st-key-tabbar [data-testid="stHorizontalBlock"]{gap:0;}
.st-key-tabbar div.stButton > button{
    width:100%;background:transparent;border:none;border-radius:0;
    border-bottom:2px solid transparent;
    color:var(--muted);font-size:15px;font-weight:600;letter-spacing:2px;
    padding:7px 4px 9px 4px;box-shadow:none;transition:color .15s ease,border-color .15s ease;
}
.st-key-tabbar div.stButton > button:hover{
    color:var(--text);background:rgba(90,123,255,.10);
    border-bottom-color:rgba(90,123,255,.50);box-shadow:none;
}
.st-key-tabbar div.stButton > button[kind="primary"]{
    color:#ffffff;background:transparent;border-bottom:2px solid var(--accent);
    box-shadow:0 10px 20px -14px var(--accent);
    text-shadow:0 0 14px rgba(90,123,255,.70);
}

/* ---------- 整页外框 ---------- */
.st-key-pageframe,.st-key-pageframe-sm{
    border:1px solid var(--line);
    border-radius:5px;
    background:linear-gradient(180deg,rgba(12,17,64,.62),rgba(5,8,30,.62));
    box-shadow:0 0 30px rgba(28,40,150,.20),inset 0 0 56px rgba(28,40,150,.10);
    padding:9px 11px 8px 11px;
    margin-top:2px;
    overflow-y:auto;overflow-x:hidden;
}
.st-key-pageframe{max-height:calc(100vh - 210px);}
.st-key-pageframe-sm{max-height:calc(100vh - 180px);}
.st-key-pageframe::-webkit-scrollbar,.st-key-pageframe-sm::-webkit-scrollbar{width:7px;}
.st-key-pageframe::-webkit-scrollbar-thumb,.st-key-pageframe-sm::-webkit-scrollbar-thumb{
    background:#25357f;border-radius:3px;}
.st-key-pageframe::-webkit-scrollbar-track,.st-key-pageframe-sm::-webkit-scrollbar-track{
    background:transparent;}

/* ---------- 面板 ---------- */
.panel{
    position:relative;border:1px solid var(--line);border-radius:4px;
    background:linear-gradient(180deg,var(--panel-a),var(--panel-b));
    padding:10px 12px;margin-bottom:8px;
    box-shadow:inset 0 0 30px rgba(28,40,150,.12);
}
.panel h4,.panel h4.ptitle{
    display:flex;align-items:center;gap:8px;
    margin:0 0 8px 0;font-size:13px;font-weight:700;letter-spacing:1.6px;
    color:#dbe4ff;
}
/* 标题前的小竖条：统一版式节奏 */
.panel h4::before,.panel h4.ptitle::before{
    content:"";flex:0 0 auto;width:3px;height:13px;border-radius:2px;
    background:linear-gradient(180deg,#8ba4ff,#2a37c9);
    box-shadow:0 0 9px rgba(90,123,255,.75);
}
/* 大标题变体（3D 剖视面板用） */
h4.ptitle.big{font-size:17px;letter-spacing:2px;margin-bottom:9px;}
h4.ptitle.big::before{width:4px;height:17px;}
.panel.accent-hot{border-color:#5c1b1b;background:linear-gradient(180deg,rgba(60,10,10,.85),rgba(30,6,6,.85));}
[data-testid="stVerticalBlockBorderWrapper"]{
    border:1px solid var(--line)!important;
    border-radius:4px;
    background:linear-gradient(180deg,var(--panel-a),var(--panel-b));
    padding:10px 12px;margin-bottom:8px;
    box-shadow:inset 0 0 30px rgba(28,40,150,.12);
}
h4.ptitle{
    display:flex;align-items:center;gap:8px;
    margin:0 0 8px 0;font-size:13px;font-weight:700;letter-spacing:1.6px;
    color:#dbe4ff;
}
h4.ptitle::before{
    content:"";flex:0 0 auto;width:3px;height:13px;border-radius:2px;
    background:linear-gradient(180deg,#8ba4ff,#2a37c9);
    box-shadow:0 0 9px rgba(90,123,255,.75);
}

/* ---------- 文本工具 ---------- */
.muted{color:var(--muted);font-size:12px;}
.num{font-family:var(--mono);letter-spacing:.6px;color:var(--text);}
.kv{display:flex;justify-content:space-between;align-items:baseline;
    padding:3.5px 0;border-bottom:1px dashed rgba(90,123,255,.20);font-size:13px;}
.kv:last-child{border-bottom:0;}
.kv .k{color:#93a0d0;}
.kv .v{font-family:var(--mono);color:var(--text);font-weight:600;}

/* ---------- 状态大字 ---------- */
.state-card{text-align:center;padding:11px 8px;border-radius:4px;
    background:rgba(8,12,48,.92);border:1px solid var(--line);}
.state-title{font-size:11px;color:var(--muted);letter-spacing:3px;}
.state-value{font-size:27px;font-weight:900;letter-spacing:2px;margin:5px 0 3px 0;}
.state-conf{font-size:12px;color:#c3cdf0;}

/* ---------- 概率条 ---------- */
.probrow{display:flex;align-items:center;gap:7px;margin:4.5px 0;font-size:12px;}
.probrow .nm{width:52px;color:#a9b6e0;}
.probrow .bar{flex:1;height:8px;background:#0d1440;border-radius:4px;overflow:hidden;
              box-shadow:inset 0 0 6px rgba(0,0,0,.6);}
.probrow .bar>i{display:block;height:100%;border-radius:4px;box-shadow:0 0 8px currentColor;}
.probrow .pc{width:48px;text-align:right;font-family:var(--mono);color:var(--text);}

.badge{display:inline-block;padding:1px 8px;border-radius:2px;font-size:11px;font-weight:700;letter-spacing:.5px;}
.badge-ok{background:rgba(0,110,60,.28);color:#5ee79a;border:1px solid #17663c;}
.badge-warn{background:rgba(120,20,20,.34);color:#ff9a90;border:1px solid #7f1d1d;}
.badge-info{background:rgba(28,40,150,.38);color:#a8b8ff;border:1px solid #25357f;}

/* ---------- 指标卡 ---------- */
.statcard{border:1px solid var(--line);border-radius:4px;padding:9px 10px;text-align:center;
    background:linear-gradient(180deg,var(--panel-a),var(--panel-b));}
.statcard .lb{font-size:11px;color:var(--muted);letter-spacing:1.4px;}
.statcard .vl{font-family:var(--mono);font-size:23px;font-weight:800;margin-top:3px;}

/* ---------- 检测结果 / 汇总：紧凑网格 ---------- */
.resgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:3px;}
.rescell{border:1px solid var(--line);border-radius:3px;padding:3px 5px;text-align:center;
    background:linear-gradient(180deg,rgba(14,20,70,.85),rgba(7,11,38,.85));}
.rescell .rk{font-size:9px;color:var(--muted);letter-spacing:.6px;}
.rescell .rv{font-family:var(--mono);font-size:14px;font-weight:800;color:var(--text);margin-top:1px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.rescell .rv.sm{font-size:10px;font-weight:600;color:#a8b8ff;font-family:var(--mono);}
.probwrap{border-top:1px dashed rgba(90,123,255,.20);padding-top:3px;margin-top:2px;}

/* ---------- 视频区：高度随画面比例自适应，等比完整显示 ---------- */
.st-key-videobox{
    width:100%;height:auto;min-height:200px;
    background:radial-gradient(circle at 50% 40%,#050a2a 0%,#00030f 88%);
    border:1px solid var(--line);border-radius:4px;
    display:flex;align-items:center;justify-content:center;
    overflow:hidden;padding:0;
}
/* 有画面时：框体贴合画面本身并居中，消除两侧大片深蓝底色。
   竖屏视频（如 720x1280）在宽框里只能占 26%，剩下 74% 都是这个蓝底 ——
   这就是"视频显示是蓝色"的原因。 */
.st-key-videobox:has(.videoinner){
    width:fit-content;max-width:100%;margin:0 auto;
    background:transparent;
}
.videoinner{display:flex;align-items:center;justify-content:center;}
.videoinner img{
    width:100%;height:100%;max-height:100%;
    object-fit:contain;object-position:center;
    display:block;border-radius:3px;
}
.st-key-videobox video{width:100%;height:42vh;object-fit:contain;display:block;}
.video-empty{display:flex;align-items:center;justify-content:center;height:200px;width:100%;
    color:#5b6b9a;font-size:14px;letter-spacing:1px;text-align:center;padding:10px;}

/* ---------- 上传：小按钮（popover） ---------- */
[data-testid="stPopover"] > button{
    background:rgba(28,40,150,.42);color:#b9c6ff;border:1px solid var(--line);
    border-radius:4px;padding:6px 13px;font-size:14px;font-weight:600;
    letter-spacing:.6px;font-family:var(--cn);width:auto!important;
}
[data-testid="stPopover"] > button:hover{
    background:rgba(90,123,255,.32);color:#ffffff;border-color:var(--accent);
    box-shadow:0 0 16px rgba(90,123,255,.45);
}
[data-testid="stPopoverBody"]{
    background:#060a26;border:1px solid var(--line);border-radius:5px;
    box-shadow:0 10px 30px rgba(0,0,0,.6);}
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"]{
    padding:6px 10px!important;min-height:0!important;
    background:rgba(9,13,48,.85);border:1px dashed var(--line);}
[data-testid="stPopoverBody"] [data-testid="stFileUploaderDropzone"] span{font-size:12px;}

/* ---------- 通用控件 ---------- */
div.stButton>button{
    width:100%;background:rgba(28,40,150,.34);color:#b9c6ff;
    border:1px solid var(--line);border-radius:4px;padding:7px 10px;font-size:15px;
    font-weight:600;letter-spacing:1.2px;font-family:var(--cn);transition:all .15s ease;
}
div.stButton>button:hover{background:rgba(90,123,255,.30);color:#ffffff;border-color:var(--accent);
    box-shadow:0 0 16px rgba(90,123,255,.40);}
div.stButton>button[kind="primary"]{
    background:linear-gradient(180deg,rgba(90,123,255,.42),rgba(42,55,201,.42));
    color:#ffffff;border-color:var(--accent);box-shadow:0 0 18px rgba(90,123,255,.45);}
[data-testid="stFileUploaderDropzone"]{padding:5px 10px!important;min-height:0!important;
    background:rgba(9,13,48,.8);border:1px dashed var(--line);}
[data-testid="stFileUploaderDropzone"] span{font-size:12px;color:#a8b8ff;}
[data-testid="stDataFrame"]{border:1px solid var(--line);}
.stSlider label,.stSelectbox label,.stTextInput label{font-size:12px!important;color:var(--muted)!important;}
[data-testid="stMetricValue"]{font-family:var(--mono);color:var(--accent-hi);}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 会话状态
# ============================================================
def init_state():
    ss = st.session_state
    ss.setdefault("page", "perf")
    ss.setdefault("analyzer", FireEyeAnalyzer())
    ss.setdefault("upload_sig", None)
    # 已载入的源，用普通 session_state 保存（不是组件状态），
    # 这样切到「历史管理/系统设置」把上传组件清掉后，源和结果依然保留。
    ss.setdefault("src_sig", None)
    ss.setdefault("src_name", None)
    ss.setdefault("src_mime", "")
    ss.setdefault("src_bytes", None)
    ss.setdefault("results", None)
    ss.setdefault("summary", None)
    ss.setdefault("ai_report", "")
    ss.setdefault("display_bgr", None)
    ss.setdefault("source_name", "")
    ss.setdefault("sample_frames", DEFAULT_SAMPLE_FRAMES)


init_state()


def reset_analysis():
    ss = st.session_state
    ss.results = ss.summary = ss.display_bgr = None
    ss.ai_report = ""
    ss.analyzer.reset()


def cell_from_name(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    return f"{stem}号槽" if stem else "未知来源"


def navigate(page: str):
    """切换页面。注意：不要叫 go —— 会遮蔽 `import plotly.graph_objects as go`。"""
    st.session_state.page = page
    st.rerun()


# ============================================================
# 图表（统一科技风）
# ============================================================
def _layout(fig, height=140, show_legend=False):
    fig.update_layout(
        height=height, margin=dict(l=6, r=6, t=8, b=6),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Consolas, DejaVu Sans Mono, monospace", size=10, color="#8b97c9"),
        showlegend=show_legend,
        legend=dict(orientation="h", y=1.22, x=0, font=dict(size=10, color="#93a0d0")),
        xaxis=dict(showgrid=False, visible=False, zeroline=False),
        yaxis=dict(gridcolor="rgba(23,80,125,.45)", zeroline=False),
    )
    return fig


def chart_series(results, key, color=CYAN, height=132, name=None):
    vals = [r.get(key, 0) for r in results]
    fig = go.Figure(go.Scatter(
        y=vals, mode="lines+markers", name=name or key,
        line=dict(color=color, width=2, shape="spline"),
        marker=dict(size=4, color=color),
        fill="tozeroy", fillcolor="rgba(0,234,255,.07)"))
    return _layout(fig, height)


def chart_multi(results, series, height=140):
    fig = go.Figure()
    for name, key, color, dash in series:
        fig.add_scatter(y=[r.get(key, 0) for r in results], mode="lines", name=name,
                        line=dict(color=color, width=2, dash=dash))
    return _layout(fig, height, show_legend=True)


def chart_donut(counts, height=190):
    keys = [k for k in ("H", "N", "C", "NoTarget") if counts.get(k, 0) > 0]
    values = [counts[k] for k in keys]
    labels = [CLASS_CN.get(k, k) for k in keys]
    colors = [STATE_COLOR[k] for k in keys]
    if not values:
        values, labels, colors = [1], ["暂无数据"], ["#12314c"]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.66,
                           marker=dict(colors=colors, line=dict(color="#04070f", width=2)),
                           textinfo="label+percent", textfont=dict(size=11, color="#d6e6f5")))
    fig.update_layout(height=height, margin=dict(l=6, r=6, t=6, b=6), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ============================================================
# 小组件
# ============================================================
def prob_bars(probs):
    rows = [("偏低 C", "#4d8dff", 0), ("偏高 H", "#c03030", 1), ("正常 N", "#3ddc84", 2)]
    html = []
    for cn, color, i in rows:
        p = float(probs[i]) if probs and i < len(probs) else 0.0
        html.append(
            f'<div class="probrow"><span class="nm">{cn}</span>'
            f'<span class="bar"><i style="width:{p*100:.1f}%;background:{color};color:{color}"></i></span>'
            f'<span class="pc">{p*100:.1f}%</span></div>')
    return "".join(html)


def kv_html(items):
    return "".join(f'<div class="kv"><span class="k">{k}</span><span class="v">{v}</span></div>'
                   for k, v in items)


def stat_cards(items):
    cols = st.columns(len(items))
    for col, (label, value, color) in zip(cols, items):
        with col:
            st.markdown(f'<div class="statcard"><div class="lb">{label}</div>'
                        f'<div class="vl" style="color:{color}">{value}</div></div>',
                        unsafe_allow_html=True)


def no_data_hint(text="尚未分析。请上传火眼图片 / 视频并点「启动分析」。"):
    st.markdown(f'<div class="panel"><div class="muted">{text}</div></div>', unsafe_allow_html=True)


# ============================================================
# 铝电解槽 3D 结构展示（剖视）
# ============================================================
# 立方体三角面索引（8 顶点 → 12 三角形，顶点顺序与 _box_trace 的 x/y/z 一致）
_CUBE_I = [0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1]
_CUBE_J = [1, 2, 6, 7, 5, 4, 2, 6, 3, 7, 5, 6]
_CUBE_K = [2, 3, 5, 6, 1, 5, 6, 7, 7, 4, 6, 2]


def _box_trace(x0, y0, z0, x1, y1, z1, color, name, opacity=1.0):
    """一个长方体（plotly Mesh3d）。"""
    return go.Mesh3d(
        x=[x0, x1, x1, x0, x0, x1, x1, x0],
        y=[y0, y0, y1, y1, y0, y0, y1, y1],
        z=[z0, z0, z0, z0, z1, z1, z1, z1],
        i=_CUBE_I, j=_CUBE_J, k=_CUBE_K,
        color=color, opacity=opacity, name=name, showlegend=False, showscale=False,
        flatshading=True, hoverinfo="name",
        lighting=dict(ambient=0.74, diffuse=0.62, specular=0.16, roughness=0.88),
    )


def cell_3d(height=290, active=False):
    """
    铝电解槽本体 3D 剖视图（可拖拽旋转 / 滚轮缩放）。

    自下而上：阴极炭块 → 铝液层 → 电解质（冰晶石熔体 ~960℃）→ 壳面结壳；
    上方是浸入电解质的阳极炭块、阳极导杆与阳极母线；
    槽壳为半透明钢壳，正面剖开便于观察内部。
    """
    L, W = 5.0, 2.2          # 槽内腔半长 / 半宽
    wall = 0.34              # 槽壁厚
    traces = []

    # ---- 槽壳（钢壳）：底 + 后/左/右三面墙，正面剖开 ----
    traces.append(_box_trace(-L - wall, -W - wall, -0.34, L + wall, W + wall, 0.0,
                             "#54748f", "槽壳（钢壳）· 槽底", 0.95))
    traces.append(_box_trace(-L - wall, -W - wall, 0.0, -L, W + wall, 3.3,
                             "#54748f", "槽壳 · 侧壁", 0.40))
    traces.append(_box_trace(L, -W - wall, 0.0, L + wall, W + wall, 3.3,
                             "#54748f", "槽壳 · 侧壁", 0.40))
    traces.append(_box_trace(-L, W, 0.0, L, W + wall, 3.3,
                             "#54748f", "槽壳 · 端壁", 0.40))

    # ---- 槽内自下而上的各层 ----
    traces.append(_box_trace(-L, -W, 0.0, L, W, 0.5, "#2b2f38", "阴极炭块（槽底）", 1.0))
    traces.append(_box_trace(-L, -W, 0.5, L, W, 1.25, "#c9d3de", "铝液层", 1.0))
    traces.append(_box_trace(-L, -W, 1.25, L, W, 2.5,
                             "#ffb454" if active else "#ff9d3c",
                             "电解质（冰晶石熔体 ~960℃）", 0.95 if active else 0.88))

    # ---- 壳面结壳（中间留出火眼口）----
    cz0, cz1 = 2.5, 2.92
    traces.append(_box_trace(-L, -W, cz0, -1.0, W, cz1, "#6b5b4a", "壳面结壳", 1.0))
    traces.append(_box_trace(1.0, -W, cz0, L, W, cz1, "#6b5b4a", "壳面结壳", 1.0))
    traces.append(_box_trace(-1.0, -W, cz0, 1.0, -0.65, cz1, "#6b5b4a", "壳面结壳", 1.0))
    traces.append(_box_trace(-1.0, 0.65, cz0, 1.0, W, cz1, "#6b5b4a", "壳面结壳", 1.0))

    # ---- 阳极炭块 + 阳极导杆（一排 4 组，下端浸入电解质）----
    n = 4
    step = (2 * L) / n
    for k in range(n):
        cx = -L + step * (k + 0.5)
        traces.append(_box_trace(cx - step * 0.34, -W * 0.74, 1.95,
                                 cx + step * 0.34, W * 0.74, 3.75,
                                 "#1b1b22", "阳极炭块", 1.0))
        traces.append(_box_trace(cx - 0.09, -0.09, 3.75, cx + 0.09, 0.09, 4.75,
                                 "#b87333", "阳极导杆", 1.0))

    # ---- 阳极母线（直流供电）----
    traces.append(_box_trace(-L, -0.15, 4.75, L, 0.15, 5.1,
                             "#b87333", "阳极母线（直流供电）", 1.0))

    # ---- 有分析数据时，高亮火眼观察口 ----
    if active:
        traces.append(go.Scatter3d(
            x=[0], y=[0], z=[2.75], mode="markers+text",
            marker=dict(size=8, color="#5a7bff", symbol="diamond",
                        line=dict(color="#ffffff", width=1)),
            text=["火眼观察口"], textposition="top center",
            textfont=dict(size=10, color="#f4f7ff",
                          family="Microsoft YaHei, PingFang SC, sans-serif"),
            hoverinfo="skip", showlegend=False))

    # ---- 部件标注 ----
    ann = [
        ("槽壳",     L + wall + 0.15, W + 0.15, 2.5),
        ("壳面结壳", -L + 0.8,        W + 0.15, 3.1),
        ("阳极炭块", -L + step * 0.5, -W - 0.25, 3.7),
        ("阳极导杆", L - step * 0.5,  -W - 0.25, 4.5),
        ("阳极母线", 0.0,             -W - 0.25, 5.2),
        ("电解质",   0.0,             -W - 0.25, 2.0),
        ("铝液层",   0.0,             -W - 0.25, 0.85),
        ("阴极炭块", 0.0,             -W - 0.25, 0.2),
    ]
    traces.append(go.Scatter3d(
        x=[a[1] for a in ann], y=[a[2] for a in ann], z=[a[3] for a in ann],
        mode="text", text=[a[0] for a in ann],
        textfont=dict(size=10, color="#a8b8ff",
                      family="Microsoft YaHei, PingFang SC, sans-serif"),
        hoverinfo="skip", showlegend=False))

    fig = go.Figure(traces)
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=6, b=0),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.55, y=-1.75, z=0.95)),
            aspectmode="manual", aspectratio=dict(x=1.45, y=0.8, z=0.95),
        ),
    )
    return fig


def result_grid(cells):
    """检测结果紧凑网格：cells = [(标签, 值, 颜色, 是否小字)]，3 列自动换行。"""
    items = []
    for label, value, color, small in cells:
        cls = "rv sm" if small else "rv"
        style = "" if small else f' style="color:{color}"'
        items.append(f'<div class="rescell"><div class="rk">{label}</div>'
                     f'<div class="{cls}"{style}>{value}</div></div>')
    return f'<div class="resgrid">{"".join(items)}</div>'


def b64_jpeg(bgr, max_w=1280, quality=85):
    """
    把 BGR 帧编码成 data URI 的内联 JPEG。

    为什么不直接用 st.image：Streamlit 会给 <img> 加自己的内联样式与包裹结构，
    外部 CSS 经常抢不过它，导致画面显示得很小。自己拼 <img> 就能完全由 CSS 控制尺寸。
    """
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return ""
    try:
        h, w = bgr.shape[:2]
        if w > max_w:                       # 限宽，避免 base64 过大拖慢页面
            bgr = cv2.resize(bgr, (max_w, max(1, int(h * max_w / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return ""


def show_big_frame(bgr, cls="videoinner", scale=1.0):
    """
    按画面自身比例给容器**确定的高度**，然后铺满显示。

    关键点：`.videoinner` 必须自己带 `height:<n>vh`，不能写 `height:100%` ——
    Streamlit 在容器外面还套了 stElementContainer / stMarkdown 等层，
    这些中间层高度是 auto，`height:100%` 会解析成 auto，图片就退回原始尺寸，
    竖屏视频会垂直溢出并被 `overflow:hidden` 裁掉上下画面。
    竖屏给更高、宽屏给更矮，尽量把画面放大又不超出预算。
    """
    uri = b64_jpeg(bgr)
    if not uri:
        return False
    h, w = bgr.shape[:2]
    ratio = w / max(1, h)
    vh = (52 if ratio < 1 else (42 if ratio < 1.6 else 38)) * scale
    style = (f"height:{vh}vh;aspect-ratio:{w}/{h};width:auto;"
             f"max-width:100%;margin:0 auto;")
    st.markdown(f'<div class="{cls}" style="{style}"><img src="{uri}"/></div>',
                unsafe_allow_html=True)
    return True


def perf_charts(results):
    """四张性能指标曲线：耗时 / 吞吐 / 检出目标数 / 分类置信度。"""
    def prep():
        ms = [r.get("infer_ms", 0) for r in results]
        fps = [(1000.0 / m) if m else 0 for m in ms]
        objs = [len(r.get("boxes", {}).get("eye", [])) + len(r.get("boxes", {}).get("fire", []))
                for r in results]
        conf = [float(r.get("cls_conf") or 0) * 100 for r in results]
        return ms, fps, objs, conf

    specs = [
        ("单帧推理耗时 (ms)", "#5a7bff", 0),
        ("等效吞吐 (FPS)", "#3ddc84", 1),
        ("检出目标数 (个)", "#ffb020", 2),
        ("分类置信度 (%)", "#a58bff", 3),
    ]
    ms, fps, objs, conf = prep()
    data = [ms, fps, objs, conf]
    figs = []
    for i, (title, color, _) in enumerate(specs):
        y = data[i]
        fig = go.Figure(go.Scatter(
            y=y, mode="lines+markers",
            line=dict(color=color, width=2, shape="spline"),
            marker=dict(size=3.5, color=color),
            fill="tozeroy", fillcolor=f"rgba{tuple(int(color.lstrip('#')[j:j+2],16) for j in (0,2,4)) + (0.09,)}"))
        fig.update_layout(
            height=66, margin=dict(l=3, r=3, t=14, b=2),
            title=dict(text=title, font=dict(size=11, color="#93a0d0"), x=0.01, y=0.97),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            font=dict(family="Consolas, DejaVu Sans Mono, monospace", size=9, color="#8b97c9"),
            xaxis=dict(showgrid=False, visible=False, zeroline=False),
            yaxis=dict(gridcolor="rgba(23,80,125,.42)", zeroline=False),
        )
        figs.append(fig)
    return specs, data, figs


# ============================================================
# 分析执行
# ============================================================
def run_analysis(data: bytes, name: str, mime: str = ""):
    ss = st.session_state
    is_image = (mime or "").startswith("image")
    analyzer = FireEyeAnalyzer()

    if is_image:
        bgr = load_bgr_from_bytes(data)
        if bgr is None:
            st.error("图片解码失败")
            return False
        res = analyzer.analyze_frame(bgr, immediate=True)
        results = [res]
        thumb = res["annotated_bgr"] if res.get("has_target") else bgr
    else:
        frames, _info = load_frames_from_bytes(data, max_frames=int(ss.sample_frames))
        if not frames:
            st.error("视频解码失败，或该视频没有可读帧")
            return False
        results = []
        bar = st.progress(0.0, text=f"正在逐帧推理 0/{len(frames)}（{get_device()}）")
        for i, f in enumerate(frames):
            results.append(analyzer.analyze_frame(f))
            bar.progress((i + 1) / len(frames),
                         text=f"正在逐帧推理 {i+1}/{len(frames)}（{get_device()}）")
        bar.empty()
        thumb = results[-1]["annotated_bgr"]

    summary = summarize(results)
    last = results[-1]
    ss.results, ss.summary = results, summary
    ss.display_bgr, ss.source_name, ss.ai_report = thumb, name, ""

    record = history_store.build_record(
        source=name, cell=cell_from_name(name), summary=summary, last=last, thumb_bgr=thumb)
    dosing = dosing_rules.analyze_dosing(results, summary)
    if dosing.get("ready"):
        record["dosing"] = {"action": dosing["action"], "delta": dosing["delta_text"],
                            "severity": dosing["severity"], "score": dosing["score"]}
    history_store.append_record(record)
    return True


def control_bar():
    """
    按页面自适应的控制条：

      性能监控：只有「载入视频/图片」——**载入即自动执行分析**，无需按钮
      过程诊断：载入（同样自动分析）+ 「🤖 AI 诊断」生成大模型报告
      加药分析：只有载入（自动分析），建议由分析结果推算
    """
    ss = st.session_state
    page = ss.page
    show_ai = (page == "diagnose")

    # 一行搞定：只有小按钮（+ 过程诊断页的 AI 诊断），不占整行、不带提示文字
    if show_ai:
        c1, c2 = st.columns([1.7, 1.6])
    else:
        c1, _ = st.columns([1.7, 8.3])

    with c1:
        with st.popover("📁 载入视频 / 图片"):
            up = st.file_uploader("源", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"],
                                  label_visibility="collapsed", key="main_upload")
            st.caption("支持 mp4 / avi / mov / jpg / png")

    ai_btn = False
    if show_ai:
        with c2:
            ai_btn = st.button("🤖 AI 诊断", type="primary", key="ai_btn")

    # ---- 只有"真的选了一个新文件"才清空重算 ----
    # 注意：切到「历史管理/系统设置」时本组件不渲染，Streamlit 会把它的状态清掉，
    # 回到分析页时 up 会变成 None。如果不加 `up is not None` 判断，就会误判成
    # "换了新文件"而调用 reset_analysis()，把已载入的视频和结果清空 ——
    # 表现就是"每进一个新页面都要重新载入"。
    if up is not None:
        data = up.getvalue()
        sig = f"{up.name}:{len(data)}"
        if sig != ss.get("src_sig"):
            ss.src_sig = sig
            ss.src_name = up.name
            ss.src_mime = up.type or ""
            ss.src_bytes = data
            reset_analysis()
            with st.spinner(f"正在推理…（{get_device()}）"):
                if run_analysis(data, up.name, up.type or ""):
                    st.rerun()

    # ---- AI 诊断（仅过程诊断页）----
    last = ss.results[-1] if ss.results else None
    if ai_btn:
        if last is None or ss.display_bgr is None:
            st.warning("请先载入火眼图片或视频")
        elif llava_analyze_frame is None:
            st.error("AI 模块未加载")
        else:
            with st.spinner(f"大模型推理中…（{llm_backend_summary()}）"):
                try:
                    frame = last["roi_bgr"] if last.get("roi_bgr") is not None else ss.display_bgr
                    ss.ai_report = llava_analyze_frame(frame, last, model="llava")
                except Exception as e:
                    ss.ai_report = f"AI 诊断失败：{e}"
    ss["_up"] = up          # 控制条现在由 main() 统一渲染，各页面从这里取
    return up


# ============================================================
# 1) 性能监控 —— 系统 / 推理性能
# ============================================================
def left_video_panel(scale=1.0, with_results=True):
    """
    左侧固定面板：视频 / 图像源 + 检测结果。

    三个分析页（性能监控 / 过程诊断 / 加药分析）共用同一个面板，
    源与分析结果都存在 session_state 里，**切页不会丢，不需要重新载入**。
    """
    ss = st.session_state
    src = ss.get("src_name")
    with st.container(key="videobox"):
        if ss.display_bgr is not None:
            show_big_frame(ss.display_bgr, scale=scale)
        elif ss.get("src_bytes") and not (ss.get("src_mime") or "").startswith("image"):
            st.video(ss.src_bytes, width="stretch")      # 已载入但还没分析时的预览
        else:
            st.markdown('<div class="video-empty">◈ 尚未载入画面 ◈<br>'
                        '点左上「📁 载入视频 / 图片」，载入后自动分析</div>',
                        unsafe_allow_html=True)

    if not with_results:
        return

    results = ss.results or []
    last = results[-1] if results else None
    code = (last or {}).get("cls_result", "NoTarget")
    conf = float((last or {}).get("cls_conf") or 0) * 100
    roi = last.get("roi_size") if last else None
    with_t = sum(1 for r in results if r.get("has_target")) if results else 0
    with st.container(border=True):
        st.markdown('<h4 class="ptitle">检测结果</h4>' + result_grid([
            ("工况判定", CLASS_CN.get(code, code), STATE_COLOR.get(code, DIM), False),
            ("置信度", f"{conf:.1f}%", "#f4f7ff", False),
            ("检出目标帧", f'{with_t} / {len(results)}' if results else "—", "#8ba4ff", False),
            ("采样帧数", str((ss.summary or {}).get("frames", 0)), "#8ba4ff", False),
            ("ROI 尺寸", f"{roi[0]}×{roi[1]}" if roi else "—", "#f4f7ff", False),
            ("来源", src or "未选择", "#a8b8ff", True),
        ]) + '<div class="probwrap">' + prob_bars(last["cls_probs"] if last else [0, 0, 0])
            + '</div>', unsafe_allow_html=True)


def page_perf():
    ss = st.session_state
    results = ss.results or []

    left, right = st.columns([6.0, 4.0])

    with left:
        left_video_panel()

    # ---------------- 右：上=3D 流程图，下=四张性能曲线 ----------------
    with right:
        with st.container(border=True):
            st.markdown('<h4 class="ptitle big">铝电解槽 3D 结构剖视</h4>', unsafe_allow_html=True)
            st.plotly_chart(cell_3d(height=290, active=bool(results)),
                            use_container_width=True, config={"displayModeBar": False})

        specs, data, figs = perf_charts(results)
        with st.container(border=True):
            st.markdown('<h4 class="ptitle">性能指标实时分析</h4>', unsafe_allow_html=True)
            r1c1, r1c2 = st.columns(2)
            r2c1, r2c2 = st.columns(2)
            for (title, _color, _i), fig, col in zip(specs, figs, [r1c1, r1c2, r2c1, r2c2]):
                with col:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            if results:
                # 压成一行 4 格，省掉原来 4 行 kv 的高度
                st.markdown(
                    '<div class="resgrid" style="grid-template-columns:repeat(4,1fr);">'
                    + f'<div class="rescell"><div class="rk">耗时 min/max</div>'
                      f'<div class="rv sm">{min(data[0])} / {max(data[0])} ms</div></div>'
                      f'<div class="rescell"><div class="rk">吞吐 min/max</div>'
                      f'<div class="rv sm">{min(data[1]):.1f} / {max(data[1]):.1f} FPS</div></div>'
                      f'<div class="rescell"><div class="rk">目标数合计</div>'
                      f'<div class="rv sm">{sum(data[2])} 个</div></div>'
                      f'<div class="rescell"><div class="rk">置信度平均</div>'
                      f'<div class="rv sm">{sum(data[3])/len(data[3]):.1f}%</div></div>'
                    + '</div>', unsafe_allow_html=True)


# ============================================================
# 2) 过程诊断 —— 工况 + AI 诊断
# ============================================================
def page_diagnose():
    ss = st.session_state
    results, summary = ss.results, ss.summary
    last = results[-1] if results else None

    left, right = st.columns([5.9, 4.1])
    with left:
        left_video_panel()

        if ss.ai_report:
            st.markdown('<div class="panel"><h4>🤖 专家诊断报告</h4>'
                        f'<div style="max-height:23vh;overflow-y:auto;white-space:pre-wrap;'
                        f'font-size:13px;line-height:1.65;color:#dbe9ff;">{ss.ai_report}</div></div>',
                        unsafe_allow_html=True)

    with right:
        code = (last or {}).get("cls_result", "NoTarget")
        conf = (last or {}).get("cls_conf", 0.0)
        color = STATE_COLOR.get(code, DIM)
        trend = (last or {}).get("trend", "")
        st.markdown(
            f'<div class="state-card" style="border-color:{color}66;box-shadow:inset 0 0 30px {color}22;">'
            f'<div class="state-title">实时工况判定</div>'
            f'<div class="state-value" style="color:{color};text-shadow:0 0 16px {color}77;">'
            f'{STATE_TEXT.get(code, code)}</div>'
            f'<div class="state-conf">置信度 <span class="num">{conf*100:.1f}%</span>'
            + (f' &nbsp;·&nbsp; <b style="color:#ffb74d">{trend}</b>' if trend else '')
            + '</div></div>', unsafe_allow_html=True)

        if last and last.get("alarm"):
            st.markdown('<div class="panel accent-hot">'
                        '<h4 style="color:#ff9a90;">⚠ 持续异常报警</h4>'
                        f'<div class="muted" style="color:#f0b8b0;">{CLASS_CN.get(code,"")} 持续 '
                        f'<span class="num">{last.get("alarm_secs",0)}</span> 秒且平均置信度 &gt; 90%</div>'
                        '</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><h4>三分类置信度</h4>'
                    + prob_bars(last["cls_probs"] if last else [0, 0, 0]) + '</div>',
                    unsafe_allow_html=True)

        if last:
            st.markdown('<div class="panel"><h4>实测特征（ROI 内计算）</h4>' + kv_html([
                ("平均灰度 μ", last["gray_mean"]),
                ("对比度 σ", last["contrast"]),
                ("饱和度 S", last["saturation"]),
                ("角点数", last["corners"]),
                ("结壳速率 V_crust", last["crust_speed"]),
                ("角点速率 V_node", last["node_speed"]),
                ("ROI 尺寸", str(last.get("roi_size") or "—")),
            ]) + '</div>', unsafe_allow_html=True)

        if summary:
            counts = summary.get("counts", {})
            st.markdown('<div class="panel"><h4>本次采样统计</h4>' + kv_html([
                ("有效判定帧", f'{summary.get("classified",0)} / {summary.get("frames",0)}'),
                ("偏高 / 正常 / 偏低",
                 f'{counts.get("H",0)} / {counts.get("N",0)} / {counts.get("C",0)}'),
                ("是否触发报警", '<span class="badge badge-warn">是</span>' if summary.get("alarm")
                 else '<span class="badge badge-ok">否</span>'),
            ]) + '</div>', unsafe_allow_html=True)

        if results and len(results) > 1:
            with st.container(border=True):
                st.markdown('<h4 class="ptitle">平均灰度 μ 趋势</h4>', unsafe_allow_html=True)
                st.plotly_chart(chart_series(results, "gray_mean"), use_container_width=True,
                                config={"displayModeBar": False})
            with st.container(border=True):
                st.markdown('<h4 class="ptitle">动态速率趋势</h4>', unsafe_allow_html=True)
                st.plotly_chart(chart_multi(results, [
                    ("V_crust", "crust_speed", "#e08a5a", "solid"),
                    ("V_node", "node_speed", CYAN, "dot")]), use_container_width=True,
                    config={"displayModeBar": False})


# ============================================================
# 3) 加药分析 —— AlF3 规则推算
# ============================================================
def page_dosing():
    ss = st.session_state
    results, summary = ss.results, ss.summary

    if not results:
        no_data_hint()
        st.markdown('<div class="panel"><h4>规则说明</h4>'
                    '<div class="muted">过热度偏高 → 增加 AlF3 加料以降分子比、降温；'
                    '过热度偏低 → 减少/暂停加料。幅度由稳定判定置信度与异常帧占比共同决定。'
                    '该结论为规则推算，非实测加料量。</div></div>', unsafe_allow_html=True)
        return

    d = dosing_rules.analyze_dosing(results, summary)
    if not d.get("ready"):
        no_data_hint()
        return

    stat_cards([
        ("建议动作", d["action"].split(" ")[0], d["color"]),
        ("建议幅度", d["delta_text"], d["color"]),
        ("推算强度", f'{d["score"]:+.0f}', d["color"]),
        ("风险等级", d["severity"], {"高": "#c03030", "中": "#ffb020", "低": "#3ddc84"}[d["severity"]]),
    ])

    left, right = st.columns([5.3, 4.7])

    with left:
        left_video_panel(scale=0.5, with_results=False)   # 左侧固定视频（跨页面保留）
        st.markdown(f'<div class="panel" style="border-color:{d["color"]}66;">'
                    f'<h4 style="color:{d["color"]};">加料建议</h4>'
                    f'<div style="font-size:21px;font-weight:900;color:{d["color"]};'
                    f'text-shadow:0 0 14px {d["color"]}66;margin:4px 0 6px 0;">{d["action"]}</div>'
                    f'<div class="muted">{d["reason"]}</div></div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<h4 class="ptitle">建议强度随采样帧演变</h4>', unsafe_allow_html=True)
            fig = go.Figure(go.Scatter(
                y=d["series"], mode="lines+markers",
                line=dict(color=CYAN, width=2), marker=dict(size=4, color=CYAN),
                fill="tozeroy", fillcolor="rgba(0,234,255,.08)"))
            fig.add_hline(y=25, line=dict(color="#ffb020", width=1, dash="dot"))
            fig.add_hline(y=-25, line=dict(color="#4d8dff", width=1, dash="dot"))
            st.plotly_chart(_layout(fig, 165), use_container_width=True,
                            config={"displayModeBar": False})
            st.markdown('<div class="muted">正值=需增加加料，负值=需减少；虚线为 ±25 的维持区间边界</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="panel"><h4>判定依据（全部为实测/推算值）</h4>'
                    + "".join(f'<div class="kv"><span class="k">·</span>'
                              f'<span class="v" style="font-weight:500;text-align:right;">{b}</span></div>'
                              for b in d["basis"])
                    + '</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><h4>操作建议</h4>'
                    + "".join(f'<div class="kv"><span class="k">▸</span>'
                              f'<span class="v" style="font-weight:500;text-align:right;">{a}</span></div>'
                              for a in d["advice"])
                    + '</div>', unsafe_allow_html=True)

        with st.container(border=True):
            frac = d["frac"]
            st.markdown('<h4 class="ptitle">本次帧状态分布</h4>'
                        + prob_bars([frac["C"], frac["H"], frac["N"]]), unsafe_allow_html=True)

        st.markdown('<div class="panel"><h4>参考区间（行业常规，非实测）</h4>'
                    + kv_html([(n, f'<span class="num">{v}</span> <span class="muted">{s}</span>')
                               for n, v, s in d["reference"]])
                    + '</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="panel" style="border-color:#7a5a10;">'
                    f'<h4 style="color:#ffb020;">⚠ 重要声明</h4>'
                    f'<div class="muted" style="line-height:1.6;">{d["disclaimer"]}</div></div>',
                    unsafe_allow_html=True)


# ============================================================
# 4) 历史管理
# ============================================================
def page_history():
    records = history_store.load_records()
    s = history_store.stats(records)

    stat_cards([
        ("总记录", s["total"], CYAN),
        ("今日新增", s["today"], "#8ba4ff"),
        ("报警次数", s["alarms"], "#ff9a90"),
        ("含AI报告", s["with_report"], "#a58bff"),
        ("最近一次", str(s["last_time"])[-8:] if s["total"] else "—", "#ffb020"),
    ])

    left, right = st.columns([5.6, 4.4])

    with left:
        f1, f2, f3, f4 = st.columns([2, 2.2, 1.2, 1.2])
        with f1:
            sel = st.selectbox("状态筛选", ["全部", "偏高", "正常", "偏低", "报警"], key="hist_filter")
        with f2:
            kw = st.text_input("搜索槽号/来源", "", key="hist_kw", placeholder="如 9306")
        with f3:
            st.download_button("⬇ CSV", data=history_store.to_csv(records).encode("utf-8-sig"),
                               file_name=f"fireeye_history_{datetime.now():%Y%m%d_%H%M}.csv",
                               mime="text/csv", key="hist_dl", use_container_width=True)
        with f4:
            if st.button("🗑 清空", key="hist_clear", use_container_width=True):
                history_store.clear_records()
                st.rerun()

        view = records
        if sel == "偏高":
            view = [r for r in view if r.get("result") == "H"]
        elif sel == "正常":
            view = [r for r in view if r.get("result") == "N"]
        elif sel == "偏低":
            view = [r for r in view if r.get("result") == "C"]
        elif sel == "报警":
            view = [r for r in view if r.get("alarm")]
        if kw.strip():
            k = kw.strip().lower()
            view = [r for r in view if k in str(r.get("cell", "")).lower()
                    or k in str(r.get("source", "")).lower()]

        df = history_store.to_dataframe(view)
        if df.empty:
            st.markdown('<div class="panel"><div class="muted">没有符合条件的记录。'
                        '到「过程诊断」上传图片/视频并点「启动分析」即可产生记录。</div></div>',
                        unsafe_allow_html=True)
        else:
            st.dataframe(df, width="stretch", height=300, hide_index=True)

    with right:
        if not records:
            no_data_hint("暂无历史记录。")
        else:
            with st.container(border=True):
                st.markdown('<h4 class="ptitle">状态分布</h4>', unsafe_allow_html=True)
                st.plotly_chart(chart_donut(s["counts"]), use_container_width=True,
                                config={"displayModeBar": False})

            with st.container(border=True):
                st.markdown('<h4 class="ptitle">记录详情</h4>', unsafe_allow_html=True)
                if view:
                    idx = st.selectbox("选择记录", list(range(len(view))), key="hist_pick",
                                       format_func=lambda i: f'{view[i].get("time","")} · '
                                                             f'{view[i].get("cell","")} · '
                                                             f'{view[i].get("result_cn") or view[i].get("result","")}')
                    r = view[idx]
                    if r.get("thumb"):
                        st.markdown(f'<img src="{r["thumb"]}" style="width:100%;border-radius:2px;'
                                    f'border:1px solid #1b2a6b;"/>', unsafe_allow_html=True)
                    st.markdown(kv_html([
                        ("时间", r.get("time", "")),
                        ("来源", f'<span style="font-size:11px;">{r.get("source","")}</span>'),
                        ("槽号", r.get("cell", "")),
                        ("状态", f'{r.get("result_cn","")}（{r.get("result","")}）'),
                        ("置信度", f'{float(r.get("conf") or 0)*100:.1f}%'),
                        ("μ / σ / S", f'{r.get("gray_mean")} / {r.get("contrast")} / {r.get("saturation")}'),
                        ("V_crust / V_node", f'{r.get("crust_speed")} / {r.get("node_speed")}'),
                        ("采样帧", r.get("frames")),
                        ("推算加药", (r.get("dosing") or {}).get("action", "—")),
                    ]), unsafe_allow_html=True)
                    if r.get("ai_report"):
                        st.markdown('<div class="muted" style="margin-top:6px;">AI 报告摘录</div>'
                                    f'<div style="max-height:18vh;overflow-y:auto;font-size:12px;'
                                    f'line-height:1.55;color:#d8c9ff;white-space:pre-wrap;">'
                                    f'{r["ai_report"][:1200]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="muted">左侧筛选后无可选记录</div>', unsafe_allow_html=True)


# ============================================================
# 5) 系统设置
# ============================================================
def page_settings():
    ss = st.session_state
    perf = ss.analyzer.perf()
    stats = history_store.stats()
    avg_ms = perf["avg_infer_ms"] or 0.0
    fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0

    # ---- 性能总览（原先在「性能监控」页，按需求整合到这里）----
    with st.container(border=True):
        st.markdown('<h4 class="ptitle">性能指标总览</h4>', unsafe_allow_html=True)
        stat_cards([
            ("平均单帧耗时", f"{avg_ms} ms", CYAN),
            ("等效吞吐", f"{fps:.1f} FPS" if fps else "—", "#3ddc84" if fps else DIM),
            ("累计分析帧", f"{perf['frames']}", "#8ba4ff"),
            ("推理设备", f"{str(perf['device']).upper()}", "#ffb020"),
            ("历史记录", f"{stats['total']}", "#a58bff"),
        ])

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="panel"><h4>大模型后端</h4>'
                    f'<div class="num" style="font-size:12px;color:#bfe6ff;">{llm_backend_summary()}</div>'
                    '<div class="muted" style="margin-top:7px;">API Key 写在项目根目录的 '
                    '<span class="num">.env</span>（DASHSCOPE_API_KEY），'
                    '或改 <span class="num">config/llm_api.json</span>。</div></div>',
                    unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<h4 class="ptitle">分析参数</h4>', unsafe_allow_html=True)
            st.session_state.sample_frames = int(st.slider(
                "视频采样帧数（越多趋势越细，耗时线性增加）", 6, 60,
                int(st.session_state.sample_frames), step=2, key="set_frames"))

        st.markdown('<div class="panel"><h4>报警判定规则</h4>'
                    '<div class="muted" style="line-height:1.65;">三分类为偏高/偏低且持续 &gt; 10 秒、'
                    '其间平均置信度 &gt; 90% 时触发报警；分类采用 5 帧滑动窗口平滑 + '
                    '连续 5 次同类别才切换的迟滞策略，抑制抖动。</div></div>', unsafe_allow_html=True)

    with c2:
        from algo_web_bridge import RESNET_WEIGHT, YOLO_WEIGHT
        rows = []
        for name, path in (("YOLO best.pt", YOLO_WEIGHT), ("DC-MMGAT .pth", RESNET_WEIGHT)):
            exists = os.path.isfile(path)
            badge = ('<span class="badge badge-ok">就位</span>' if exists
                     else '<span class="badge badge-warn">缺失</span>')
            size = f"{os.path.getsize(path)/1048576:.1f} MB" if exists else "—"
            rows.append((name, f'{badge} <span class="num">{size}</span>'))
        st.markdown('<div class="panel"><h4>模型权重</h4>' + kv_html(rows) +
                    f'<div class="muted" style="margin-top:6px;">推理设备 '
                    f'<span class="num">{get_device()}</span>，权重全部就位：'
                    f'{"是" if models_ready() else "否"}</div></div>', unsafe_allow_html=True)

        s = history_store.stats()
        st.markdown('<div class="panel"><h4>历史数据</h4>'
                    + kv_html([("存储位置", f'<span style="font-size:11px;">{history_store.HISTORY_PATH}</span>'),
                               ("记录数 / 上限", f'{s["total"]} / {history_store.MAX_RECORDS}')])
                    + '</div>', unsafe_allow_html=True)
        if st.button("🗑 清空全部历史", key="set_clear_hist", use_container_width=True):
            history_store.clear_records()
            st.success("已清空")


# ============================================================
# 顶部标题 + 顶部导航
# ============================================================
def top_header():
    c1, c2 = st.columns([8.0, 3.0])
    with c1:
        st.markdown(
            '<div class="app-title-wrap">'
            '<div class="app-title-bar"></div>'
            '<div class="app-title">基于<span class="accent">YOLO</span>火眼识别与'
            '多模态大模型<span class="accent">铝电解诊断系统</span></div>'
            '</div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="hdr-right">{datetime.now():%Y-%m-%d %H:%M:%S}</div>',
                    unsafe_allow_html=True)


def top_nav():
    """顶部导航：下划线标签（tab 风格），选中项底部亮条 + 白字。"""
    with st.container(key="tabbar"):
        cols = st.columns(len(PAGES))
        for (key, label), col in zip(PAGES, cols):
            with col:
                active = st.session_state.page == key
                if st.button(label, key=f"nav_{key}",
                             type="primary" if active else "secondary",
                             use_container_width=True):
                    navigate(key)


# ============================================================
# 主入口
# ============================================================
ANALYSIS_PAGES = ("perf", "diagnose", "dosing")


def main():
    top_header()
    top_nav()

    page = st.session_state.page
    # 控制条现在**所有页面都渲染**：
    # ① 上传组件的状态不会被 Streamlit 清理，载入的视频跨页面一直都在；
    # ② 任何页面都能看到当前载入的源，也能随时换源。
    control_bar()

    # 整页半透明外框：用 st.container(key=...) 生成 .st-key-pageframe 供 CSS 定尺寸
    with st.container(key="pageframe"):
        if page == "perf":
            page_perf()
        elif page == "diagnose":
            page_diagnose()
        elif page == "dosing":
            page_dosing()
        elif page == "history":
            page_history()
        else:
            page_settings()


main()
