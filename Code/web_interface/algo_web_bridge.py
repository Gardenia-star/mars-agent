# algo_web_bridge.py
# -*- coding: utf-8 -*-
"""
Web 端火眼分析器 —— 与桌面端 AlgoWorker **同一套推理链路**，不再是占位桩。

流程（逐帧）：
    YOLO(fireeye / fire) 检测
        → 火眼最优框筛选 + 火焰框 IoU 合并
        → 并集 ROI（含 padding）
        → 静态特征 μ / σ / S / 角点数
        → 动态特征 V_crust / V_node（帧间差分）
        → DC-MMGAT 三分类（ResNet18 骨干）+ 滑动窗口平滑 + 迟滞切换
        → 趋势 (Heating/Cooling) 与持续报警判定

对外接口：
    FireEyeAnalyzer(model_type="DC-MMGAT").analyze_frame(bgr) -> dict
    FireEyeAnalyzer.analyze_batch(frames)                  -> list[dict]
    load_bgr_from_bytes(data) / load_frames_from_bytes(data, max_frames)

注意：本模块**不再依赖 PyQt5**（原实现从 algo_fireeye 借常量，顺带把 Qt 拖了进来）。
"""

from __future__ import annotations

import os
import tempfile
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

# =========【配置路径】=========
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_WEIGHT = os.path.join(_PROJECT_ROOT, "weights", "best.pt")
RESNET_WEIGHT = os.path.join(_PROJECT_ROOT, "weights", "resnet18_fireeye_3cls.pth")

# ========== 与桌面端保持一致的常量 ==========
CLASS_NAMES = ["C", "H", "N"]
CLASS_CN = {"C": "偏低", "H": "偏高", "N": "正常", "NoTarget": "搜索中..."}
CLASS_EN = {"C": "COLD", "H": "HOT", "N": "NORMAL", "NoTarget": "SCANNING"}
TEMP_ORDER = {"C": 0, "N": 1, "H": 2}

IMG_SIZE = 224
CONF_EYE = 0.18
CONF_FIRE = 0.15
IOU_MERGE = 0.5
MIN_SIDE = 20
UNION_PAD_X = 0.05
UNION_PAD_Y_TOP = 0.02
UNION_PAD_Y_BOTTOM = 0.02
MAX_HOLD = 5
CLS_EVERY = 2
K_SMOOTH = 5
SWITCH_THR = 0.8

# 报警：H/C 持续超过该秒数且期间平均置信度高于阈值
ALARM_HOLD_SEC = 10.0
ALARM_CONF = 0.90

# 画面标注配色（BGR）
COLOR_EYE = (255, 128, 0)
COLOR_FIRE = (0, 200, 0)
COLOR_UNION = (255, 255, 0)
COLOR_MAP = {"C": (255, 100, 0), "N": (0, 255, 0), "H": (0, 0, 255), "NoTarget": (128, 128, 128)}


# ================== 基础几何 / 图像工具 ==================
def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def merge_overlaps(boxes: List[List[float]], thr: float = 0.5) -> List[List[float]]:
    boxes = [list(b) for b in boxes]
    used = [False] * len(boxes)
    out: List[List[float]] = []
    for i in range(len(boxes)):
        if used[i]:
            continue
        group = [boxes[i]]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if not used[j] and iou(boxes[i], boxes[j]) >= thr:
                group.append(boxes[j])
                used[j] = True
        out.append([
            min(b[0] for b in group), min(b[1] for b in group),
            max(b[2] for b in group), max(b[3] for b in group),
        ])
    return out


def sharpness_score(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_resnet(weight_path: str, num_classes: int = 3, device: str = "cpu"):
    try:
        ckpt = torch.load(weight_path, map_location=device)
        state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        return model
    except Exception as e:  # pragma: no cover
        print(f"[algo_web_bridge] DC-MMGAT 加载失败: {e}")
        return None


def calc_static_features(roi_img: np.ndarray) -> Tuple[float, float, float, int]:
    """μ（平均灰度）、σ（对比度）、S（归一化二阶矩）、角点数。"""
    if roi_img is None or roi_img.size == 0:
        return 0.0, 0.0, 0.0, 0
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    img_float = gray.astype(np.float32)
    n = img_float.size
    mu = float(np.mean(img_float))
    sigma = float(np.std(img_float, ddof=1)) if n > 1 else 0.0
    if mu > 1e-6 and n > 1:
        s = float(np.sqrt(np.sum((img_float / mu) ** 2) / (n - 1)))
    else:
        s = 0.0
    corners = cv2.goodFeaturesToTrack(enhanced, maxCorners=150, qualityLevel=0.005, minDistance=3)
    return mu, sigma, s, int(corners.shape[0]) if corners is not None else 0


# ================== 模型缓存（进程级，多 session 共享） ==================
_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_models(device: str) -> Dict[str, Any]:
    if device not in _MODEL_CACHE:
        yolo = YOLO(YOLO_WEIGHT)
        name2id = {v: k for k, v in yolo.names.items()}
        _MODEL_CACHE[device] = {
            "yolo": yolo,
            "resnet": load_resnet(RESNET_WEIGHT, len(CLASS_NAMES), device),
            "tfm": build_transform(),
            "EYE_ID": name2id.get("fireeye", 0),
            "FIRE_ID": name2id.get("fire", 1),
        }
    return _MODEL_CACHE[device]


def models_ready() -> bool:
    """给界面用：模型权重是否就位。"""
    return os.path.isfile(YOLO_WEIGHT) and os.path.isfile(RESNET_WEIGHT)


# ================== 核心分析器 ==================
class FireEyeAnalyzer:
    """
    有状态的单帧分析器：内部维护滑动窗口、动态特征基线与报警计时，
    因此**必须按时间顺序喂帧**才能得到正确的 V_crust / V_node / 趋势 / 报警。
    """

    def __init__(self, model_type: str = "DC-MMGAT", device: Optional[str] = None):
        self.model_type = model_type
        self.device = device or get_device()
        self.reset()

    # ---------- 状态 ----------
    def reset(self) -> None:
        self.prob_queue: deque = deque(maxlen=K_SMOOTH)
        self.frame_cnt = 0
        self.stable_idx: Optional[int] = None
        self.cand_idx: Optional[int] = None
        self.cand_hold = 0
        self.last_mu: Optional[float] = None
        self.last_corners: Optional[int] = None
        self.last_time: Optional[float] = None
        self.wall_start = time.time()
        self.alarm_start: Optional[float] = None
        self.alarm_conf: List[float] = []
        self.last_cls = "NoTarget"
        self.infer_ms_total = 0.0
        self.frames_analyzed = 0

    # ---------- 一帧 ----------
    @torch.no_grad()
    def analyze_frame(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None,
                      immediate: bool = False) -> Dict[str, Any]:
        models = _load_models(self.device)
        self.frame_cnt += 1
        t0 = time.perf_counter()

        h, w = frame_bgr.shape[:2]
        now = float(timestamp) if timestamp is not None else (time.time() - self.wall_start)
        clean = frame_bgr.copy()
        frame = frame_bgr.copy()

        # 1) YOLO 检测
        res = models["yolo"].predict(
            frame, imgsz=640, conf=min(CONF_EYE, CONF_FIRE), device=self.device, verbose=False
        )[0]
        if len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cfd = res.boxes.conf.cpu().numpy()
        else:
            xyxy, cls, cfd = [], [], []

        eyes_raw = [xyxy[i].tolist() for i, c in enumerate(cls) if c == models["EYE_ID"] and cfd[i] >= CONF_EYE]
        fires = [xyxy[i].tolist() for i, c in enumerate(cls) if c == models["FIRE_ID"] and cfd[i] >= CONF_FIRE]

        # 2) 火眼筛选：面积最大且长宽比合理的一个
        def is_eye(b) -> bool:
            bw, bh = b[2] - b[0], b[3] - b[1]
            return bw >= 16 and bh >= 16 and 0.5 <= (bw / max(bh, 1e-6)) <= 4.0

        eyes_sorted = sorted(eyes_raw, key=lambda b: ((b[2] - b[0]) * (b[3] - b[1]), -b[3]), reverse=True)
        eye_main = next((c for c in eyes_sorted if is_eye(c)), None)
        eyes = [eye_main] if eye_main else []
        fires = merge_overlaps(fires, IOU_MERGE)

        # 3) 并集 ROI
        roi_img = None
        union_box = None
        have_union = False
        if eyes or fires:
            all_boxes = eyes + fires
            ux1 = min(b[0] for b in all_boxes)
            uy1 = min(b[1] for b in all_boxes)
            ux2 = max(b[2] for b in all_boxes)
            uy2 = max(b[3] for b in all_boxes)
            bw, bh = ux2 - ux1, uy2 - uy1
            ux1 = int(max(0, ux1 - bw * UNION_PAD_X))
            ux2 = int(min(w, ux2 + bw * UNION_PAD_X))
            uy1 = int(max(0, uy1 - bh * UNION_PAD_Y_TOP))
            uy2 = int(min(h, uy2 + bh * UNION_PAD_Y_BOTTOM))
            if (ux2 - ux1) >= MIN_SIDE and (uy2 - uy1) >= MIN_SIDE:
                have_union = True
                union_box = [ux1, uy1, ux2, uy2]
                roi_img = clean[uy1:uy2, ux1:ux2].copy()

        # 4) 特征 + 分类
        mu = sigma = sat = 0.0
        corners = 0
        v_crust = v_node = 0.0
        probs = [0.0, 0.0, 0.0]
        trend = ""
        final_cls, final_conf = "NoTarget", 0.0

        if have_union:
            mu, sigma, sat, corners = calc_static_features(roi_img)
            if self.last_mu is not None and self.last_time is not None:
                dt = now - self.last_time
                if dt > 0.01:
                    v_crust = abs(mu - self.last_mu) / dt
                    v_node = abs(corners - (self.last_corners or 0)) / dt
            self.last_mu, self.last_corners, self.last_time = mu, corners, now

            should_cls = models["resnet"] is not None and (immediate or self.frame_cnt % CLS_EVERY == 0)
            if should_cls:
                try:
                    if sharpness_score(roi_img) > 10:
                        pil = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
                        x = models["tfm"](pil).unsqueeze(0).to(self.device)
                        p = torch.softmax(models["resnet"](x)[0], dim=0).cpu().numpy()
                        self.prob_queue.append(p)
                except Exception:
                    pass

            if self.prob_queue:
                avg = np.mean(np.stack(self.prob_queue), axis=0)
                if immediate:
                    # 静帧（上传单张图片）：不做时序迟滞。
                    # 否则按桌面端的 MAX_HOLD/CLS_EVERY 规则，单张图永远返回 NoTarget。
                    self.stable_idx = int(np.argmax(avg))
                else:
                    idx = int(np.argmax(avg))
                    if float(avg[idx]) > SWITCH_THR:
                        if idx == self.cand_idx:
                            self.cand_hold += 1
                        else:
                            self.cand_idx, self.cand_hold = idx, 1
                        if self.cand_hold >= MAX_HOLD:
                            self.stable_idx = idx
                    else:
                        self.cand_hold = 0

                if self.stable_idx is not None:
                    probs = [float(v) for v in avg]
                    final_cls = CLASS_NAMES[self.stable_idx]
                    final_conf = probs[self.stable_idx]
                    self.last_cls = final_cls
                    others = [i for i in range(3) if i != self.stable_idx]
                    best_other = max(others, key=lambda i: avg[i])
                    other_name, other_p = CLASS_NAMES[best_other], float(avg[best_other])
                    if 0.1 < other_p < SWITCH_THR:
                        if TEMP_ORDER[other_name] > TEMP_ORDER[final_cls]:
                            trend = "Heating..."
                        elif TEMP_ORDER[other_name] < TEMP_ORDER[final_cls]:
                            trend = "Cooling..."
        else:
            self.prob_queue.clear()
            self.stable_idx = None
            self.last_mu = self.last_corners = None
            self.last_cls = "NoTarget"

        # 5) 持续报警
        alarm, alarm_secs = False, 0.0
        if final_cls in ("H", "C"):
            if self.alarm_start is None:
                self.alarm_start = now
            self.alarm_conf.append(final_conf)
            alarm_secs = now - self.alarm_start
            avg_conf = sum(self.alarm_conf) / len(self.alarm_conf)
            alarm = alarm_secs > ALARM_HOLD_SEC and avg_conf > ALARM_CONF
        else:
            self.alarm_start, self.alarm_conf = None, []

        infer_ms = (time.perf_counter() - t0) * 1000.0
        self.infer_ms_total += infer_ms
        self.frames_analyzed += 1

        return {
            # 特征
            "gray_mean": round(mu, 2),
            "contrast": round(sigma, 2),
            "saturation": round(sat, 4),
            "corners": corners,
            "crust_speed": round(v_crust, 2),
            "node_speed": round(v_node, 2),
            # 分类
            "cls_result": final_cls,
            "cls_result_cn": CLASS_CN.get(final_cls, final_cls),
            "cls_conf": round(final_conf, 4),
            "cls_probs": [round(p, 4) for p in probs],
            "trend": trend,
            "alarm": bool(alarm),
            "alarm_secs": round(alarm_secs, 1),
            # 目标与图像
            "has_target": have_union,
            "boxes": {"eye": eyes, "fire": fires, "union": union_box},
            "roi_bgr": roi_img,
            "annotated_bgr": annotate(frame, eyes, fires, union_box, final_cls, final_conf, trend),
            "roi_size": None if roi_img is None else (roi_img.shape[1], roi_img.shape[0]),
            # 运行信息
            "infer_ms": round(infer_ms, 1),
            "device": self.device,
            "timestamp": round(now, 3),
        }

    # ---------- 一批（视频采样帧） ----------
    def analyze_batch(self, frames: List[np.ndarray], times: Optional[List[float]] = None,
                      immediate: bool = False) -> List[Dict[str, Any]]:
        out = []
        for k, f in enumerate(frames):
            out.append(self.analyze_frame(f, None if times is None else times[k], immediate=immediate))
        return out

    # ---------- 运行统计 ----------
    def perf(self) -> Dict[str, Any]:
        n = max(1, self.frames_analyzed)
        return {
            "frames": self.frames_analyzed,
            "avg_infer_ms": round(self.infer_ms_total / n, 1),
            "device": self.device,
        }


def annotate(frame, eyes, fires, union_box, cls_name, cls_conf, trend) -> np.ndarray:
    """在帧上画检测框 + 顶部 HUD（与桌面端风格一致）。"""
    out = frame.copy()
    for b in fires:
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), COLOR_FIRE, 3)
    for b in eyes:
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), COLOR_EYE, 3)
    if union_box:
        cv2.rectangle(out, (union_box[0], union_box[1]), (union_box[2], union_box[3]), COLOR_UNION, 4)

    theme = COLOR_MAP.get(cls_name, (128, 128, 128))
    h, w = out.shape[:2]
    bar = 78
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
    cv2.rectangle(out, (0, 0), (9, bar), theme, -1)
    lines = [f"{CLASS_EN.get(cls_name, 'SCANNING')} | "
             f"{'CONF: %.2f' % cls_conf if cls_name != 'NoTarget' else 'Searching...'}"]
    if trend:
        lines.append(trend)
    y = 32
    for ln in lines:
        cv2.putText(out, ln, (22, y), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, ln, (22, y), cv2.FONT_HERSHEY_DUPLEX, 0.85, theme, 1, cv2.LINE_AA)
        y += 32
    return out


# ================== 输入解码 ==================
def load_bgr_from_bytes(img_bytes: bytes) -> Optional[np.ndarray]:
    """上传的图片二进制 → BGR ndarray。"""
    if not img_bytes:
        return None
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def load_frames_from_bytes(video_bytes: bytes, max_frames: int = 24, suffix: str = ".mp4"
                           ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    上传的视频二进制 → 均匀采样的若干帧（用于让动态特征/趋势/报警真正有意义）。

    返回 (frames, info)；info 含 fps / total_frames / sampled / duration。
    """
    frames: List[np.ndarray] = []
    info: Dict[str, Any] = {"fps": 0.0, "total_frames": 0, "sampled": 0, "duration": 0.0}
    if not video_bytes:
        return frames, info

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(video_bytes)
            tmp_path = tf.name
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return frames, info

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and 0 < fps <= 240 else 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        info.update({"fps": round(fps, 2), "total_frames": total,
                     "duration": round(total / fps, 2) if total else 0.0})

        if total > 0:
            idxs = np.linspace(0, max(total - 1, 0), num=min(max_frames, total), dtype=int)
            wanted = set(int(i) for i in idxs)
            i = 0
            while len(frames) < len(wanted):
                ok, fr = cap.read()
                if not ok:
                    break
                if i in wanted:
                    frames.append(fr)
                i += 1
        else:  # 拿不到总帧数时顺序读
            while len(frames) < max_frames:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
        cap.release()
        info["sampled"] = len(frames)
    except Exception as e:  # pragma: no cover
        print(f"[algo_web_bridge] 视频解码失败: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return frames, info


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把一段结果的序列压成"本次分析"的结论（给历史记录用）。"""
    if not results:
        return {}
    classified = [r for r in results if r["cls_result"] != "NoTarget"]
    last = results[-1]
    counts = {"C": 0, "H": 0, "N": 0}
    for r in classified:
        counts[r["cls_result"]] += 1
    return {
        "frames": len(results),
        "classified": len(classified),
        "counts": counts,
        "final_cls": last["cls_result"],
        "final_conf": last["cls_conf"],
        "alarm": any(r["alarm"] for r in results),
        "max_crust": max((r["crust_speed"] for r in results), default=0.0),
        "max_node": max((r["node_speed"] for r in results), default=0.0),
        "avg_gray": round(float(np.mean([r["gray_mean"] for r in results])), 2),
        "avg_contrast": round(float(np.mean([r["contrast"] for r in results])), 2),
        "avg_saturation": round(float(np.mean([r["saturation"] for r in results])), 4),
    }


__all__ = [
    "FireEyeAnalyzer", "load_bgr_from_bytes", "load_frames_from_bytes", "summarize",
    "CLASS_CN", "CLASS_NAMES", "CLASS_EN", "models_ready", "get_device",
    "YOLO_WEIGHT", "RESNET_WEIGHT",
]
