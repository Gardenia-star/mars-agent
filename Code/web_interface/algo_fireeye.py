# algo_fireeye.py
# -*- coding: utf-8 -*-

import time
from collections import deque
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

# =========【配置路径】=========
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
YOLO_WEIGHT = os.path.join(_project_root, "weights", "best.pt")
RESNET_WEIGHT = os.path.join(_project_root, "weights", "resnet18_fireeye_3cls.pth")
# ============================

# ========== 全局常量 ==========
CLASS_NAMES = ["C", "H", "N"]
CLASS_CN = {"C": "偏低", "H": "偏高", "N": "正常", "NoTarget": "搜索中..."}
CLASS_EN = {"C": "COLD", "H": "HOT", "N": "NORMAL", "NoTarget": "SCANNING"}
TEMP_ORDER = {"C": 0, "N": 1, "H": 2}

# --- 算法参数 ---
IMG_SIZE = 224
CONF_EYE = 0.18
CONF_FIRE = 0.15
IOU_MERGE = 0.5
MIN_SIDE = 20
UNION_PAD_X = 0.05
UNION_PAD_Y_TOP = 0.02
UNION_PAD_Y_BOTTOM = 0.02
EXTRA_UP_NO_FIRE = 0.40
MAX_HOLD = 5
CLS_EVERY = 2
K_SMOOTH = 5
SWITCH_THR = 0.8

# ================== 图像大模型中文提示词 ==================
LLAVA_PROMPT_CN = """
你是一名长期驻厂的铝电解工艺工程师，熟悉电解质过热度识别、火眼视频诊断、
结壳 / 角点动态特征 (V_crust, V_node) 等内容。

现在给你一帧来自“火眼监控”的 ROI 图像，该图像已经通过目标检测裁剪，
主要包含电解质表面和火焰区域。请你根据图像进行诊断分析，并**只用简体中文回答**。

请严格按以下结构输出（标题也要保留）：

【一、图像整体描述】
- 用 2~3 句话描述当前火眼画面的整体情况（亮度、颜色分布、是否有明显结壳块、黑色冷料等）。

【二、静态特征评估】
结合画面的亮度和纹理，对下列指标给出定性判断（“偏低 / 正常 / 偏高”），
并用自然语言解释原因（从颜色、纹理、细节清晰度等角度）：
- 平均灰度 μ_x（反映电解质表面整体明暗）
- 对比度 σ_x（灰度变化，是否“花”“杂乱”）
- 饱和度 s_x（彩色成分，尤其是红色/黄色强度）

【三、动态特征与过热度判断】
根据火焰形态、亮区位置、边缘清晰度等，判断当前电解槽过热度状态
（偏低 / 正常 / 偏高），并说明：
- 你是依据哪些局部特征做出判断；
- 结壳速率 V_crust 可能是偏高、正常还是偏低；
- 角点速率 V_node 可能的变化趋势（结壺是否在破裂 / 剥落）。

【四、风险等级与工艺建议】
- 给出一个“风险等级”（低 / 中 / 高），
- 提出不超过 3 条、尽量可落地的工艺建议，例如：
  · 是否建议微调槽电压（↑ / ↓ 及幅度区间）；
  · 是否需要调整加料节奏、捞渣或检查阳极状态；
  · 是否建议加强短期监测或人工巡检。

【五、需要持续观察的现象】
- 列出 1~2 个需要在接下来 5~10 分钟重点关注的现象，
  如“火眼亮区是否继续上移”“黑色冷料是否增多”“结壳裂纹是否扩展”等。

要求：
- 全程使用专业但易懂的简体中文；
- 不要出现“作为 AI 模型”“我只能”之类的措辞；
- 如果信息不足，请说明假设条件，再给出慎重的建议。
"""



# ========== 辅助函数 ==========

def iou(a, b):
    x1 = max(a[0], b[0]);
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]);
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def merge_overlaps(boxes, thr=0.5):
    boxes = boxes[:]
    used = [False] * len(boxes)
    out = []
    for i in range(len(boxes)):
        if used[i]: continue
        g = [boxes[i]]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if not used[j] and iou(boxes[i], boxes[j]) >= thr:
                g.append(boxes[j])
                used[j] = True
        xs = [b[0] for b in g];
        ys = [b[1] for b in g]
        xe = [b[2] for b in g];
        ye = [b[3] for b in g]
        out.append([min(xs), min(ys), max(xe), max(ye)])
    return out


def sharpness_score(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_resnet(weight_path, num_classes=3, device="cuda"):
    try:
        ckpt = torch.load(weight_path, map_location=device)
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        else:
            state_dict = ckpt

        model = models.resnet18(weights=None)
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"DC-MMGAT 加载错误: {e}")
        return None


# ========== 物理特征计算 ==========
def calc_static_features(roi_img):
    if roi_img is None or roi_img.size == 0:
        return 0.0, 0.0, 0.0, 0

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    img_float = gray.astype(np.float32)
    N = img_float.size

    mu_x = np.mean(img_float)
    sigma_x = np.std(img_float, ddof=1) if N > 1 else 0.0

    if mu_x > 1e-6 and N > 1:
        ratio_sq = (img_float / mu_x) ** 2
        sum_ratio = np.sum(ratio_sq)
        s_x = np.sqrt(sum_ratio / (N - 1))
    else:
        s_x = 0.0

    corners = cv2.goodFeaturesToTrack(
        enhanced_gray, maxCorners=150, qualityLevel=0.005, minDistance=3
    )
    num_corners = corners.shape[0] if corners is not None else 0

    return float(mu_x), float(sigma_x), float(s_x), int(num_corners)


# ========== 绘图辅助 ==========
def draw_hud(frame, text_lines, color_theme=(0, 255, 0)):
    h, w, _ = frame.shape
    overlay = frame.copy()
    bar_height = 90
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (20, 20, 20), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (0, 0), (10, bar_height), color_theme, -1)
    x = 25
    y = 35
    for line in text_lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.9, color_theme, 1, cv2.LINE_AA)
        y += 35


# ========== 核心工作线程 ==========
class AlgoWorker(QThread):
    data_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(dict)

    def __init__(self, source, model_type="DC-MMGAT", parent=None):
        super().__init__(parent)
        self.source = source
        self.model_type = model_type
        self._stop_flag = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 绘图开关
        self.draw_fire = True
        self.draw_eye = True
        self.draw_union = True

    def set_draw_flags(self, show_fire, show_eye, show_union):
        self.draw_fire = show_fire
        self.draw_eye = show_eye
        self.draw_union = show_union

    def stop(self):
        self._stop_flag = True

    @torch.no_grad()
    def run(self):
        self.yolo_model = YOLO(YOLO_WEIGHT)
        self.resnet = load_resnet(RESNET_WEIGHT, num_classes=len(CLASS_NAMES), device=self.device)
        self.tfm = build_transform()

        name2id = {v: k for k, v in self.yolo_model.names.items()}
        self.EYE_ID = name2id.get("fireeye", 0)
        self.FIRE_ID = name2id.get("fire", 1)

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.finished_signal.emit({"status": "Error", "msg": "无法打开视频源"})
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or fps > 200: fps = 25.0
        frame_interval = 1.0 / fps

        prob_queue = deque(maxlen=K_SMOOTH)
        current_stable_cls_idx = None
        candidate_cls_idx = None
        candidate_hold_count = 0

        last_mu_x = None
        last_corners = None
        last_time_sec = 0.0

        # 【新增】严重报警逻辑变量
        abnormal_start_time = None  # 记录异常开始的时间点
        abnormal_conf_buffer = []  # 记录异常期间的置信度
        alarm_triggered = False  # 是否触发报警

        is_camera = isinstance(self.source, int)
        start_clock = time.time()
        frame_cnt = 0

        # 记录最后的状态用于历史记录
        last_cls_for_history = "NoTarget"

        while not self._stop_flag:
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok: break

            frame_cnt += 1
            elapsed_time = time.time() - start_clock if is_camera else frame_cnt / fps
            clean = frame.copy()

            # 1. YOLO
            r = self.yolo_model.predict(frame, imgsz=640, conf=min(CONF_EYE, CONF_FIRE),
                                        device=self.device, verbose=False)[0]

            if len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cfd = r.boxes.conf.cpu().numpy()
            else:
                xyxy, cls, cfd = [], [], []

            eyes_raw = [xyxy[i].tolist() for i, c in enumerate(cls) if c == self.EYE_ID and cfd[i] >= CONF_EYE]
            fires = [xyxy[i].tolist() for i, c in enumerate(cls) if c == self.FIRE_ID and cfd[i] >= CONF_FIRE]

            # 2. FireEye Filter
            def is_eye(b):
                w0, h0 = b[2] - b[0], b[3] - b[1]
                return (w0 >= 16 and h0 >= 16 and 0.5 <= (w0 / h0) <= 4.0)

            eyes_sorted = sorted(eyes_raw, key=lambda b: ((b[2] - b[0]) * (b[3] - b[1]), -b[3]), reverse=True)
            eye_main = next((cand for cand in eyes_sorted if is_eye(cand)), None)
            eyes = [eye_main] if eye_main else []
            fires = merge_overlaps(fires, IOU_MERGE)

            # 3. Draw
            if self.draw_eye:
                for e in eyes:
                    cv2.rectangle(frame, (int(e[0]), int(e[1])), (int(e[2]), int(e[3])), (255, 128, 0), 3)

            if self.draw_fire:
                for f in fires:
                    cv2.rectangle(frame, (int(f[0]), int(f[1])), (int(f[2]), int(f[3])), (0, 200, 0), 3)

            # 4. Union ROI
            roi_img = None
            have_union = False

            if eyes or fires:
                all_boxes = eyes + fires
                ux1 = min(b[0] for b in all_boxes);
                uy1 = min(b[1] for b in all_boxes)
                ux2 = max(b[2] for b in all_boxes);
                uy2 = max(b[3] for b in all_boxes)
                bw, bh = ux2 - ux1, uy2 - uy1
                ux1 = max(0, int(ux1 - bw * UNION_PAD_X))
                ux2 = min(frame.shape[1], int(ux2 + bw * UNION_PAD_X))
                uy1 = max(0, int(uy1 - int(bh * UNION_PAD_Y_TOP)))
                uy2 = min(frame.shape[0], int(uy2 + int(bh * UNION_PAD_Y_BOTTOM)))

                if (ux2 - ux1) >= MIN_SIDE and (uy2 - uy1) >= MIN_SIDE:
                    have_union = True
                    roi_img = clean[uy1:uy2, ux1:ux2].copy()
                    if self.draw_union:
                        cv2.rectangle(frame, (ux1, uy1), (ux2, uy2), (255, 255, 0), 4)

            # 5. Features & DC-MMGAT（ResNet18 骨干）
            feat_mu, feat_sigma, feat_sat, feat_corners = 0.0, 0.0, 0.0, 0
            v_crust, v_node = 0.0, 0.0
            probs = [0.0, 0.0, 0.0]
            trend_text = ""
            final_cls_name = "NoTarget"
            final_conf = 0.0

            if have_union:
                feat_mu, feat_sigma, feat_sat, feat_corners = calc_static_features(roi_img)
                if last_mu_x is not None:
                    dt = elapsed_time - last_time_sec
                    if dt > 0.01:
                        v_crust = abs(feat_mu - last_mu_x) / dt
                        v_node = abs(feat_corners - last_corners) / dt
                last_mu_x = feat_mu
                last_corners = feat_corners
                last_time_sec = elapsed_time

                if self.resnet is not None and frame_cnt % CLS_EVERY == 0:
                    try:
                        if sharpness_score(roi_img) > 10:
                            img_pil = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
                            x = self.tfm(img_pil).unsqueeze(0).to(self.device)
                            logits = self.resnet(x)[0]
                            p = torch.softmax(logits, dim=0).cpu().numpy()
                            prob_queue.append(p)
                    except:
                        pass

                if prob_queue:
                    avg_prob = np.mean(np.stack(prob_queue), axis=0)
                    probs = avg_prob.tolist()
                    instant_max_idx = int(np.argmax(avg_prob))
                    instant_max_prob = float(avg_prob[instant_max_idx])

                    if instant_max_prob > SWITCH_THR:
                        if instant_max_idx == candidate_cls_idx:
                            candidate_hold_count += 1
                        else:
                            candidate_cls_idx = instant_max_idx
                            candidate_hold_count = 1
                        if candidate_hold_count >= MAX_HOLD:
                            current_stable_cls_idx = instant_max_idx
                    else:
                        candidate_hold_count = 0

                    if current_stable_cls_idx is not None:
                        final_cls_name = CLASS_NAMES[current_stable_cls_idx]
                        final_conf = probs[current_stable_cls_idx]
                        last_cls_for_history = final_cls_name  # 记录状态

                        # 趋势
                        other_ids = [i for i in range(3) if i != current_stable_cls_idx]
                        best_other = max(other_ids, key=lambda i: avg_prob[i])
                        other_name = CLASS_NAMES[best_other]
                        other_prob = float(avg_prob[best_other])
                        if other_prob > 0.1 and other_prob < SWITCH_THR:
                            if TEMP_ORDER[other_name] > TEMP_ORDER[final_cls_name]:
                                trend_text = "Heating..."
                            elif TEMP_ORDER[other_name] < TEMP_ORDER[final_cls_name]:
                                trend_text = "Cooling..."
            else:
                prob_queue.clear()
                current_stable_cls_idx = None
                last_mu_x = None
                last_corners = None
                last_cls_for_history = "NoTarget"

            # 6. 【核心新增】严重报警逻辑 (10秒 & 90%置信度)
            if final_cls_name in ["H", "C"]:
                if abnormal_start_time is None:
                    abnormal_start_time = time.time()  # 开始计时
                abnormal_conf_buffer.append(final_conf)

                # 计算持续时间
                duration = time.time() - abnormal_start_time
                # 计算平均置信度
                avg_conf_period = sum(abnormal_conf_buffer) / len(abnormal_conf_buffer) if abnormal_conf_buffer else 0

                # 判断条件
                if duration > 10.0 and avg_conf_period > 0.90:
                    alarm_triggered = True
                else:
                    alarm_triggered = False
            else:
                # 状态正常或无目标，重置计时器
                abnormal_start_time = None
                abnormal_conf_buffer = []
                alarm_triggered = False

            # 7. Draw HUD
            color_map = {
                "C": (255, 100, 0), "N": (0, 255, 0), "H": (0, 0, 255),
                "NoTarget": (128, 128, 128)
            }
            theme_color = color_map.get(final_cls_name, (128, 128, 128))
            en_text = CLASS_EN.get(final_cls_name, "SCANNING")
            conf_text = f"CONF: {final_conf:.2f}" if final_cls_name != "NoTarget" else "Searching..."
            hud_lines = [f"{en_text} | {conf_text}"]
            if trend_text: hud_lines.append(trend_text)
            draw_hud(frame, hud_lines, theme_color)

            # 8. Send Data
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qimg_frame = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888).copy()
            qimg_roi = None
            if roi_img is not None:
                rgb_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
                rh, rw, rch = rgb_roi.shape
                qimg_roi = QImage(rgb_roi.data, rw, rh, rch * rw, QImage.Format_RGB888).copy()

            data_packet = {
                "frame_img": qimg_frame,
                "roi_img": qimg_roi,
                "timestamp": elapsed_time,
                "gray_mean": feat_mu,
                "contrast": feat_sigma,
                "saturation": feat_sat,
                "crust_speed": v_crust,
                "node_speed": v_node,
                "cls_result": final_cls_name,
                "cls_conf": final_conf,
                "cls_probs": probs,
                "trend": trend_text,
                "alarm": alarm_triggered  # 发送报警标志
            }

            self.data_signal.emit(data_packet)

            process_time = time.time() - loop_start
            wait_time = frame_interval - process_time
            if wait_time > 0: time.sleep(wait_time)

        cap.release()
        # 9. 【历史记录修复】结束时，把 source 和 最后状态传出去
        self.finished_signal.emit({
            "status": "Finished",
            "source": self.source,
            "last_cls": last_cls_for_history
        })