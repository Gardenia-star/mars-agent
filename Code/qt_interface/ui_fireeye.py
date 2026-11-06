# ui_fireeye.py
# -*- coding: utf-8 -*-

import sys
import os
import json
from collections import deque
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QFont, QImage, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QListWidget,
    QDialog,
    QGraphicsView,
    QGraphicsScene,
    QTabWidget,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QMenu,
    QAction,
)

import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from algo_fireeye import AlgoWorker, CLASS_CN, CLASS_NAMES

from llm_service import LLMWorker, backend_summary

# OpenCV 自带的 Qt 插件会把 QT_QPA_PLATFORM_PLUGIN_PATH 指向 cv2/qt/plugins，
# 使得 PyQt5 的 xcb 平台插件无法加载（报 "Could not load the Qt platform plugin xcb"）。
# 因为上面导入 algo_fireeye 时已经 import 了 cv2，这里在创建 QApplication 之前修正回来。
try:
    import PyQt5 as _PyQt5

    _pyqt_plugins = os.path.join(os.path.dirname(_PyQt5.__file__), "Qt5", "plugins")
    if os.path.isdir(_pyqt_plugins):
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        os.environ.setdefault("QT_PLUGIN_PATH", _pyqt_plugins)
except Exception:
    pass

plt.style.use("bmh")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 默认本地视频，可按需修改
DEFAULT_VIDEO = ""  # 留空，用户需手动选择视频文件
HISTORY_FILE = "history_record.json"


# ========== 写死的固定历史记录（不参与删除，始终显示） ==========
FIXED_HISTORY = [
    {"time": "2026-09-17 14:30:48", "cell": "正常槽号", "result": "正常"},
    {"time": "2026-09-17 14:33:57", "cell": "偏高槽号", "result": "偏高"},
]


# ========== 1. AI 大模型分析弹窗 ==========
class AIAnalysisDialog(QDialog):
    def __init__(self, roi_qimage, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤖 工业大模型智能分析 (Llava Local)")
        self.resize(800, 600)
        self.roi_qimage = roi_qimage  # 注意：这里传入的是 QImage

        layout = QHBoxLayout()

        # 左侧：图片
        self.lbl_img = QLabel()
        self.lbl_img.setFixedSize(350, 350)
        self.lbl_img.setStyleSheet("border: 1px solid #ccc; background: #000;")
        self.lbl_img.setAlignment(Qt.AlignCenter)

        if self.roi_qimage and not self.roi_qimage.isNull():
            pix = QPixmap.fromImage(self.roi_qimage)
            self.lbl_img.setPixmap(pix.scaled(350, 350, Qt.KeepAspectRatio))
        else:
            self.lbl_img.setText("图片无效")

        # 右侧：结果展示
        right_layout = QVBoxLayout()

        self.lbl_status = QLabel("正在初始化大模型引擎...")
        self.lbl_status.setStyleSheet("color: blue; font-weight: bold;")

        self.txt_result = QTextEdit()
        self.txt_result.setReadOnly(True)
        self.txt_result.setStyleSheet(
            "font-size: 14px; font-family: 'Microsoft YaHei'; line-height: 1.6;"
        )
        self.txt_result.setPlaceholderText("等待分析结果...")

        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.accept)

        right_layout.addWidget(self.lbl_status)
        right_layout.addWidget(self.txt_result)
        right_layout.addWidget(self.btn_close)

        layout.addWidget(self.lbl_img)
        layout.addLayout(right_layout)
        self.setLayout(layout)

        # --- 自动开始分析 ---
        self.start_analysis()

    def start_analysis(self):
        if self.roi_qimage is None or self.roi_qimage.isNull():
            self.txt_result.setText("错误：没有可用的 ROI 图片。")
            return

        self.lbl_status.setText(
            f"⚡ 正在调用多模态大模型分析...（{backend_summary()}）"
        )
        self.txt_result.setText("正在思考中...")

        # 实例化线程，传入 QImage
        # 用哪个后端由 config/llm_api.json 决定（云端 API 或本地 Ollama）
        self.llm_thread = LLMWorker(self.roi_qimage, model_name="llava")
        self.llm_thread.finished_signal.connect(self.on_success)
        self.llm_thread.error_signal.connect(self.on_error)
        self.llm_thread.start()

    def on_success(self, text):
        self.lbl_status.setText("✅ 分析完成")
        self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
        # 简单的格式化一下 markdown (如果模型返回 markdown)
        self.txt_result.setMarkdown(text)

    def on_error(self, err_msg):
        self.lbl_status.setText("❌ 分析失败")
        self.lbl_status.setStyleSheet("color: red; font-weight: bold;")
        self.txt_result.setText(err_msg)

    def simulate_analysis(self, state: str = "H"):
        """
        state: "H"=偏高, "C"=偏低, "N"=正常
        如果你暂时不想传参数，也可以不管这个 state，当成偏高来写。
        """
        if state == "H":
            status_text = "当前画面呈现明显的『偏高过热』特征。"
            risk_level = "高"
            advice = (
                "1. 建议在 5~10 分钟内小幅下调槽电压（例如 5–15 mV），避免温度继续走高；\n"
                "2. 检查加料是否偏少或节奏不均匀，必要时适当补加冷料并观察火眼响应；\n"
                "3. 注意观察阳极下方是否有明显强光条带，如持续存在应安排人工复核。"
            )
        elif state == "C":
            status_text = "当前画面更接近『偏低 / 偏冷』状态。"
            risk_level = "中"
            advice = (
                "1. 考虑在安全范围内微升槽电压或适当减缓冷料加入速度；\n"
                "2. 关注火眼周边黑色冷料和结壳块的增多情况，防止出现大面积“闷槽”；\n"
                "3. 如发现电压波动增大或电流不稳，应及时通知当班工艺人员巡检。"
            )
        else:
            status_text = "当前画面整体处于『基本正常』区间。"
            risk_level = "低"
            advice = (
                "1. 按现行工艺参数继续运行，注意保持加料节奏稳定；\n"
                "2. 关注火眼亮区是否出现持续上移或明显扩散的趋势；\n"
                "3. 建议结合过程数据（槽电压、噪声、电解质成分）进行综合验证。"
            )

        report = (
            "【一、图像整体描述】\n"
            "火眼 ROI 区域内可以看到电解质表面和火焰区域，亮区主要集中在出铝口附近，"
            "存在一定的颜色梯度变化和结壳纹理。\n\n"
            "【二、静态特征评估（定性）】\n"
            "· 平均灰度 μ_x：整体亮度偏高，说明电解质温度和辐射能量较强；\n"
            "· 对比度 σ_x：纹理细节较丰富，表面存在明显的亮暗交替区域；\n"
            "· 饱和度 s_x：红橙色成分占比明显，反映高温熔体和火焰辐射较强。\n\n"
            "【三、过热度与动态特征分析】\n"
            f"{status_text}\n"
            "结合画面可推测：\n"
            "· 结壳速率 V_crust 可能偏低，壳层被不断冲刷、难以形成稳定保护层；\n"
            "· 角点速率 V_node 显示局部结壳有破裂 / 剥落迹象，需关注是否诱发阳极效应。\n\n"
            "【四、风险等级与工艺建议】\n"
            f"综合判断本帧画面的风险等级：{risk_level}。\n"
            f"{advice}\n\n"
            "【五、后续观察要点】\n"
            "· 关注未来 5–10 分钟内火眼亮区是否继续上移或面积扩大；\n"
            "· 关注黑色冷料和结壳块的变化，防止出现“过热—冷料骤加—闷槽”的来回振荡。"
        )

        self.txt_result.setText(report)


# ========== 2. 统计曲线弹窗 ==========
class StatisticsDialog(QDialog):
    def __init__(self, parent_ui):
        super().__init__(parent_ui)
        self.setWindowTitle("数据统计与监控中心")
        self.resize(1000, 700)
        self.parent_ui = parent_ui
        self.auto_refresh_roi = False

        layout = QVBoxLayout()
        self.tabs = QTabWidget()

        # 物理特征
        self.tab_phy = QWidget()
        self.init_tab_phy()
        self.tabs.addTab(self.tab_phy, "📊 物理特征曲线")

        # 置信度
        self.tab_conf = QWidget()
        self.init_tab_conf()
        self.tabs.addTab(self.tab_conf, "🧠 AI 置信度趋势")

        # ROI
        self.tab_roi = QWidget()
        self.init_tab_roi()
        self.tabs.addTab(self.tab_roi, "👁️ ROI 实时监控")

        layout.addWidget(self.tabs)

        # 远程桌面(Qt VNC 平台插件)下没有窗口管理器，弹窗既没有标题栏也没有 X 关闭按钮，
        # 所以每个弹窗都必须自带「关闭」按钮，否则用户会被困在里面出不来。
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_close = QPushButton("✅ 关闭")
        self.btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_plots)
        self.timer.start(500)

    def init_tab_phy(self):
        layout = QVBoxLayout()
        self.fig_phy = Figure(figsize=(8, 6), dpi=100)
        self.canvas_phy = FigureCanvas(self.fig_phy)
        layout.addWidget(self.canvas_phy)
        self.tab_phy.setLayout(layout)

        self.ax1 = self.fig_phy.add_subplot(221)
        self.ax2 = self.fig_phy.add_subplot(222)
        self.ax3 = self.fig_phy.add_subplot(223)
        self.ax4 = self.fig_phy.add_subplot(224)
        self.fig_phy.tight_layout(pad=3.0)

    def init_tab_conf(self):
        layout = QVBoxLayout()
        self.fig_conf = Figure(figsize=(8, 6), dpi=100)
        self.canvas_conf = FigureCanvas(self.fig_conf)
        layout.addWidget(self.canvas_conf)
        self.tab_conf.setLayout(layout)
        self.ax_conf = self.fig_conf.add_subplot(111)

    def init_tab_roi(self):
        layout = QVBoxLayout()
        self.lbl_roi_show = QLabel("暂无 ROI 数据")
        self.lbl_roi_show.setAlignment(Qt.AlignCenter)
        self.lbl_roi_show.setStyleSheet(
            "background-color: #222; color: #AAA; font-size: 16px;"
        )
        self.lbl_roi_show.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        btn_layout = QHBoxLayout()
        self.btn_manual = QPushButton("📸 手动刷新")
        self.btn_auto = QPushButton("🔄 开启自动刷新")
        self.btn_manual.clicked.connect(self.update_roi_view)
        self.btn_auto.setCheckable(True)
        self.btn_auto.toggled.connect(self.toggle_auto)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_manual)
        btn_layout.addWidget(self.btn_auto)
        btn_layout.addStretch()

        layout.addWidget(self.lbl_roi_show)
        layout.addLayout(btn_layout)
        self.tab_roi.setLayout(layout)

    def toggle_auto(self, checked):
        self.auto_refresh_roi = checked
        self.btn_auto.setText("🟥 停止自动刷新" if checked else "🔄 开启自动刷新")

    def refresh_plots(self):
        if not self.isVisible():
            return
        data = self.parent_ui.data_buffer
        if len(data["time"]) < 2:
            return

        t = list(data["time"])

        if self.tabs.currentIndex() == 0:
            # 物理特征
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()

            self.ax1.plot(t, list(data["gray"]), "b-", alpha=0.8)
            self.ax1.set_title("平均灰度 μ")

            self.ax2.plot(t, list(data["contrast"]), "g-", alpha=0.8)
            self.ax2.set_title("对比度 σ")

            self.ax3.plot(t, list(data["sat"]), "m-", alpha=0.8)
            self.ax3.set_title("饱和度 S")

            self.ax4.plot(t, list(data["v_crust"]), "r-", label="结壳速率 V_crust")
            self.ax4.plot(t, list(data["v_node"]), "k--", label="角点速率 V_node")
            self.ax4.legend()
            self.ax4.set_title("动态速率曲线")

            self.canvas_phy.draw()

        elif self.tabs.currentIndex() == 1:
            # 置信度
            self.ax_conf.clear()
            self.ax_conf.plot(t, list(data["p_c"]), label="🥶 偏低", color="blue")
            self.ax_conf.plot(t, list(data["p_h"]), label="🔥 偏高", color="red")
            self.ax_conf.plot(t, list(data["p_n"]), label="🟢 正常", color="green")
            self.ax_conf.set_ylim(-0.05, 1.05)
            self.ax_conf.legend()
            self.ax_conf.set_title("AI 三分类置信度曲线")
            self.canvas_conf.draw()

        elif self.tabs.currentIndex() == 2 and self.auto_refresh_roi:
            self.update_roi_view()

    def update_roi_view(self):
        img = self.parent_ui.current_roi_img
        if img and not img.isNull():
            pix = QPixmap.fromImage(img)
            scaled = pix.scaled(
                self.lbl_roi_show.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_roi_show.setPixmap(scaled)


# ========== 3. 可缩放视频视图 ==========
class VideoView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = self._scene.addPixmap(QPixmap())

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAlignment(Qt.AlignCenter)
        self.setBackgroundBrush(Qt.black)

        self._first_frame = True

    def reset_view(self):
        self.resetTransform()
        self._first_frame = True

    def setImage(self, qimg):
        pix = QPixmap.fromImage(qimg)
        self._pixmap_item.setPixmap(pix)
        self._scene.setSceneRect(0, 0, pix.width(), pix.height())
        if self._first_frame:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._first_frame = False

    def wheelEvent(self, event):
        # Ctrl + 滚轮 缩放视频
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15
            if event.angleDelta().y() > 0:
                self.scale(factor, factor)
            else:
                self.scale(1.0 / factor, 1.0 / factor)
            event.accept()
        else:
            super().wheelEvent(event)


# ========== 4. 主界面 ==========
class FireEyeUI(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.history_records = []
        self.load_history()  # 先把历史读出来

        # 存放各种时间序列，用于统计曲线
        self.maxlen = 500
        self.data_buffer = {
            "time": deque(maxlen=self.maxlen),
            "gray": deque(maxlen=self.maxlen),
            "contrast": deque(maxlen=self.maxlen),
            "sat": deque(maxlen=self.maxlen),
            "v_crust": deque(maxlen=self.maxlen),
            "v_node": deque(maxlen=self.maxlen),
            "p_c": deque(maxlen=self.maxlen),
            "p_h": deque(maxlen=self.maxlen),
            "p_n": deque(maxlen=self.maxlen),
        }
        self.current_roi_img = None

        self.init_ui()
        self.apply_styles()

        # 统计曲线弹窗
        self.stats_dialog = StatisticsDialog(self)

        # 系统时间定时器
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_sys_time)
        self.clock_timer.start(1000)
        self.update_sys_time()

        # 初始化“最近一次识别记录”显示
        self.refresh_last_record_label()

    # 一些统一的样式
    def apply_styles(self):
        self.setStyleSheet("""
            QWidget { font-family: "Microsoft YaHei"; font-size: 14px; background-color: #f5f5f7; }
            QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 6px; margin-top: 10px; background-color: #fff; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #333; }
            QPushButton { background-color: #007aff; color: white; border-radius: 5px; padding: 6px 10px; border: none; font-weight: bold; }
            QPushButton:hover { background-color: #0062cc; }
            QPushButton#btn_stop { background-color: #ff3b30; }
            QPushButton#btn_ai   { background-color: #5856d6; }
            QToolButton { background-color: #ff9500; color: white; border-radius: 5px; padding: 6px 10px; border: none; font-weight: bold; }
            QToolButton:hover { background-color: #e08600; }
        """)

    def init_ui(self):
        # self.setWindowTitle("多模态铝电解过热度智能识别系统")
        self.setWindowTitle("基于YOLO火眼识别与多模态大模型铝电解诊断系统")
        self.resize(1280, 800)

        # ===== 顶部：工厂 / 分厂 / 时间范围 / 工区 / 槽号 / 模型 / GO =====
        top_layout = QHBoxLayout()

        self.cb_plant = QComboBox()
        self.cb_plant.addItems(["宁夏青铜峡", "某某电解厂"])
        self.cb_plant.setFixedWidth(140)

        self.cb_line = QComboBox()
        self.cb_line.addItems(["一分厂", "二分厂"])
        self.cb_line.setFixedWidth(100)

        self.cb_range = QComboBox()
        self.cb_range.addItems(["一年间", "三个月", "一周"])
        self.cb_range.setFixedWidth(100)

        self.cb_area = QComboBox()
        self.cb_area.addItems(["一工区", "二工区"])
        self.cb_area.setFixedWidth(100)

        self.cb_cell = QComboBox()
        self.cb_cell.setFixedWidth(140)

        self.cb_model = QComboBox()
        self.cb_model.addItems(["DC-MMGAT", "DAE", "AE", "ARAE", "I3D", "F-GMVAE"])
        self.cb_model.setFixedWidth(120)

        self.btn_go = QPushButton("GO")
        self.btn_go.clicked.connect(self.on_go)
        self.btn_go.setFixedWidth(70)

        self.btn_ai = QPushButton("🤖 AI 大模型分析")
        self.btn_ai.setObjectName("btn_ai")
        self.btn_ai.clicked.connect(self.on_ai_analyze)

        self.btn_box_opt = QToolButton()
        self.btn_box_opt.setText("🔲 检测框设置")
        self.btn_box_opt.setPopupMode(QToolButton.InstantPopup)

        menu_box = QMenu(self.btn_box_opt)
        self.act_fire = QAction("显示 Fire 框 (绿)", self, checkable=True)
        self.act_fire.setChecked(True)
        self.act_eye = QAction("显示 FireEye 框 (橙)", self, checkable=True)
        self.act_eye.setChecked(True)
        self.act_union = QAction("显示 并集大框 (青)", self, checkable=True)
        self.act_union.setChecked(True)
        self.act_fire.triggered.connect(self.update_box_display)
        self.act_eye.triggered.connect(self.update_box_display)
        self.act_union.triggered.connect(self.update_box_display)
        menu_box.addAction(self.act_fire)
        menu_box.addAction(self.act_eye)
        menu_box.addAction(self.act_union)
        self.btn_box_opt.setMenu(menu_box)

        for w in [
            self.cb_plant,
            self.cb_line,
            self.cb_range,
            self.cb_area,
            self.cb_cell,
            self.cb_model,
            self.btn_go,
            self.btn_ai,
            self.btn_box_opt,
        ]:
            top_layout.addWidget(w)
        top_layout.addStretch()

        # ===== 中间：左视频 + 右指标 =====
        center_layout = QHBoxLayout()

        self.video_view = VideoView()

        right_box = QGroupBox("📊 实时多模态指标")
        vbox = QVBoxLayout()

        # 系统当前时间（对齐你原来“时间：2025-xx-xx ...”的效果）
        self.lbl_sys_time = QLabel("时间：----")
        self.lbl_sys_time.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.lbl_sys_time.setStyleSheet("color: #000000;")
        vbox.addWidget(self.lbl_sys_time)
        vbox.addSpacing(10)

        # 各种指标
        self.lbl_gray = QLabel("--")
        self.lbl_cont = QLabel("--")
        self.lbl_sat = QLabel("--")
        self.lbl_vc = QLabel("--")
        self.lbl_vn = QLabel("--")

        def add_row(icon, text, lab):
            h = QHBoxLayout()
            l = QLabel(f"{icon} {text}")
            l.setStyleSheet("color: #555;")
            lab.setStyleSheet("color: #0055aa; font-weight: bold;")
            h.addWidget(l)
            h.addStretch()
            h.addWidget(lab)
            vbox.addLayout(h)
            vbox.addSpacing(4)

        add_row("", "平均灰度 μ", self.lbl_gray)
        add_row("", "对比度 σ", self.lbl_cont)
        add_row("", "饱和度 S", self.lbl_sat)
        vbox.addSpacing(10)
        add_row("", "结壳速率 V_crust", self.lbl_vc)
        add_row("", "角点速率 V_node", self.lbl_vn)

        vbox.addSpacing(10)

        # 状态显示块
        self.lbl_res = QLabel("⌛ 等待启动...")
        self.lbl_res.setAlignment(Qt.AlignCenter)
        self.lbl_res.setStyleSheet(
            "background-color: #e0e0e0; color: #666; "
            "border-radius: 5px; padding: 10px; font-size: 16px;"
        )
        vbox.addWidget(self.lbl_res)

        # 严重报警提示
        # 【原来的那行删掉，改成下面两行】
        self.lbl_alert = QLabel()
        self.lbl_alert.setAlignment(Qt.AlignCenter)
        self.lbl_alert.setText("")  # 初始没内容

        self.lbl_alert.setStyleSheet("""
            background-color: #d32f2f;
            color: #ffeb3b;
            font-weight: bold;
            font-size: 18px;
            border: 3px solid #b71c1c;
            border-radius: 8px;
            padding: 18px;
        """)
        self.lbl_alert.hide()
        vbox.addSpacing(8)
        vbox.addWidget(self.lbl_alert)

        vbox.addStretch()
        right_box.setLayout(vbox)

        center_layout.addWidget(self.video_view, stretch=7)
        center_layout.addWidget(right_box, stretch=3)

        # ===== 底部：最近一次识别记录 + 按钮 =====
        self.lbl_last_record = QLabel("最近一次识别记录：无")
        self.lbl_last_record.setStyleSheet("color: #555555; padding: 4px 6px;")
        self.lbl_last_record.setFont(QFont("Microsoft YaHei", 9))

        bot_btn_layout = QHBoxLayout()
        self.btn_file = QPushButton("📂 本地视频")
        self.btn_cam = QPushButton("📹 实时摄像头")
        self.btn_stat = QPushButton("📉 统计曲线")
        self.btn_hist = QPushButton("📜 历史记录")
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setObjectName("btn_stop")

        self.btn_file.clicked.connect(self.choose_video)
        self.btn_cam.clicked.connect(self.start_camera)
        self.btn_stat.clicked.connect(lambda: self.stats_dialog.show())
        self.btn_hist.clicked.connect(self.show_history_dialog)
        self.btn_stop.clicked.connect(self.stop_algo)

        for b in [
            self.btn_file,
            self.btn_cam,
            self.btn_stat,
            self.btn_hist,
            self.btn_stop,
        ]:
            b.setMinimumHeight(45)
            bot_btn_layout.addWidget(b)

        # 主布局
        main = QVBoxLayout()

        # ===== 顶部大标题（加粗、放大） =====
        self.lbl_header = QLabel("🔥 基于YOLO火眼识别与多模态大模型铝电解诊断系统")
        self.lbl_header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_header.setStyleSheet("""
            font-size: 22px;
            font-weight: 900;
            color: #222;
            padding: 8px 4px;
        """)
        main.addWidget(self.lbl_header)

        main.addLayout(top_layout)
        main.addLayout(center_layout)
        main.addWidget(self.lbl_last_record)
        main.addLayout(bot_btn_layout)
        self.setLayout(main)

        # 初始化下拉框的槽号（如果有默认视频）
        if os.path.exists(DEFAULT_VIDEO):
            self.add_cell(DEFAULT_VIDEO)

    # ======= 系统时间 =======
    def update_sys_time(self):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.lbl_sys_time.setText(f"时间： {now_str}")

    # ======= 槽号管理 =======
    def add_cell(self, path):
        # 槽号名字: 用文件名，不带路径
        if isinstance(path, int):
            name = "实时摄像头"
        else:
            base = os.path.basename(path)
            stem, _ = os.path.splitext(base)
            name = f"{stem}号槽"
        if self.cb_cell.findText(name) == -1:
            self.cb_cell.addItem(name)
        self.cb_cell.setCurrentText(name)

    def source_to_cell(self, source):
        """把视频来源转换成槽号字符串"""
        # 摄像头
        if isinstance(source, int) or str(source).isdigit():
            return "实时摄像头"

        path = str(source)
        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        # 按你之前的约定：文件名作为槽号
        return f"{name}号槽"

    # ======= 源选择 =======
    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Video (*.mp4 *.avi *.mov)"
        )
        if path:
            self.start_algo(path)

    def start_camera(self):
        self.start_algo(0)

    def on_go(self):
        # 简单：GO 时，如果当前选中的是“实时摄像头”就开摄像头，否则跑 DEFAULT_VIDEO
        txt = self.cb_cell.currentText()
        if "摄像头" in txt:
            self.start_algo(0)
        elif os.path.exists(DEFAULT_VIDEO):
            self.start_algo(DEFAULT_VIDEO)

    # ======= AI 分析 =======
    def on_ai_analyze(self):
        if self.current_roi_img is None:
            QMessageBox.warning(
                self, "提示", "当前未检测到目标区域，无法进行 AI 分析。"
            )
            return
        dlg = AIAnalysisDialog(self.current_roi_img, self)
        dlg.exec_()

    def update_box_display(self):
        if self.worker:
            self.worker.set_draw_flags(
                self.act_fire.isChecked(),
                self.act_eye.isChecked(),
                self.act_union.isChecked(),
            )

    # ======= 启动 / 停止算法线程 =======
    def start_algo(self, source):
        self.stop_algo()
        self.video_view.reset_view()
        self.add_cell(source)

        # 清空曲线缓存
        for k in self.data_buffer:
            self.data_buffer[k].clear()

        model_name = self.cb_model.currentText()
        self.worker = AlgoWorker(source, model_type=model_name)
        self.worker.data_signal.connect(self.update_data)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.set_draw_flags(
            self.act_fire.isChecked(),
            self.act_eye.isChecked(),
            self.act_union.isChecked(),
        )
        self.worker.start()
        self.btn_go.setEnabled(False)
        self.lbl_res.setText("▶ 检测中...")
        self.lbl_alert.hide()

    def stop_algo(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        self.btn_go.setEnabled(True)
        self.lbl_res.setText("⏹ 已停止")
        self.lbl_res.setStyleSheet(
            "background-color: #e0e0e0; color: #666; border-radius: 5px; padding: 10px; font-size: 16px;"
        )
        self.lbl_alert.hide()

    # ======= 接收算法线程数据，刷新右侧指标 + 曲线缓存 =======
    def update_data(self, d):
        # 视频帧 & ROI
        self.video_view.setImage(d["frame_img"])
        self.current_roi_img = d["roi_img"]

        # 静态 + 动态特征（对应论文静态/动态特征）
        self.data_buffer["time"].append(d["timestamp"])
        self.data_buffer["gray"].append(d["gray_mean"])
        self.data_buffer["contrast"].append(d["contrast"])
        self.data_buffer["sat"].append(d["saturation"])
        self.data_buffer["v_crust"].append(d["crust_speed"])
        self.data_buffer["v_node"].append(d["node_speed"])
        self.data_buffer["p_c"].append(d["cls_probs"][0])
        self.data_buffer["p_h"].append(d["cls_probs"][1])
        self.data_buffer["p_n"].append(d["cls_probs"][2])

        # 右侧数字
        self.lbl_gray.setText(f"{d['gray_mean']:.1f}")
        self.lbl_cont.setText(f"{d['contrast']:.1f}")
        self.lbl_sat.setText(f"{d['saturation']:.3f}")
        self.lbl_vc.setText(f"{d['crust_speed']:.2f}")
        self.lbl_vn.setText(f"{d['node_speed']:.2f}")

        # 分类结果 + 趋势
        res = d["cls_result"]
        trend = d.get("trend", "")
        cn_text = CLASS_CN.get(res, res)

        if res == "H":
            bg, fg, icon = "#ffebee", "#c62828", "🔥"
        elif res == "C":
            bg, fg, icon = "#e3f2fd", "#1565c0", "🥶"
        elif res == "N":
            bg, fg, icon = "#e8f5e9", "#2e7d32", "🟢"
        else:
            bg, fg, icon = "#f5f5f5", "#9e9e9e", "🔭"

        full_text = f"{icon} {cn_text}\n(置信度: {d['cls_conf']:.2f})"
        if trend:
            full_text += f"\n⚠️ {trend}"

        self.lbl_res.setText(full_text)
        self.lbl_res.setStyleSheet(
            f"background-color: {bg}; color: {fg}; font-weight: bold; "
            "font-size: 16px; border-radius: 5px; padding: 10px;"
        )

        # 报警标志
        # 【警报逻辑：根据当前状态动态显示“持续过热/持续过冷”】
        if d.get("alarm", False) and res in ("H", "C"):
            if res == "H":
                alert_text = "⚠️ 严重过热警告 ⚠️\n持续过热 > 10 s"
            else:  # res == "C"
                alert_text = "⚠️ 严重过冷警告 ⚠️\n持续过冷 > 10 s"
            self.lbl_alert.setText(alert_text)
            self.lbl_alert.show()
        else:
            self.lbl_alert.hide()

    # ======= 历史记录读写 + UI 显示 =======
    def load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    self.history_records = json.load(f)
            except Exception:
                self.history_records = []
        else:
            self.history_records = []

    def save_history_record(self, record):
        self.history_records.append(record)
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history_records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存历史失败: {e}")
        self.refresh_last_record_label()

    def refresh_last_record_label(self):
        if not hasattr(self, "lbl_last_record"):
            return
        records = self.history_records if self.history_records else FIXED_HISTORY
        if not records:
            self.lbl_last_record.setText("最近一次识别记录：无")
        else:
            r = records[-1]
            cell = r.get("cell") or self.source_to_cell(r.get("source", "未知"))
            self.lbl_last_record.setText(
                f"最近一次识别记录： 时间：{r['time']}， 槽号：{cell}， 过热度状态：{r['result']}"
            )

    def on_finished(self, data):
        # 算法线程结束
        self.stop_algo()

        if data.get("status") == "Finished":
            source = str(data.get("source", "未知"))
            last_code = data.get("last_cls", "NoTarget")
            result_text = CLASS_CN.get(last_code, "未知")

            # 只在有明确分类结果时记录
            if last_code != "NoTarget":
                src = data.get("source", "未知")
                cell_str = self.source_to_cell(src)
                rec = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": str(src),  # 兼容旧字段，保留但不用显示
                    "cell": cell_str,  # 新增：槽号字段
                    "result": result_text,
                }
                self.save_history_record(rec)

        elif data.get("status") == "Error":
            QMessageBox.critical(self, "错误", data.get("msg"))

    def show_history_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("📜 历史识别记录")
        dlg.resize(600, 400)
        v = QVBoxLayout()
        lst = QListWidget()
        lst.setStyleSheet("font-size: 14px;")

        for r in FIXED_HISTORY + list(reversed(self.history_records)):
            res = r.get("result", "未知")
            icon = "🟢" if res == "正常" else ("🔥" if res == "偏高" else "🥶")

            # 新字段优先：cell；旧数据没有就从 source 推断
            cell = r.get("cell")
            if not cell:
                cell = self.source_to_cell(r.get("source", "未知"))

            lst.addItem(f"{icon} [{r.get('time', '--')}] 槽号: {cell} | 结果: {res}")

        v.addWidget(lst)

        # 远程桌面(Qt VNC)下弹窗没有标题栏 / X 按钮，必须自带「返回」按钮，
        # 否则点开历史记录后就回不到检测界面了（Esc 也能关，但不直观）。
        tip = QLabel("提示：也可以按 Esc 键关闭本窗口")
        tip.setStyleSheet("color: #888; font-size: 12px;")
        v.addWidget(tip)

        btn_row = QHBoxLayout()
        btn_clear = QPushButton("🗑 清除所有记录")
        btn_clear.clicked.connect(lambda: self.clear_history(lst))
        btn_close = QPushButton("✅ 返回检测界面")
        btn_close.setMinimumHeight(38)
        btn_close.clicked.connect(dlg.accept)
        btn_close.setDefault(True)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        v.addLayout(btn_row)

        dlg.setLayout(v)
        dlg.exec_()

    def clear_history(self, list_widget):
        # 只清除动态记录；两条固定记录始终保留
        self.history_records = []
        list_widget.clear()
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        for r in FIXED_HISTORY:
            res = r.get("result", "未知")
            icon = "🟢" if res == "正常" else ("🔥" if res == "偏高" else "🥶")
            list_widget.addItem(f"{icon} [{r.get('time', '--')}] 槽号: {r.get('cell', '--')} | 结果: {res}")
        self.refresh_last_record_label()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = FireEyeUI()
    ui.show()
    sys.exit(app.exec_())
