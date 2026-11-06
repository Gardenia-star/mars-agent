#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MARS-Agent 火眼检测系统 - PyQt5 桌面界面启动脚本
运行此脚本启动桌面应用程序
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from ui_fireeye import FireEyeUI

def main():
    print("=" * 60)
    print("MARS-Agent 火眼检测系统 v1.0")
    print("基于 YOLOv10 + DC-MMGAT 的铝电解槽过热度检测")
    print("=" * 60)
    print("\n正在启动应用程序...")

    app = QApplication(sys.argv)
    window = FireEyeUI()
    window.show()

    print("\n✅ 应用程序启动成功！")
    print("📌 使用说明：")
    print("   1. 点击 '打开视频' 选择视频文件")
    print("   2. 或选择摄像头编号（0, 1, 2...）")
    print("   3. 点击 '开始' 按钮开始检测")
    print("   4. 点击 'AI分析' 获取智能诊断报告（需要Ollama）\n")

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
