@echo off
chcp 65001 >nul
echo ============================================================
echo MARS-Agent 火眼检测系统 - 快速开始指南
echo ============================================================
echo.
echo 📦 项目已整理完成！可以直接分享给他人使用。
echo.
echo 【快速启动】
echo.
echo 1️⃣  启动 PyQt5 桌面界面：
echo    - Windows: 双击 "启动QT界面.bat"
echo    - Linux/Mac: 运行 "bash 启动QT界面.sh"
echo.
echo 2️⃣  启动 Streamlit Web界面：
echo    - Windows: 双击 "启动Web界面.bat"
echo    - Linux/Mac: 运行 "bash 启动Web界面.sh"
echo.
echo 【首次使用前】
echo.
echo ✅ 1. 确保已安装 Python 3.8 或以上版本
echo ✅ 2. 运行: pip install -r requirements.txt
echo ✅ 3. （可选）安装 Ollama 以使用AI分析功能
echo.
echo 【文件说明】
echo.
echo 📄 README.md - 完整的使用说明文档
echo 📄 requirements.txt - Python依赖清单
echo 📄 文件清单.md - 项目文件详细清单
echo 📂 qt_interface/ - PyQt5桌面应用代码
echo 📂 web_interface/ - Streamlit Web应用代码
echo 📂 weights/ - 模型权重文件（YOLO + ResNet）
echo 📂 config/ - 配置文档
echo 📂 assets/ - 静态资源
echo.
echo 【常见问题】
echo.
echo ❓ 如果启动失败，请检查：
echo    1. Python是否正确安装
echo    2. 是否已安装requirements.txt中的依赖
echo    3. 权重文件是否完整（weights/目录）
echo.
echo 💡 更多帮助请查看 README.md
echo.
echo ============================================================
pause
