#!/bin/bash
# MARS-Agent 火眼检测系统 - Web启动脚本（Linux/Mac）

echo "============================================================"
echo "MARS-Agent 火眼检测系统 - Web版"
echo "============================================================"
echo ""

cd "$(dirname "$0")/web_interface"

echo "正在检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3！请先安装Python 3.8+"
    exit 1
fi

echo "✅ Python环境检测通过"
echo ""
echo "正在启动Web服务..."
echo "📌 浏览器将自动打开 http://localhost:8501"
echo "📌 按 Ctrl+C 停止服务"
echo ""

python3 run_web.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 启动失败！请检查："
    echo "   1. 是否安装了requirements.txt中的依赖"
    echo "   2. 权重文件是否存在于 weights/ 目录"
    echo "   3. 端口8501是否被占用"
fi
