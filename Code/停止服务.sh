#!/bin/bash
# ============================================================
# MARS-Agent 停止脚本 —— 停掉本项目所有运行中的进程
#
#   用法:  bash 停止服务.sh
#   可选:  DISPLAY_NUM=100 bash 停止服务.sh   # 指定虚拟显示号（默认 100）
#
#   会依次处理：
#     1) Streamlit Web 服务（含 run_web.py 启动器）
#     2) noVNC 桥接 websockify
#     3) xpra RFB 输出 / Xvfb 虚拟显示（远程桌面栈）
#     4) PyQt5 桌面端（含 run_qt.py 启动器）
#     5) 残留的占用 8501 / 5900 / 6080 端口的进程
#     6) X 的锁文件 /tmp/.X<N>-lock（残留会导致下次报 "Server is already active"）
#
#   注意：算法是跑在 UI 进程里的线程（QThread），不是独立后端进程，
#        所以杀掉 UI 进程即可，不存在"只杀后端"的说法。
# ============================================================
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
PORTS="8501 5900 6080"
DISPLAY_NUM="${DISPLAY_NUM:-100}"

stop_pattern() {
    local pat="$1" label="$2" pids
    pids="$(pgrep -f "$pat" 2>/dev/null | grep -v "^$$\$" || true)"
    if [ -n "$pids" ]; then
        echo "  ⏹  $label  (PID: $(echo "$pids" | tr '\n' ' '))"
        kill $pids 2>/dev/null || true
    else
        echo "  ·  $label  未在运行"
    fi
}

port_pids() {
    # 不依赖 lsof/fuser：直接用 python 读 /proc 找占用端口的进程
    python3 - "$1" <<'PY' 2>/dev/null
import os, sys, glob, socket, struct
target = int(sys.argv[1])
inodes = set()
for line in open("/proc/net/tcp"):
    f = line.split()
    if len(f) < 10 or f[3] != "0A":          # 0A = LISTEN
        continue
    if int(f[1].split(":")[1], 16) == target:
        inodes.add(f[9])
if not inodes:
    raise SystemExit
for fd in glob.glob("/proc/[0-9]*/fd/*"):
    try:
        link = os.readlink(fd)
    except OSError:
        continue
    if link.startswith("socket:[") and link[8:-1] in inodes:
        print(fd.split("/")[2])
PY
}

echo "============================================================"
echo "MARS-Agent 停止服务"
echo "============================================================"

stop_pattern "$HERE/web_interface/streamlit_app.py" "Streamlit Web 服务"
stop_pattern "run_web.py"                          "Web 启动器 (run_web.py)"
stop_pattern "websockify"                          "noVNC 桥接 (websockify)"
stop_pattern "xpra shadow :$DISPLAY_NUM"           "xpra RFB 输出"
stop_pattern "ui_fireeye.py"                       "PyQt5 桌面端"
stop_pattern "run_qt.py"                           "桌面端启动器 (run_qt.py)"

sleep 2

# 还没退出的，强制结束
for pat in "$HERE/web_interface/streamlit_app.py" "websockify" "ui_fireeye.py" \
           "xpra shadow :$DISPLAY_NUM"; do
    pids="$(pgrep -f "$pat" 2>/dev/null || true)"
    if [ -n "$pids" ]; then
        echo "  ⚠️  强制结束残留进程: $(echo "$pids" | tr '\n' ' ')"
        kill -9 $pids 2>/dev/null || true
    fi
done

# Xvfb 放在端口清理之后（它不监听这些端口），最后单独处理
stop_pattern "Xvfb :$DISPLAY_NUM" "Xvfb 虚拟显示"
pids="$(pgrep -f "Xvfb :$DISPLAY_NUM" 2>/dev/null || true)"
if [ -n "$pids" ]; then
    echo "  ⚠️  强制结束残留进程: $(echo "$pids" | tr '\n' ' ')"
    kill -9 $pids 2>/dev/null || true
fi

# 按端口兜底清理
for p in $PORTS; do
    pids="$(port_pids "$p" | sort -u | tr '\n' ' ' || true)"
    if [ -n "${pids// /}" ]; then
        echo "  ⚠️  端口 $p 仍被占用，强制释放 (PID: $pids)"
        kill -9 $pids 2>/dev/null || true
    fi
done

# 清掉 X 的锁文件与 socket（非正常退出时的残留会让下次启动报
# "(EE) Server is already active for display N"）
if [ -e "/tmp/.X${DISPLAY_NUM}-lock" ] || [ -e "/tmp/.X11-unix/X${DISPLAY_NUM}" ]; then
    echo "  ⏹  清理 X 锁文件 /tmp/.X${DISPLAY_NUM}-lock"
fi
rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}" 2>/dev/null

sleep 1
echo
echo "--- 端口状态 ---"
python3 - <<'PY'
import socket
for port, name in [(8501, "Streamlit Web"), (5900, "xpra RFB"), (6080, "noVNC")]:
    s = socket.socket(); s.settimeout(2)
    try:
        s.connect(("127.0.0.1", port)); print(f"  :{port} {name:14} 仍在运行 ⚠️")
    except Exception:
        print(f"  :{port} {name:14} 已停止 ✅")
    finally:
        s.close()
PY
echo
echo "完成。重新启动："
echo "  Web 端    : conda activate yu && bash 启动Web界面.sh"
echo "  远程桌面  : bash 启动远程桌面.sh"
