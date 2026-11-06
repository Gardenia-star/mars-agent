#!/bin/bash
# ============================================================
# MARS-Agent 远程桌面启动器
#   在浏览器里使用 PyQt5 桌面端（本机无需物理显示器）
#
#   原理：
#     1) Xvfb 起一个虚拟 X11 显示（默认 :100）
#     2) 桌面端跑在这个真实 X11 显示上 —— 渲染走标准 X11 路径，颜色正确
#     3) xpra shadow 把该显示以 RFB(VNC) 协议输出（默认 :5900）
#     4) websockify 把 noVNC 网页（默认 :6080）与 VNC 端口桥接
#     5) 浏览器用 noVNC 打开，即可看到并操作桌面端窗口
#
#   ⚠️ 为什么不用 Qt 自带的 vnc 平台插件：
#     `QT_QPA_PLATFORM=vnc` 的渲染会把**红蓝通道对调**（绿色不变），
#     于是视频里橙红色的火焰会显示成蓝色。用纯色块实测：
#        Qt VNC 插件：左(红)→显示蓝、右(蓝)→显示红   ❌
#        Xvfb 路径  ：左(红)→红、中(绿)→绿、右(蓝)→蓝 ✅
#
#   用法：
#     bash 启动远程桌面.sh
#     桌面端口只监听 127.0.0.1（无密码，避免同网段任何人可操作桌面），
#     所以需要在**你自己的电脑**上先建隧道（SSH 转发只能由客户端发起）：
#       · VS Code：底部「端口 / PORTS」面板 → 转发端口 → 填 6080
#       · Windows：运行 Code\本地隧道.bat（等价于下面的 ssh -L）
#       · 命令行：ssh -N -L 6080:127.0.0.1:6080 use_yu@<本机IP>
#     然后浏览器打开  http://127.0.0.1:6080/vnc.html?autoconnect=1&resize=scale
#     停止：Ctrl+C（或 bash 停止服务.sh）
#
#   可覆盖的环境变量： VNC_PORT / WEB_PORT / SCREEN / DISPLAY_NUM
# ============================================================
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
CONDA_SH="/ExtHDD/Apps/Anaconda/etc/profile.d/conda.sh"
CONDA_ENV="yu"
GUI_ENV="/home/use_yu/.conda/envs/gui"
NOVNC_DIR="/home/use_yu/.local/share/novnc"

VNC_PORT="${VNC_PORT:-5900}"
WEB_PORT="${WEB_PORT:-6080}"
SCREEN="${SCREEN:-1280x800}"
DISPLAY_NUM="${DISPLAY_NUM:-100}"

for p in "$CONDA_SH" "$GUI_ENV/bin/Xvfb" "$GUI_ENV/bin/xpra" \
         "$GUI_ENV/bin/websockify" "$NOVNC_DIR/vnc.html"; do
    if [ ! -e "$p" ]; then
        echo "❌ 缺少组件: $p"
        exit 1
    fi
done

# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "$CONDA_ENV" || {
    echo "❌ 无法激活 conda 环境 $CONDA_ENV"
    exit 1
}

cleanup() {
    echo
    echo "正在停止..."
    [ -n "${APP_PID:-}" ] && kill "$APP_PID" 2>/dev/null
    [ -n "${XPRA_PID:-}" ] && kill "$XPRA_PID" 2>/dev/null
    [ -n "${WS_PID:-}" ] && kill "$WS_PID" 2>/dev/null
    [ -n "${XVFB_PID:-}" ] && kill "$XVFB_PID" 2>/dev/null
    wait 2>/dev/null
    rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}" 2>/dev/null
    echo "已停止。"
}
trap cleanup EXIT INT TERM

# ---------- 工具函数 ----------
port_pids() {
    # 不依赖 lsof/fuser：直接用 python 读 /proc 找监听目标端口的进程
    python3 - "$1" <<'PY' 2>/dev/null
import os, sys, glob
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

is_listening() { (echo >/dev/tcp/127.0.0.1/"$1") >/dev/null 2>&1; }

kill_matching() {
    local pat="$1" label="$2" pids
    pids="$(pgrep -f "$pat" 2>/dev/null | grep -v "^$$\$" || true)"
    if [ -n "$pids" ]; then
        echo "   ⏹  结束$label (PID: $(echo "$pids" | tr '\n' ' '))"
        kill $pids 2>/dev/null || true
    fi
}

# ---------- 0) 预检：清掉上一次的残留实例 ----------
# 上一次异常退出（Ctrl+C 之外被 kill、SSH 断开、终端关闭等）会留下：
#   · Xvfb 仍占着 :$DISPLAY_NUM → 再启动报 "(EE) Server is already active for display N"
#   · xpra/websockify 仍占着 $VNC_PORT/$WEB_PORT → 端口被占用
#   · /tmp/.X<N>-lock 残留 → Xvfb 认为显示已被占用
preflight() {
    echo "▶ 预检：清理上一次的残留实例..."
    kill_matching "xpra shadow :$DISPLAY_NUM" "xpra RFB 服务"
    kill_matching "ui_fireeye.py"             "PyQt5 桌面端"
    kill_matching "websockify.*$WEB_PORT"     "noVNC 网页服务"
    kill_matching "Xvfb :$DISPLAY_NUM"        "Xvfb 虚拟显示"

    local pids port waited=0
    for port in "$VNC_PORT" "$WEB_PORT"; do
        pids="$(port_pids "$port" | sort -u | tr '\n' ' ' || true)"
        if [ -n "${pids// /}" ]; then
            echo "   ⏹  释放端口 $port (PID: $pids)"
            kill $pids 2>/dev/null || true
        fi
    done

    # 等它们真正退出（最多 5 秒）
    while [ "$waited" -lt 10 ]; do
        if ! is_listening "$VNC_PORT" && ! is_listening "$WEB_PORT"; then break; fi
        sleep 0.5; waited=$((waited + 1))
    done

    # 超时还在的，强制结束
    for port in "$VNC_PORT" "$WEB_PORT"; do
        pids="$(port_pids "$port" | sort -u | tr '\n' ' ' || true)"
        if [ -n "${pids// /}" ]; then
            echo "   ⚠️  强制结束占用端口 $port 的进程 (PID: $pids)"
            kill -9 $pids 2>/dev/null || true
        fi
    done

    # 清掉 X 的锁文件与 socket（非正常退出时的残留）
    rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}" 2>/dev/null
    echo "   预检完成。"
}

preflight

# ---------- 1) 虚拟 X11 显示 ----------
echo "▶ 启动 Xvfb 虚拟显示 :$DISPLAY_NUM （$SCREEN）..."
"$GUI_ENV/bin/Xvfb" ":$DISPLAY_NUM" -screen 0 "${SCREEN}x24" -nolisten tcp >/tmp/mars_xvfb.log 2>&1 &
XVFB_PID=$!
sleep 3
if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    echo "❌ Xvfb 启动失败，日志："; tail -5 /tmp/mars_xvfb.log
    exit 1
fi

# ---------- 2) 桌面端（跑在真实 X11 上，颜色正确） ----------
cd "$HERE/qt_interface" || exit 1
echo "▶ 启动 PyQt5 桌面端..."
DISPLAY=":$DISPLAY_NUM" python3 ui_fireeye.py >/tmp/mars_app.log 2>&1 &
APP_PID=$!
sleep 4
if ! kill -0 "$APP_PID" 2>/dev/null; then
    echo "❌ 桌面端启动失败，日志："; tail -10 /tmp/mars_app.log
    exit 1
fi

# ---------- 3) xpra 把该显示输出为 RFB(VNC) ----------
echo "▶ 启动 xpra RFB 输出（端口 $VNC_PORT）..."
"$GUI_ENV/bin/xpra" shadow ":$DISPLAY_NUM" \
    --bind-rfb="127.0.0.1:$VNC_PORT" --rfb-auth=none --daemon=no \
    >/tmp/mars_xpra.log 2>&1 &
XPRA_PID=$!
sleep 5
if ! kill -0 "$XPRA_PID" 2>/dev/null; then
    echo "❌ xpra 启动失败，日志："; tail -10 /tmp/mars_xpra.log
    exit 1
fi

# ---------- 4) noVNC 网页服务 ----------
# 只监听 127.0.0.1：局域网无法直连，必须通过 VS Code 端口转发或 ssh -L 隧道访问
# （该桌面没有密码，暴露到 0.0.0.0 等于同网段任何人都能操作你的桌面）
echo "▶ 启动 noVNC 网页服务（127.0.0.1:$WEB_PORT，仅本机/隧道可访问）..."
"$GUI_ENV/bin/websockify" --web "$NOVNC_DIR" "127.0.0.1:$WEB_PORT" "127.0.0.1:$VNC_PORT" \
    >/tmp/mars_websockify.log 2>&1 &
WS_PID=$!
sleep 2
if ! kill -0 "$WS_PID" 2>/dev/null; then
    echo "❌ noVNC 网页服务启动失败，日志："; tail -10 /tmp/mars_websockify.log
    exit 1
fi

# ---------- 5) 确认两个端口真的在监听 ----------
ok=1
for port in "$VNC_PORT" "$WEB_PORT"; do
    if is_listening "$port"; then
        echo "   ✅ 端口 $port 已就绪"
    else
        echo "   ❌ 端口 $port 未监听"
        ok=0
    fi
done
if [ "$ok" != "1" ]; then
    echo "❌ 启动不完整，日志： /tmp/mars_xpra.log  /tmp/mars_websockify.log"
    exit 1
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

echo
echo "============================================================"
echo "✅ 已启动（桌面端口仅监听 127.0.0.1，局域网无法直连）"
echo
echo "① 先在你自己的电脑上建隧道，任选一种："
echo "   · VS Code：底部「端口 / PORTS」面板 → 转发端口 → 填 $WEB_PORT"
echo "   · Windows 一键脚本：在本地运行  Code\\local_tunnel.bat"
echo "   · 命令行：  ssh -N -L $WEB_PORT:127.0.0.1:$WEB_PORT ${USER}@${IP:-<本机IP>}"
echo
echo "② 然后浏览器打开："
echo "   http://127.0.0.1:$WEB_PORT/vnc.html?autoconnect=1&resize=scale"
echo
echo "   桌面端日志: /tmp/mars_app.log   xpra: /tmp/mars_xpra.log"
echo "   ⚠️ 该桌面无密码保护，请勿把隧道端口暴露到公网。"
echo "   按 Ctrl+C 停止。"
echo "============================================================"
wait
