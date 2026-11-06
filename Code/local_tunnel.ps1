# ============================================================
#  MARS-Agent 本地隧道（在【你自己的 Windows 电脑】上运行）
#
#  作用：把远程主机的 noVNC 桌面(6080) 和 Web 端(8501) 转发到本机
#        127.0.0.1。远程服务只监听它自己的 127.0.0.1，所以必须靠隧道。
#
#  用法（任选）：
#    · 双击同目录的  local_tunnel.bat （推荐）
#    · 右键本文件 → 使用 PowerShell 运行
#    · powershell -ExecutionPolicy Bypass -File .\local_tunnel.ps1
#
#  常用参数：
#    -RemoteHost 172.17.0.9     远程主机
#    -RemoteUser use_yu         远程登录用户
#    -LocalDesktopPort 16080    本机映射端口（本机 6080 被占用时改这个）
#    -LocalWebPort 18501        本机映射端口（本机 8501 被占用时改这个）
#    -SkipWeb                   只转发桌面，不转发 Web 端
#    -NoBrowser                 不自动打开浏览器
#
#  按 Ctrl+C 结束隧道。
# ============================================================
[CmdletBinding()]
param(
    [string]$RemoteHost       = "172.17.0.9",
    [string]$RemoteUser       = "use_yu",
    [int]   $DesktopPort      = 6080,
    [int]   $WebPort          = 8501,
    [int]   $LocalDesktopPort = 6080,
    [int]   $LocalWebPort     = 8501,
    [switch]$SkipWeb,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

function Write-Info($msg) { Write-Host $msg -ForegroundColor Cyan }
function Write-Err ($msg) { Write-Host $msg -ForegroundColor Red }

function Test-LocalPortFree([int]$Port) {
    try {
        $l = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $Port)
        $l.Start(); $l.Stop()
        return $true
    } catch {
        return $false
    }
}

if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Err "找不到 ssh 命令。请先安装 OpenSSH 客户端："
    Write-Err "  设置 → 应用 → 可选功能 → 添加功能 → OpenSSH 客户端"
    Write-Err "或（管理员 PowerShell）："
    Write-Err "  Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0"
    exit 1
}

if (-not (Test-LocalPortFree $LocalDesktopPort)) {
    Write-Err "本机端口 $LocalDesktopPort 已被占用，换个端口重试，例如："
    Write-Err "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -LocalDesktopPort 16080"
    exit 1
}
if (-not $SkipWeb -and -not (Test-LocalPortFree $LocalWebPort)) {
    Write-Err "本机端口 $LocalWebPort 已被占用："
    Write-Err "  换端口  -LocalWebPort 18501   或只转发桌面  -SkipWeb"
    exit 1
}

$sshArgs = @(
    "-N",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-L", "${LocalDesktopPort}:127.0.0.1:${DesktopPort}"
)
if (-not $SkipWeb) {
    $sshArgs += @("-L", "${LocalWebPort}:127.0.0.1:${WebPort}")
}
$sshArgs += "$RemoteUser@$RemoteHost"

$desktopUrl = "http://127.0.0.1:${LocalDesktopPort}/vnc.html?autoconnect=1&resize=scale"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Info "MARS-Agent 本地隧道"
Write-Host "  远程主机  : $RemoteUser@$RemoteHost"
Write-Info "  远程桌面  : 127.0.0.1:$LocalDesktopPort  ->  远程 127.0.0.1:$DesktopPort"
if (-not $SkipWeb) {
    Write-Info "  Web 端    : 127.0.0.1:$LocalWebPort  ->  远程 127.0.0.1:$WebPort"
}
Write-Info "  桌面地址  : $desktopUrl"
if (-not $SkipWeb) {
    Write-Info "  Web  地址 : http://127.0.0.1:$LocalWebPort/"
}
Write-Host "  结束隧道  : 在本窗口按 Ctrl+C" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Info "正在建立隧道（若提示输入密码，请输入远程 $RemoteUser 的密码）..."
Write-Host ""

$browserJob = $null
if (-not $NoBrowser) {
    $browserJob = Start-Job -ScriptBlock {
        param($url)
        Start-Sleep -Seconds 4
        Start-Process $url
    } -ArgumentList $desktopUrl
}

$code = 0
try {
    & ssh @sshArgs
    $code = $LASTEXITCODE
} finally {
    if ($browserJob) { Remove-Job $browserJob -Force -ErrorAction SilentlyContinue }
}

Write-Host ""
if ($code -eq 0) {
    Write-Info "隧道已关闭。"
} else {
    Write-Err "ssh 退出（代码 $code）。常见原因："
    Write-Err "  · 远程地址/用户名不对（当前 $RemoteUser@$RemoteHost）"
    Write-Err "  · 认证失败（密码或密钥）"
    Write-Err "  · 远程服务没启动：先在远程执行  bash 启动远程桌面.sh"
    Write-Err "  · 远程端口没在监听：$DesktopPort"
}
if ($Host.Name -eq "ConsoleHost") {
    Write-Host "按回车关闭..." -ForegroundColor DarkGray
    [void](Read-Host)
}
