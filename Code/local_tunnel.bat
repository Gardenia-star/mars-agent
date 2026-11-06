@echo off
rem ============================================================
rem  MARS-Agent local tunnel (double-click to run)
rem  Forwards remote noVNC desktop 6080 and Web UI 8501 to localhost.
rem  If a port is busy:  local_tunnel.bat -LocalDesktopPort 16080
rem ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0local_tunnel.ps1" %*
