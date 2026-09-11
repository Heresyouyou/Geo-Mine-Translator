@echo off
chcp 65001 >nul
title PostNews Translate — llama-server + Flask
cd /d D:\PostNews\Translate

echo ╔══════════════════════════════════════════════════════════╗
echo ║  PostNews Translate — 一键启动                           ║
echo ║  后端: llama-server (Qwen3.5 GGUF, CUDA GPU)             ║
echo ║  前端: Flask web_app.py                                  ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM ── 1. 找模型 ──
set MODEL_DIR=D:\PostNews\Translate\models-gguf
set LLAMA=D:\PostNews\Translate\llama-cpp\llama-server.exe

set MODEL=
if exist "%MODEL_DIR%\Qwen3.5-4B-Q4_K_M.gguf" (
    set MODEL=%MODEL_DIR%\Qwen3.5-4B-Q4_K_M.gguf
    goto found
)
if exist "%MODEL_DIR%\Qwen3.5-4B-Q5_K_M.gguf" (
    set MODEL=%MODEL_DIR%\Qwen3.5-4B-Q5_K_M.gguf
    goto found
)
if exist "%MODEL_DIR%\Qwen3.5-0.8B-Q4_K_M.gguf" (
    set MODEL=%MODEL_DIR%\Qwen3.5-0.8B-Q4_K_M.gguf
    goto found
)

echo ❌ 没找到 GGUF 模型! 请下载到 %MODEL_DIR%
echo    推荐: Qwen3.5-4B-Q4_K_M.gguf (2.55GB)
echo.
pause
exit /b 1

:found
echo ✅ 模型: %MODEL%

REM ── 2. 检查端口 ──
set PORT=8090
netstat -ano | findstr ":%PORT%" | findstr "LISTENING" >nul
if %errorlevel%==0 (
    echo ⚠️  端口 %PORT% 已被占用, 假设 llama-server 已启动
    goto web
)

REM ── 3. 启动 llama-server ──
echo.
echo [1/2] 启动 llama-server...
start "llama-server" /min "%LLAMA%" -m "%MODEL%" -ngl -1 -c 4096 -t 8 -rea off --port %PORT%

REM 等 server 起来
echo 等待 server 就绪...
setlocal enabledelayedexpansion
for /L %%i in (1,1,30) do (
    timeout /t 1 /nobreak >nul
    curl -s http://127.0.0.1:%PORT%/health >nul 2>&1
    if !errorlevel!==0 (
        echo ✅ llama-server 已就绪
        goto web
    )
)
echo ⚠️  server 启动超时, 继续尝试...

:web
REM ── 4. 启动 Flask ──
echo.
echo [2/2] 启动 Flask web_app.py...
start "Flask-WebApp" /min py -3.12 D:\PostNews\Translate\web_app.py

echo.
echo ✅ 启动完成!
echo    llama-server: http://127.0.0.1:%PORT%
echo    Flask API:    http://127.0.0.1:5000
echo.
echo 提示: 关掉此窗口不会停止后台进程
echo        要完全退出, 关掉 llama-server 和 Flask 两个窗口即可
pause
