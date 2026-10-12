@echo off
rem ====================================================================
rem  Qwen2.5-Coder-7B-Instruct Q4_K_M - agent API  (CPU only)
rem  Measured on THIS box (i5-8500 6c / 20 GB RAM / no NVIDIA GPU):
rem    -ngl 0 -t 4  -> 3.9 t/s generate, 26 t/s prompt (llama-bench b10743)
rem    -c 32768 -ctk q8_0 -> native 32k window, KV 0.88 GiB
rem    total RSS ~6 GiB, so MAXRAM=9 keeps Windows happy
rem
rem  This file is deliberately pure ASCII - cmd.exe mis-decodes UTF-8
rem  .cmd files on codepage 936. See agent-lfm25.cmd for the full
rem  list of traps (no "|" on a set line, etc).
rem
rem  Docs: README-qwen.md
rem ====================================================================
setlocal EnableDelayedExpansion
title Qwen2.5-Coder-7B Agent API - 127.0.0.1:8080

rem -------------------------- knobs --------------------------
set "MODEL=..\models\qwen2.5-coder-7b-instruct-q4_k_m.gguf"
set "ALIAS=qwen2.5-coder-7b"
set "PSIZE=q7b"         rem llama-plan.ps1 peak table
set "CTX=32768"         rem native window for Qwen2.5-Coder (32768 trained)
set "CTK=q8_0"          rem KV 56 KiB/tok -> 32k = 0.88 GiB at q8_0
set "THREADS=4"         rem measured fastest; 2 of 6 cores stay free
set "TBATCH=4"
set "UBATCH=512"
set "MAXRAM=9"          rem HARD cap GiB for llama-server.exe
set "MAXTHREADS=5"
set "HEADROOM=10"
set "PORT=8080"
set "NPAR=1"
set "CACHEMB=2048"
set "CTXCP=32"
set "CPMINSTEP=256"
set "CPCREUSE=64"
set "LOWPRI=1"
set "TEMP=0.1"          rem model card: temp 0.1 / top_p 0.95 / top_k 20
set "TOPP=0.95"
set "TOPK=20"
rem -----------------------------------------------------------

cd /d "%~dp0Bonsai-demo"
set "BIN=bin\cpu\llama-server.exe"
set "PLANPS=%~dp0llama-plan.ps1"

if not exist "%BIN%"   ( echo [ERR] not found: %CD%\%BIN%    & pause & exit /b 1 )
if not exist "%MODEL%" ( echo [ERR] not found: %CD%\%MODEL%  & pause & exit /b 1 )
if not exist "%PLANPS%" ( echo [ERR] not found: %PLANPS%       & pause & exit /b 1 )

for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    echo [SKIP] Port %PORT% is already in use ^(PID %%P^).
    echo        Server is probably already running - use
    echo        http://127.0.0.1:%PORT%/
    echo        Restart: Ctrl+C, or
    echo        powershell -c "Stop-Process -Name llama-server -Force"
    echo.
    pause
    exit /b 0
)

set "RUN_CTX=%CTX%" & set "RUN_CTK=%CTK%" & set "RUN_T=%THREADS%"
set "PSPLAN=powershell -NoProfile -ExecutionPolicy Bypass -File %PLANPS% -Size %PSIZE% -Ctx %RUN_CTX% -CtK %RUN_CTK% -Threads %RUN_T% -MaxRam %MAXRAM% -MaxThreads %MAXTHREADS% -HeadroomPct %HEADROOM%"
set "PLAN_ERR=0"
for /f %%e in ('%PSPLAN% -Only ERROR') do set "PLAN_ERR=%%e"
if "%PLAN_ERR%"=="1" (
    echo [ERR] Not enough RAM headroom for even the smallest plan.
    for /f "tokens=*" %%n in ('%PSPLAN% -Only NOTE') do echo        %%n
    echo        Close some apps and run again, or lower CTX in this file.
    pause & exit /b 1
)
for /f %%v in ('%PSPLAN% -Only CTX')     do set "CTX=%%v"
for /f %%v in ('%PSPLAN% -Only CTK')     do set "CTK=%%v"
for /f %%v in ('%PSPLAN% -Only THREADS') do set "THREADS=%%v"
set "P_EST=" & set "P_BUDGET=" & set "P_USED=" & set "P_TOTAL=" & set "P_NOTE="
for /f %%v in ('%PSPLAN% -Only EST')     do set "P_EST=%%v"
for /f %%v in ('%PSPLAN% -Only BUDGET')  do set "P_BUDGET=%%v"
for /f %%v in ('%PSPLAN% -Only USERAM')  do set "P_USED=%%v"
for /f %%v in ('%PSPLAN% -Only TOTRAM')  do set "P_TOTAL=%%v"
for /f "tokens=*" %%v in ('%PSPLAN% -Only NOTE') do set "P_NOTE=%%v"

echo.
echo  =========================================================
echo   Qwen2.5-Coder-7B-Instruct  Q4_K_M  4.36 GiB   CPU only (-ngl 0)
echo   alias    : %ALIAS%
echo   window   : %CTX% (native, no RoPE extension needed)
echo   KV cache : %CTK%  (peak ~%P_EST% GiB, cap %P_BUDGET% GiB)
echo   threads  : %THREADS% generate / %TBATCH% prompt / ub %UBATCH%
echo   measured : ~3.9 t/s generate, ~26 t/s prompt (llama-bench)
echo   RAM now  : %P_USED% GiB used of %P_TOTAL% GiB
echo  =========================================================
if not "%P_NOTE%"=="" echo   [auto] %P_NOTE%
echo   API: http://127.0.0.1:%PORT%/v1
echo.

set "ARGS=-m %MODEL% --alias %ALIAS% -ngl 0 -fa on -c %CTX% -ctk %CTK% -ctv %CTK% -t %THREADS% -tb %TBATCH% -ub %UBATCH% -n 0 --host 127.0.0.1 --port %PORT% --parallel %NPAR% --cache-ram %CACHEMB% --ctx-checkpoints %CTXCP% --checkpoint-min-step %CPMINSTEP% --cache-reuse %CPCREUSE% --cache-idle-slots --context-shift --jinja --temp %TEMP% --top-p %TOPP% --top-k %TOPK%"

if "%LOWPRI%"=="1" (
    powershell -NoProfile -Command "$p = Start-Process -FilePath '.\%BIN%' -ArgumentList '%ARGS%' -NoNewWindow -PassThru; $p.PriorityClass = 'BelowNormal'; $p.WaitForExit()"
) else (
    "%BIN%" %ARGS%
)

echo.
echo [done] server stopped.
pause
