@echo off
rem ====================================================================
rem  Bonsai 8B - OpenAI-compatible SERVER for agents  (double-click)
rem    base_url : http://127.0.0.1:8080/v1     model: bonsai-8b
rem
rem  SELF-LIMITING: llama-plan.ps1 measures your free RAM and refuses to
rem  go above (total RAM - 10% - what is already in use), auto-downgrading
rem  the KV cache / context if needed. Threads are capped at 90% of cores.
rem  Measured peaks (GiB RSS): 64k f16 10.9 | 64k q8_0 6.7 | 64k q4_0 4.0
rem  Stop with Ctrl+C or close this window.
rem ====================================================================
setlocal
chcp 65001 >nul
title Bonsai 8B Agent API - 127.0.0.1:8080

rem -------------------------- knobs --------------------------
set "CTX=131072"        rem as large as the 8 GiB RAM cap allows (q4_0 KV)
set "CTK=q4_0"          rem q4_0 is what makes 128k fit in 8 GiB
set "THREADS=4"         rem HARD cap 4 of 6 cores; also measured fastest
set "MAXRAM=8"          rem HARD cap GiB for llama-server.exe
set "MAXTHREADS=4"      rem HARD cap for llama threads
set "HEADROOM=10"       rem %% RAM that must stay free for Windows + your apps
set "PORT=8080"
set "NPAR=1"            rem parallel slots; >1 splits CTX between them
set "CACHEMB=2048"      rem prompt-cache budget MiB (long system prompt = 1x cost)
set "CTXCP=16"          rem context checkpoints kept for prefix reuse
set "LOWPRI=1"          rem 1 = BelowNormal so the desktop stays smooth
rem -----------------------------------------------------------

cd /d "%~dp0Bonsai-demo"
set "BIN=bin\cpu\llama-server.exe"
rem NOTE: the 8B weights are NOT in models\ yet -- the guard below will stop
rem       with a clear error.  Drop the GGUF in models\ to enable this script.
set "MODEL=%~dp0models\Ternary-Bonsai-8B-PQ2_0.gguf"
set "PLANPS=%~dp0llama-plan.ps1"

if not exist "%BIN%"   ( echo [ERR] not found: %CD%\%BIN%   & pause & exit /b 1 )
if not exist "%MODEL%" ( echo [ERR] not found: %CD%\%MODEL% & pause & exit /b 1 )
if not exist "%PLANPS%" ( echo [ERR] not found: %PLANPS%      & pause & exit /b 1 )

rem ---- ask the planner what we can afford right now ----
set "RUN_CTX=%CTX%" & set "RUN_CTK=%CTK%" & set "RUN_T=%THREADS%"
set "PSPLAN=powershell -NoProfile -ExecutionPolicy Bypass -File %PLANPS% -Ctx %RUN_CTX% -CtK %RUN_CTK% -Threads %RUN_T% -MaxRam %MAXRAM% -MaxThreads %MAXTHREADS% -HeadroomPct %HEADROOM%"
set "PLAN_ERR=0"
for /f %%e in ('%PSPLAN% -Only ERROR') do set "PLAN_ERR=%%e"
if "%PLAN_ERR%"=="1" (
    echo.
    echo  [ERR] Not enough RAM headroom for even the smallest plan.
    for /f %%n in ('%PSPLAN% -Only NOTE') do echo        %%n
    echo        Close some apps - VMware / Edge / QQ - and run again, or lower
    echo        CTX=16336 in this file.
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
for /f %%v in ('%PSPLAN% -Only NOTE')    do set "P_NOTE=%%v"

echo.
echo  ================================================================
echo   Bonsai 8B agent API   (Ternary-Bonsai-8B PQ2_0, CPU only)
echo   endpoint  : http://127.0.0.1:%PORT%/v1     model: bonsai-8b
echo   context   : %CTX%      KV cache: %CTK%      threads: %THREADS%
echo   RAM cap   : ~%P_EST% GiB peak  (limit %P_BUDGET% GiB, %HEADROOM%%% headroom)
echo   RAM now   : %P_USED% GiB used of %P_TOTAL% GiB
echo   CPU cap   : %THREADS% threads, hard cap %MAXTHREADS% of 6 cores, prio %LOWPRI%
echo  ================================================================
if not "%P_NOTE%"=="" echo   [auto] %P_NOTE%
echo.
echo   curl http://127.0.0.1:%PORT%/health
echo   send: {"model":"bonsai-8b","messages":[{"role":"user","content":"hi"}]}
echo.
echo   Ctrl+C to stop.
echo.

set "ARGS=-m %MODEL% --alias bonsai-8b -ngl 0 -fa on -c %CTX% -ctk %CTK% -ctv %CTK% -t %THREADS% -tb %THREADS% -n 0 --host 127.0.0.1 --port %PORT% --parallel %NPAR% --cache-ram %CACHEMB% --ctx-checkpoints %CTXCP% --cache-idle-slots --jinja -rea off --temp 0.5 --top-p 0.85 --top-k 20 --min-p 0"

if "%LOWPRI%"=="1" (
    powershell -NoProfile -Command "$p = Start-Process -FilePath '.\%BIN%' -ArgumentList '%ARGS%' -NoNewWindow -PassThru; $p.PriorityClass = 'BelowNormal'; $p.WaitForExit()"
) else (
    "%BIN%" %ARGS%
)

echo.
echo [done] server stopped.
pause
