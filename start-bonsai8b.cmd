@echo off
rem ====================================================================
rem  Bonsai 8B - CPU chat launcher        (double-click to run)
rem  Ternary-Bonsai-8B-PQ2_0.gguf  +  PrismML llama.cpp fork  b10743
rem
rem  Tuned for THIS box (i5-8500 6c / 20 GB RAM / no NVIDIA GPU):
rem    -c 131072 + Q4_0 KV   -> 128k context in ~7.0 GiB RSS
rem                            (F16 KV at 128k would need 13.6 GiB)
rem    -t 4 / -tb 4         -> measured FASTEST (8.0 t/s vs 5.9 at -t 6)
rem                            and it leaves 2 of 6 cores free
rem    BelowNormal priority -> Windows keeps UI/apps responsive
rem  Type /exit or Ctrl+C to quit.
rem ====================================================================
setlocal
chcp 65001 >nul
title Bonsai 8B - CPU Chat - 131072 ctx

rem -------------------------- knobs --------------------------
set "CTX=131072"        rem as large as the 8 GiB RAM cap allows
set "CTK=q4_0"          rem q4_0 is what makes 128k fit in 8 GiB
set "CTV=q4_0"
set "THREADS=4"         rem HARD cap 4 of 6 cores; also measured fastest
set "TBATCH=4"
set "MAXRAM=8"          rem HARD cap GiB for llama-cli.exe
set "MAXTHREADS=4"      rem HARD cap for llama threads
set "HEADROOM=10"       rem %% RAM that must stay free for Windows + your apps
set "LOWPRI=1"          rem 1 = BelowNormal so the desktop stays smooth
set "MAXTOK=-1"         rem -1 = unlimited, or e.g. 512
rem -----------------------------------------------------------

cd /d "%~dp0Bonsai-demo"
set "BIN=bin\cpu\llama-cli.exe"
set "MODEL=%~dp0models\Ternary-Bonsai-8B-PQ2_0.gguf"
set "PLANPS=%~dp0llama-plan.ps1"

if not exist "%BIN%" (
    echo [ERR] llama-cli.exe not found:  %CD%\%BIN%
    echo       Run  install_cn.ps1  once to download it.
    pause & exit /b 1
)
if not exist "%MODEL%" (
    echo [ERR] model not found:  %CD%\%MODEL%
    pause & exit /b 1
)

set "RUN_CTX=%CTX%" & set "RUN_CTK=%CTK%" & set "RUN_T=%THREADS%"
set "PSPLAN=powershell -NoProfile -ExecutionPolicy Bypass -File %PLANPS% -Ctx %RUN_CTX% -CtK %RUN_CTK% -Threads %RUN_T% -MaxRam %MAXRAM% -MaxThreads %MAXTHREADS% -HeadroomPct %HEADROOM%"
set "PLAN_ERR=0"
for /f %%e in ('%PSPLAN% -Only ERROR') do set "PLAN_ERR=%%e"
if "%PLAN_ERR%"=="1" (
    echo [ERR] Not enough RAM headroom for even the smallest plan.
    for /f %%n in ('%PSPLAN% -Only NOTE') do echo        %%n
    echo        Close some apps and run again, or lower CTX in this file.
    pause & exit /b 1
)
for /f %%v in ('%PSPLAN% -Only CTX')     do set "CTX=%%v"
for /f %%v in ('%PSPLAN% -Only CTK')     do set "CTK=%%v"
for /f %%v in ('%PSPLAN% -Only THREADS') do set "THREADS=%%v"
set "CTV=%CTK%"
set "P_EST=" & set "P_BUDGET=" & set "P_USED=" & set "P_TOTAL=" & set "P_NOTE="
for /f %%v in ('%PSPLAN% -Only EST')     do set "P_EST=%%v"
for /f %%v in ('%PSPLAN% -Only BUDGET')  do set "P_BUDGET=%%v"
for /f %%v in ('%PSPLAN% -Only USERAM')  do set "P_USED=%%v"
for /f %%v in ('%PSPLAN% -Only TOTRAM')  do set "P_TOTAL=%%v"
for /f %%v in ('%PSPLAN% -Only NOTE')    do set "P_NOTE=%%v"

echo.
echo  =========================================================
echo   Ternary-Bonsai-8B   PQ2_0   2.03 GiB   CPU only (-ngl 0)
echo   context   : %CTX%   KV cache: %CTK%   (peak ~%P_EST% GiB, cap %P_BUDGET% GiB)
echo   threads   : %THREADS% generate / %TBATCH% prompt  (cap %MAXTHREADS% of 6 cores)
echo   RAM now   : %P_USED% GiB used of %P_TOTAL% GiB - %HEADROOM%%% stays free
echo   priority  : %LOWPRI%   (1 = BelowNormal, keeps the PC usable)
echo  =========================================================
if not "%P_NOTE%"=="" echo   [auto] %P_NOTE%
echo   in-chat:  /exit  /clear  /regen  /read ^<file^>
echo   e.g.     /read notes.txt
echo.

set "ARGS=-m %MODEL% -ngl 0 -fa on -c %CTX% -ctk %CTK% -ctv %CTV% -t %THREADS% -tb %TBATCH% -rea off --temp 0.5 --top-p 0.85 --top-k 20 --min-p 0 -n %MAXTOK%"

if "%LOWPRI%"=="1" (
    powershell -NoProfile -Command "$p = Start-Process -FilePath '.\%BIN%' -ArgumentList '%ARGS%' -NoNewWindow -PassThru; $p.PriorityClass = 'BelowNormal'; $p.WaitForExit()"
) else (
    "%BIN%" %ARGS%
)

echo.
echo [done] session ended.
pause
