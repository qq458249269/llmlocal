@echo off
rem ====================================================================
rem  Bonsai 1.7B - CPU chat launcher        (double-click to run)
rem  Ternary-Bonsai-1.7B-PQ2_0.gguf 0.43 GiB + llama.cpp b10743
rem
rem  Tuned for THIS box (i5-8500 6c / 20 GB RAM / no NVIDIA GPU):
rem    -c 32768 + q4_0 KV   -> native window, 1.4 GiB RSS, 34 t/s gen
rem                            (a bigger -c is NOT free: 64k 21 t/s,
rem                             128k 18 t/s, and +3 GiB RAM)
rem                            (f16 KV measured 2x slower prefill,
rem                             5x slower generate, 10.5 GiB RSS)
rem    -t 4 / -tb 4         -> measured FASTEST (35.9 t/s gen,
rem                             47.9 t/s prompt), 2 cores stay free
rem    -ub 512              -> measured faster than -ub 2048 here
rem    BelowNormal priority -> Windows stays responsive
rem
rem  CONTEXT: trained window is 32768, so 128k needs YaRN RoPE extension.
rem    ROPE=1 -> 128k (verified sane output)
rem    ROPE=0 -> native 32768 (best quality, no extrapolation)
rem
rem  Faster but 1-bit (50 t/s, lower quality)?
rem    set MODEL=..\models\Bonsai-1.7B-Q1_0.gguf
rem
rem  In-chat commands: /exit  /clear  /regen  /read ^<file^>
rem  NOTE: this file is deliberately pure ASCII - see agent-bonsai17b.cmd
rem        for why (cmd.exe mis-decodes UTF-8 .cmd files on codepage 936).
rem        Docs: README-bonsai.md
rem ====================================================================
setlocal EnableDelayedExpansion
chcp 65001 >nul
title Bonsai 1.7B - CPU Chat

rem -------------------------- knobs --------------------------
set "CTX=32768"         rem native window: fastest (34 t/s gen), best recall, 1.4 GiB. 65536 -> 21 t/s, 131072 -> 18 t/s. Raise only if you feed 10k+ / 30k+ tokens per turn.
set "CTK=q4_0"          rem q8_0 = better long-context recall (~7.9 GiB)
set "THREADS=4"         rem measured fastest; 2 of 6 cores stay free
set "TBATCH=4"
set "UBATCH=512"        rem measured faster than 2048 on this CPU
set "MAXRAM=10"         rem HARD cap GiB for llama-cli.exe
set "MAXTHREADS=5"      rem HARD cap for llama threads
set "HEADROOM=10"       rem pct RAM that must stay free for Windows + apps
set "LOWPRI=1"          rem 1 = BelowNormal so the desktop stays smooth
set "MAXTOK=-1"         rem -1 = unlimited, or e.g. 512
set "MODEL=..\models\Ternary-Bonsai-1.7B-PQ2_0.gguf"
set "ROPE=1"            rem 1 = YaRN-extend CTX past the training window
set "ORIGCTX=32768"     rem the model's real training window
set "ARCH=qwen3"        rem GGUF arch prefix used by --override-kv
rem -----------------------------------------------------------

cd /d "%~dp0Bonsai-demo"
set "BIN=bin\cpu\llama-cli.exe"
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
set "PSPLAN=powershell -NoProfile -ExecutionPolicy Bypass -File %PLANPS% -Size 17b -Ctx %RUN_CTX% -CtK %RUN_CTK% -Threads %RUN_T% -MaxRam %MAXRAM% -MaxThreads %MAXTHREADS% -HeadroomPct %HEADROOM%"
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
set "CTV=%CTK%"
set "P_EST=" & set "P_BUDGET=" & set "P_USED=" & set "P_TOTAL=" & set "P_NOTE="
for /f %%v in ('%PSPLAN% -Only EST')     do set "P_EST=%%v"
for /f %%v in ('%PSPLAN% -Only BUDGET')  do set "P_BUDGET=%%v"
for /f %%v in ('%PSPLAN% -Only USERAM')  do set "P_USED=%%v"
for /f %%v in ('%PSPLAN% -Only TOTRAM')  do set "P_TOTAL=%%v"
for /f "tokens=*" %%v in ('%PSPLAN% -Only NOTE')  do set "P_NOTE=%%v"

rem ---- YaRN context extension (needed when CTX > the model's training window) ----
set "ROPEARGS="
set "SCALE=1"
if "%ROPE%"=="1" (
    set /a SCALE=%CTX%/%ORIGCTX%
    if !SCALE! GTR 1 set "ROPEARGS=--override-kv %ARCH%.context_length=int:%CTX% --rope-scaling yarn --rope-scale !SCALE! --yarn-orig-ctx %ORIGCTX%"
)

echo.
echo  =========================================================
echo   Ternary-Bonsai-1.7B   PQ2_0 (2-bit)   0.43 GiB   CPU only (-ngl 0)
if "!SCALE!" GTR 1 echo   window   : %CTX% via YaRN RoPE extension (native %ORIGCTX%, scale !SCALE!)
if "!SCALE!" LEQ 1 echo   window   : %CTX% native, no RoPE extension (best recall)
echo   KV cache  : %CTK%   (peak ~%P_EST% GiB, cap %P_BUDGET% GiB)
echo   threads   : %THREADS% generate / %TBATCH% prompt / ub %UBATCH%  (cap %MAXTHREADS% of 6 cores)
echo   expected  : about 34 t/s generate, 100 t/s prompt at this window (measured)
echo   RAM now   : %P_USED% GiB used of %P_TOTAL% GiB - %HEADROOM%% stays free
echo   priority  : %LOWPRI%   (1 = BelowNormal, keeps the PC usable)
echo  =========================================================
if not "%P_NOTE%"=="" echo   [auto] %P_NOTE%
echo   in-chat:  /exit  /clear  /regen  /read ^<file^>
echo   e.g.     /read notes.txt
echo.

set "ARGS=-m %MODEL% -ngl 0 -fa on -c %CTX% -ctk %CTK% -ctv %CTV% -t %THREADS% -tb %TBATCH% -ub %UBATCH% -rea off --temp 0.5 --top-p 0.85 --top-k 20 -n %MAXTOK%"

if "%LOWPRI%"=="1" (
    powershell -NoProfile -Command "$p = Start-Process -FilePath '.\%BIN%' -ArgumentList '%ARGS% %ROPEARGS%' -NoNewWindow -PassThru; $p.PriorityClass = 'BelowNormal'; $p.WaitForExit()"
) else (
    "%BIN%" %ARGS% %ROPEARGS%
)

echo.
echo [done] session ended.
pause
