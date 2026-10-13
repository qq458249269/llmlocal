@echo off
rem ====================================================================
rem  Qwen3.5-4B Q4_K_M - agent API  (CPU only)
rem
rem  Qwen3.5 is a HYBRID linear/full attention model, not dense GQA:
rem  32 layers, full_attention_interval 4 -> only 8 full-attn layers
rem  (4 KV heads x 256 dim) pay KV; the other 24 are linear_attention
rem  with a CONSTANT recurrent state.
rem      8 x 4 x 256 x 2 (K+V) x 2 B = 32 KiB/token f16
rem  => 32k window costs ~1.0 GiB of KV, not the 4.5 GiB a dense 4B
rem     would need. That is why CTK stays f16 here: q4_0 KV would save
rem     0.5 GiB we don't need and doubles prefill cost.
rem  native max_position_embeddings = 262144, so 32k needs no RoPE ext.
rem
rem  Tuned for THIS box (i5-8500 6c / 20 GB RAM / no NVIDIA GPU):
rem    -t 4 / -tb 4  -> matches every other launcher here; 4 measured
rem                      FASTEST on this CPU and keeps 2 of 6 cores free
rem    BelowNormal priority -> Windows stays responsive
rem  EXPECTED (NOT measured yet): ~5-7 t/s generate, ~25-35 t/s prompt.
rem  The 7B Q4_K_M on the same box does 3.9 t/s with 4.36 GiB of weights;
rem  this one carries 2.81 GiB, so 1.3-1.8x is the reasonable band.
rem  Calibration: ./bench_decode.py then paste real numbers in here and
rem  in $PEAK_Q35_4B (llama-plan.ps1). See agent-qwen7b.cmd for the
rem  format and README-qwen.md for the full trap list.
rem
rem  This file is deliberately pure ASCII - cmd.exe mis-decodes UTF-8
rem  .cmd files on codepage 936.
rem ====================================================================
setlocal EnableDelayedExpansion
title Qwen3.5-4B Q4_K_M Agent API - 127.0.0.1:8080

rem -------------------------- knobs --------------------------
set "MODEL=..\models\Qwen_Qwen3.5-4B-Q4_K_M.gguf"
set "ALIAS=qwen3.5-4b"
set "PSIZE=q35_4b"     rem llama-plan.ps1 peak table (derived, see header)
set "CTX=32768"         rem 1.0 GiB KV at f16; planner may raise it to 128k
set "CTK=f16"           rem 32 KiB/tok is cheap enough that q4_0 is a downgrade
set "THREADS=4"
set "TBATCH=4"
set "UBATCH=512"
set "MAXRAM=6"          rem HARD cap GiB; 32k/f16 peak is ~3.8 GiB
set "MAXTHREADS=5"
set "HEADROOM=10"
set "PORT=8080"
set "NPAR=1"
set "CACHEMB=2048"
set "CTXCP=32"
set "CPMINSTEP=256"
set "CPCREUSE=64"
set "LOWPRI=1"
set "SPEC_MTP=0"        rem 1 = --spec-type draft-mtp, self-draft. +20-40%%?
rem                      MEASURE IT before leaving it on: MTP drafts are pure
rem                      extra bandwidth, and this box is bandwidth-bound.
set "TEMP=0.7"          rem no generation_config.json on the model card;
set "TOPP=0.8"          rem these are the Qwen3 non-thinking defaults
set "TOPK=20"
rem -----------------------------------------------------------

cd /d "%~dp0Bonsai-demo"
set "BIN=bin\cpu\llama-server.exe"
set "PLANPS=%~dp0llama-plan.ps1"

if not exist "%BIN%"   ( echo [ERR] not found: %CD%\%BIN%    & pause & exit /b 1 )
if not exist "%MODEL%" ( echo [ERR] not found: %CD%\%MODEL%  & echo        Run: ./dl.sh bartowski/Qwen_Qwen3.5-4B-GGUF Qwen_Qwen3.5-4B-Q4_K_M.gguf models 8  & pause & exit /b 1 )
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
echo   Qwen3.5-4B   Q4_K_M   2.81 GiB   CPU only (-ngl 0)
echo   alias    : %ALIAS%
echo   window   : %CTX% (native max 262144, no RoPE extension needed)
echo   KV cache : %CTK%  32 KiB/tok -> peak ~%P_EST% GiB, cap %P_BUDGET% GiB
echo   threads  : %THREADS% generate / %TBATCH% prompt / ub %UBATCH%
echo   expected : ~5-7 t/s generate, ~25-35 t/s prompt (UNMEASURED)
echo   MTP spec : %SPEC_MTP%
echo   RAM now  : %P_USED% GiB used of %P_TOTAL% GiB
echo  =========================================================
if not "%P_NOTE%"=="" echo   [auto] %P_NOTE%
echo   API: http://127.0.0.1:%PORT%/v1
echo.

set "ARGS=-m %MODEL% --alias %ALIAS% -ngl 0 -fa on -c %CTX% -ctk %CTK% -ctv %CTK% -t %THREADS% -tb %TBATCH% -ub %UBATCH% -n 0 --host 127.0.0.1 --port %PORT% --parallel %NPAR% --cache-ram %CACHEMB% --ctx-checkpoints %CTXCP% --checkpoint-min-step %CPMINSTEP% --cache-reuse %CPCREUSE% --cache-idle-slots --context-shift --jinja --temp %TEMP% --top-p %TOPP% --top-k %TOPK%"
if "%SPEC_MTP%"=="1" set "ARGS=%ARGS% --spec-type draft-mtp --spec-draft-model %MODEL% --spec-draft-n-max 2"

if "%LOWPRI%"=="1" (
    powershell -NoProfile -Command "$p = Start-Process -FilePath '.\%BIN%' -ArgumentList '%ARGS%' -NoNewWindow -PassThru; $p.PriorityClass = 'BelowNormal'; $p.WaitForExit()"
) else (
    "%BIN%" %ARGS%
)

echo.
echo [done] server stopped.
pause