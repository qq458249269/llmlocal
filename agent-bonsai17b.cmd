@echo off
rem ====================================================================
rem  Bonsai 1.7B - OpenAI-compatible SERVER for agents  (double-click)
rem  Ternary-Bonsai-1.7B-PQ2_0.gguf (2-bit, 0.43 GiB) + llama.cpp b10743
rem
rem  ENDPOINT : http://127.0.0.1:8080/v1      MODEL : bonsai-1.7b
rem  WEB UI   : http://127.0.0.1:8080/       (use 127.0.0.1, not localhost -
rem             the server binds IPv4 only and some browsers resolve
rem             localhost to ::1 first)
rem
rem  KEEP THIS WINDOW OPEN. Closing it stops the server.
rem
rem  SELF-LIMITING: llama-plan.ps1 measures free RAM right now and never goes
rem  above (total - 10%% headroom - in use), downgrading KV type / context if
rem  needed. Threads are hard-capped at 5 of 6 cores.
rem  Measured peaks (GiB RSS): 128k q4_0 4.4 | 128k q8_0 7.9 | 128k f16 10.5
rem  Measured speed: ~36 t/s generate, ~48 t/s prompt (short context)
rem
rem  PREFIX CACHE (biggest CPU win):
rem    2275-token system prompt: 82.6 s cold -> 0.67 s warm (99.2%% reused)
rem    6-turn agent loop (system + 3 tools): TTFT 1.5-2.5 s, reuse 90-94%%
rem    --checkpoint-min-step 256  makes checkpoints possible for short prompts
rem                                (default 8192 is way too big for an agent)
rem    --cache-reuse 64          KV-shifts the prefix when a turn is edited
rem    --context-shift           slides the window instead of erroring out
rem    Client rules (get them for free with bonsai_client.py):
rem      1) system prompt + tools first, byte-identical (no timestamps/IDs)
rem      2) keep the tools array in the same order
rem      3) append-only history, tool results in a stable order
rem      4) keep alias / temp / top_p constant
rem
rem  Faster but 1-bit (50 t/s, lower quality)?
rem      set MODEL=..\models\Bonsai-1.7B-Q1_0.gguf
rem
rem  NOTE: this file is deliberately pure ASCII. cmd.exe decodes a UTF-8 .cmd
rem  with the system codepage (936 here) and mis-pairs Chinese bytes into pipe
rem  characters, which then runs comment tails as commands. Docs: README-bonsai.md
rem ====================================================================
setlocal EnableDelayedExpansion
chcp 65001 >nul
title Bonsai 1.7B Agent API - 127.0.0.1:8080

rem -------------------------- knobs --------------------------
rem 32768 because it is the model's NATIVE training window: no YaRN
rem extrapolation = best recall, and only 1.4 GiB.  It does NOT cost
rem speed.  Measured llama-bench tg64 is flat across the whole range:
rem 512 -> 34.8, 2k -> 34.7, 4k -> 34.9, 8k -> 34.3 t/s, because decode
rem only walks the KV that actually exists, not the whole -c allocation.
rem An older comment here claimed 32k/64k/128k -> 34/21/18 t/s; that
rem conflated the -c cap with real conversation length and does not
rem reproduce on this b10743 build.  Raise it to 65536/131072 freely.
set "CTK=q4_0"          rem q4_0; q8_0 = better long-context recall (7.9 GiB)
set "THREADS=4"         rem measured fastest, leaves 2 of 6 cores free
set "TBATCH=4"
set "UBATCH=512"        rem measured faster than 2048 on this CPU
set "MAXRAM=10"         rem HARD cap GiB for llama-server.exe
set "MAXTHREADS=5"      rem HARD cap for llama threads
set "HEADROOM=10"       rem pct RAM that must stay free for Windows + apps
set "PORT=8080"
set "NPAR=1"            rem parallel slots; >1 splits CTX between them
set "CACHEMB=2048"      rem --cache-ram: MiB budget for cached prefixes
set "CTXCP=32"          rem --ctx-checkpoints per slot (llama.cpp default 32)
set "CPMINSTEP=256"     rem --checkpoint-min-step: spacing between checkpoints
set "CPCREUSE=64"       rem --cache-reuse: min chunk for KV-shifting
set "LOWPRI=1"          rem 1 = BelowNormal so the desktop stays smooth
set "MODEL=..\models\Ternary-Bonsai-1.7B-PQ2_0.gguf"
set "ROPE=1"            rem 1 = YaRN-extend CTX past the training window
set "ORIGCTX=32768"     rem the model's real training window
set "ARCH=qwen3"        rem GGUF arch prefix used by --override-kv
rem -----------------------------------------------------------

cd /d "%~dp0Bonsai-demo"
set "BIN=bin\cpu\llama-server.exe"
set "PLANPS=%~dp0llama-plan.ps1"

if not exist "%BIN%"   ( echo [ERR] not found: %CD%\%BIN%    & pause & exit /b 1 )
if not exist "%MODEL%" ( echo [ERR] not found: %CD%\%MODEL%  & pause & exit /b 1 )
if not exist "%PLANPS%" ( echo [ERR] not found: %PLANPS%       & pause & exit /b 1 )

rem ---- port already taken? do not start a second copy ----
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%PORT% .*LISTENING"') do (
    echo [SKIP] Port %PORT% is already in use ^(PID %%P^).
    echo        The server is probably already running - just use
    echo        http://127.0.0.1:%PORT%/
    echo        To restart it: Ctrl+C in its window, or run
    echo        powershell -c "Stop-Process -Name llama-server -Force"
    echo.
    pause
    exit /b 0
)

rem ---- ask the planner what we can afford right now ----
set "RUN_CTX=%CTX%" & set "RUN_CTK=%CTK%" & set "RUN_T=%THREADS%"
set "PSPLAN=powershell -NoProfile -ExecutionPolicy Bypass -File %PLANPS% -Size 17b -Ctx %RUN_CTX% -CtK %RUN_CTK% -Threads %RUN_T% -MaxRam %MAXRAM% -MaxThreads %MAXTHREADS% -HeadroomPct %HEADROOM%"
set "PLAN_ERR=0"
for /f %%e in ('%PSPLAN% -Only ERROR') do set "PLAN_ERR=%%e"
if "%PLAN_ERR%"=="1" (
    echo.
    echo  [ERR] Not enough RAM headroom for even the smallest plan.
    for /f "tokens=*" %%n in ('%PSPLAN% -Only NOTE') do echo        %%n
    echo        Close some apps - VMware / Edge / QQ - and run again, or lower
    echo        or lower CTX in this file.
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
for /f "tokens=*" %%v in ('%PSPLAN% -Only NOTE')  do set "P_NOTE=%%v"
set "P_NOTE=!P_NOTE:|=+!"   rem cmd would treat a pipe in the text as a pipe

rem ---- YaRN context extension (needed when CTX > the model's training window) ----
set "ROPEARGS="
set "SCALE=1"
if "%ROPE%"=="1" (
    set /a SCALE=%CTX%/%ORIGCTX%
    if !SCALE! GTR 1 set "ROPEARGS=--override-kv %ARCH%.context_length=int:%CTX% --rope-scaling yarn --rope-scale !SCALE! --yarn-orig-ctx %ORIGCTX%"
)

echo.
echo  ================================================================
echo   Bonsai 1.7B agent API  (Ternary-Bonsai-1.7B PQ2_0, 2-bit, 0.43 GiB, CPU only)
echo   endpoint  : http://127.0.0.1:%PORT%/v1      model: bonsai-1.7b
echo   web UI    : http://127.0.0.1:%PORT%/
echo   context   : %CTX%      KV cache: %CTK%      threads: %THREADS%
if "!SCALE!" GTR 1 echo   window   : %CTX% via YaRN RoPE extension (native %ORIGCTX%, scale !SCALE!)
if "!SCALE!" LEQ 1 echo   window   : %CTX% native, no RoPE extension (best recall)
echo   speed     : about 34 t/s generate, 100 t/s prompt at this window (measured)
echo   note      : a bigger window is slower - 64k ~21 t/s, 128k ~18 t/s (same request)
echo   cache     : %CACHEMB% MiB prefix cache, %CTXCP% checkpoints / %CPMINSTEP% tok, KV-shift from %CPCREUSE% tok, context-shift on
echo   RAM cap   : about %P_EST% GiB peak (limit %P_BUDGET% GiB, keep %HEADROOM% pct free)
echo   RAM now   : %P_USED% GiB used of %P_TOTAL% GiB (keep %HEADROOM% pct free)
echo   CPU cap   : %THREADS% threads, hard cap %MAXTHREADS% of 6 cores, prio %LOWPRI%
echo  ================================================================
if not "%P_NOTE%"=="" echo   [auto] %P_NOTE%
echo.
echo   health : curl http://127.0.0.1:%PORT%/health
echo   chat   : curl http://127.0.0.1:%PORT%/v1/chat/completions -H "Content-Type: application/json" -d "{""model"":""bonsai-1.7b"",""messages"":[{""role"":""user"",""content"":""hi""}]}"
echo.
echo   Loading the model takes about 25 seconds. Then it stays up.
echo   Ctrl+C to stop.  KEEP THIS WINDOW OPEN.
echo.

set "ARGS=-m %MODEL% --alias bonsai-1.7b -ngl 0 -fa on -c %CTX% -ctk %CTK% -ctv %CTK% -t %THREADS% -tb %TBATCH% -ub %UBATCH% -n 0 --host 127.0.0.1 --port %PORT% --parallel %NPAR% --cache-ram %CACHEMB% --ctx-checkpoints %CTXCP% --checkpoint-min-step %CPMINSTEP% --cache-reuse %CPCREUSE% --cache-idle-slots --context-shift --jinja -rea off --temp 0.5 --top-p 0.85 --top-k 20 --min-p 0"

if "%LOWPRI%"=="1" (
    powershell -NoProfile -Command "$p = Start-Process -FilePath '.\%BIN%' -ArgumentList '%ARGS% %ROPEARGS%' -NoNewWindow -PassThru; $p.PriorityClass = 'BelowNormal'; $p.WaitForExit()"
) else (
    "%BIN%" %ARGS% %ROPEARGS%
)

echo.
echo [done] server stopped.
pause
