@echo off
rem ====================================================================
rem  LFM2.5 local OpenAI-compatible SERVER for agents  (double-click)
rem  llama.cpp b10743 (PrismML fork), CPU only, no GPU needed.
rem
rem  PICK=fast    LFM2.5-1.2B-Instruct Q4_K_M   0.68 GiB   ~19-24 t/s
rem  PICK=quality LFM2.5-2.6B        Q4_K_M   1.63 GiB   ~10 t/s
rem
rem  WHY THIS FAMILY: post-trained for agentic tool use, returns native
rem  OpenAI tool_calls. The arch is lfm2: 30 layers of which only 8 are GQA
rem  attention, the other 22 are 3-token short convolutions that use NO KV
rem  cache at all. So the KV cache is only 16 KiB/token f16, about 9x
rem  cheaper per token than a qwen3 model - a real agent context is nearly
rem  free here, and that is what makes a 32k window practical on 20 GB.
rem
rem  MEASURED ON THIS BOX (i5-8500, 6 cores, ~16 GB/s RAM bandwidth):
rem    1.2B  gen 19.5-24 t/s   prefill 176 t/s   tool calls 3/3   5-turn loop 19 s
rem    2.6B  gen 10.4 t/s      prefill  65 t/s   tool calls 3/3   5-turn loop 202 s
rem  The 2.6B writes exact arguments (full Windows paths); the 1.2B is much
rem  faster but occasionally typos a path. That is the whole trade-off.
rem
rem  ENDPOINT : http://127.0.0.1:8080/v1      MODEL : lfm25-1.2b or lfm25-2.6b
rem  WEB UI   : http://127.0.0.1:8080/       (use 127.0.0.1, not localhost -
rem             the server binds IPv4 only and some browsers resolve
rem             localhost to ::1 first)
rem
rem  KEEP THIS WINDOW OPEN. Closing it stops the server.
rem
rem  TOOL CALLING: native, no prompt hacks. The model writes Pythonic calls
rem  between <tool_call_start> / <tool_call_end> special tokens; llama.cpp
rem  >= b10455 parses them into OpenAI tool_calls. This build has the parser.
rem  Use llm_client.py for the prefix-cache discipline.
rem
rem  THINKING: the 2.6B is a reasoning model - it spends 60-460 tokens of
rem  "reasoning_content" before the answer or the tool call, and those
rem  tokens come out of the same max_tokens budget. Give it 600+, or a
rem  long turn ends with finish_reason=length and an EMPTY content.
rem  The 1.2B-Instruct does not think, so 300-400 is plenty.
rem
rem  SELF-LIMITING: llama-plan.ps1 measures free RAM right now and never goes
rem  above (total - 10 pct headroom - in use), downgrading KV type / context if
rem  needed. Threads are hard-capped at 5 of 6 cores.
rem
rem  SAMPLING: the model card's values - temp 0.1, top_k 50, repeat 1.1.
rem  Do not "improve" these to 0.7/0.95 like a Qwen-style model; LFM2.5 is
rem  trained for the low-entropy regime and a hot temp breaks tool-call JSON.
rem
rem  DO NOT put a pipe character anywhere in this file, even inside a rem
rem  comment on a set line. cmd.exe splits the whole line on "|" first, so
rem  "set X=1 rem a | b" silently runs "b" as a command. (That is the same
rem  trap as the non-ASCII one above, and it fails just as silently.)
rem
rem  PREFIX CACHE (biggest CPU win): keep system prompt + tools first and
rem  byte-identical, tools array order fixed, history append-only. Measured
rem  92-98 pct reuse after the first turn, which cuts prefill to ~0.3 s.
rem
rem  DO NOT use speculative decoding here. The official DSpark drafter
rem  (LFM2.5-2.6B-DSpark-Q4_K_M, 0.2 GiB) loads and initialises, but on
rem  this CPU build it accepts 0 of 780 drafted tokens, so it only adds
rem  overhead: 9.9 t/s -> 1.1 t/s. Do not pass -md / --spec-draft-model.
rem
rem  NOTE: this file is deliberately pure ASCII. cmd.exe decodes a UTF-8 .cmd
rem  with the system codepage (936 here) and mis-pairs non-ASCII bytes into
rem  pipe characters, which then runs comment tails as commands.
rem ====================================================================
setlocal EnableDelayedExpansion
title LFM2.5 Agent API - 127.0.0.1:8080

rem -------------------------- knobs --------------------------
set "PICK=fast"         rem fast = 1.2B-Instruct, quality = 2.6B (see PICK block)
if /i "%PICK%"=="quality" (
    set "MODEL=..\models\LFM2.5-2.6B-Q4_K_M.gguf"
    set "ALIAS=lfm25-2.6b"
    set "PSIZE=lfm25"
    set "MAXTOK=600"
    set "TAG=2.6B (reasoning model - give it MAXTOK>=600)"
) else (
    set "MODEL=..\models\LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
    set "ALIAS=lfm25-1.2b"
    set "PSIZE=lfm12"
    set "MAXTOK=400"
    set "TAG=1.2B-Instruct (no thinking - fast)"
)
set "CTX=16384"        rem agent window. Lowered from 32768 on 2026-10-12.
rem                       Why: --context-shift is ON below, so -c is the ceiling
rem                       the sliding window uses. A tighter ceiling keeps the
rem                       CURRENT kv length small, and current length is what
rem                       actually costs decode speed (measured 1.2B: 24 t/s
rem                       short -> ~20 t/s at 2k kv). It does NOT cost quality:
rem                       12-16 KiB/token f16, so 16384 is only 0.19-0.25 GiB
rem                       and still 20x the real agent prefix.
rem                       Both models stay inside their native window here, so
rem                       still NO RoPE extension. Set 32768 for the 2.6B, or
rem                       128000 if you must read a huge file in one go.
set "CTK=f16"           rem f16 is affordable here (12-16 KiB/token, NOT the
rem                       16 KiB quoted before -- kvcalc.py measures 12.0 for
rem                       the 1.2B: 6 attn layers x 8 kv_head x 64 dim x 2x2B).
rem                       q8_0/q4_0 only make sense once CTX reaches 128000.
set "THREADS=4"         rem measured fastest on this 6-core box, leaves 2 free
set "TBATCH=4"
set "UBATCH=512"        rem measured faster than 2048 on this CPU
set "MAXRAM=6"          rem HARD cap GiB for llama-server.exe
set "MAXTHREADS=5"      rem HARD cap for llama threads
set "HEADROOM=10"       rem pct RAM that must stay free for Windows + apps
set "PORT=8080"
set "NPAR=1"            rem parallel slots; >1 splits CTX between them
set "CACHEMB=2048"      rem --cache-ram: MiB budget for cached prefixes
set "CTXCP=32"          rem --ctx-checkpoints per slot
set "CPMINSTEP=256"     rem --checkpoint-min-step (default 8192 is too big)
set "CPCREUSE=64"       rem --cache-reuse: min chunk for KV-shifting
set "LOWPRI=1"          rem 1 = BelowNormal so the desktop stays smooth
set "TEMP=0.1"          rem model card
set "TOPK=50"           rem model card
set "REPEAT=1.1"        rem model card
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
set "PSPLAN=powershell -NoProfile -ExecutionPolicy Bypass -File %PLANPS% -Size %PSIZE% -Ctx %CTX% -CtK %CTK% -Threads %THREADS% -MaxRam %MAXRAM% -MaxThreads %MAXTHREADS% -HeadroomPct %HEADROOM%"
set "PLAN_ERR=0"
for /f %%e in ('%PSPLAN% -Only ERROR') do set "PLAN_ERR=%%e"
if "%PLAN_ERR%"=="1" (
    echo.
    echo  [ERR] Not enough RAM headroom for even the smallest plan.
    for /f "tokens=*" %%n in ('%PSPLAN% -Only NOTE') do echo        %%n
    echo        Close some apps and run again, or lower CTX in this file.
    pause & exit /b 1
)
for /f %%v in ('%PSPLAN% -Only CTX')     do set "CTX=%%v"
for /f %%v in ('%PSPLAN% -Only CTK')     do set "CTK=%%v"
for /f %%v in ('%PSPLAN% -Only THREADS') do set "THREADS=%%v"
set "P_EST=" & set "P_BUDGET=" & set "P_USED=" & set "P_TOTAL="
for /f %%v in ('%PSPLAN% -Only EST')     do set "P_EST=%%v"
for /f %%v in ('%PSPLAN% -Only BUDGET')  do set "P_BUDGET=%%v"
for /f %%v in ('%PSPLAN% -Only USERAM')  do set "P_USED=%%v"
for /f %%v in ('%PSPLAN% -Only TOTRAM')  do set "P_TOTAL=%%v"

echo.
echo  ================================================================
echo   LFM2.5 agent API   [PICK=%PICK%]   %TAG%
echo   model     : %ALIAS%   (%MODEL%)
echo   endpoint  : http://127.0.0.1:%PORT%/v1
echo   web UI    : http://127.0.0.1:%PORT%/
echo   context   : %CTX%      KV cache: %CTK%   threads: %THREADS%
echo   sampling  : temp %TEMP% / top_k %TOPK% / repeat %REPEAT% (model card values)
echo   max_tokens: at least %MAXTOK% per turn (the 2.6B thinks first)
echo   tools     : native OpenAI tool_calls (--jinja, pythonic parser)
echo   cache     : %CACHEMB% MiB prefix cache, %CTXCP% checkpoints / %CPMINSTEP% tok, KV-shift from %CPCREUSE% tok, context-shift on
echo   RAM       : about %P_EST% GiB peak (cap %P_BUDGET% GiB, keep %HEADROOM% pct free)
echo   RAM now   : %P_USED% GiB used of %P_TOTAL% GiB
echo   CPU cap   : %THREADS% threads, hard cap %MAXTHREADS% of 6 cores, prio %LOWPRI%
echo  ================================================================
for /f "tokens=*" %%v in ('%PSPLAN% -Only NOTE') do echo   [auto] %%v
echo.
echo   health : curl http://127.0.0.1:%PORT%/health
echo   test   : cd /d %~dp0 ^& python tool_test.py %PORT% %ALIAS% %TEMP%
echo.
echo   Ctrl+C to stop.  KEEP THIS WINDOW OPEN.
echo.

set "ARGS=-m %MODEL% --alias %ALIAS% -ngl 0 -fa on -c %CTX% -ctk %CTK% -ctv %CTK% -t %THREADS% -tb %TBATCH% -ub %UBATCH% -n 0 --host 127.0.0.1 --port %PORT% --parallel %NPAR% --cache-ram %CACHEMB% --ctx-checkpoints %CTXCP% --checkpoint-min-step %CPMINSTEP% --cache-reuse %CPCREUSE% --cache-idle-slots --context-shift --jinja -rea off --temp %TEMP% --top-k %TOPK% --repeat-penalty %REPEAT%"

if "%LOWPRI%"=="1" (
    powershell -NoProfile -Command "$p = Start-Process -FilePath '.\%BIN%' -ArgumentList '%ARGS%' -NoNewWindow -PassThru; $p.PriorityClass = 'BelowNormal'; $p.WaitForExit()"
) else (
    "%BIN%" %ARGS%
)

echo.
echo [done] server stopped.
pause
