@echo off
REM ===============================================================
REM  agent-lfm12-fast.cmd  --  LFM2.5-1.2B-Instruct Q4_K_M, CPU only
REM
REM  2026-10-13: this file ABSORBED agent-lfm25.cmd and that one was
REM  deleted.  agent-lfm25.cmd had a PICK knob whose "fast" branch
REM  loaded this exact GGUF under this exact alias (lfm25-1.2b) with
REM  the same CTX/KV/threads -- two scripts, one model.  The PICK
REM  "quality" branch is gone too: the 2.6B has its own script,
REM  agent-lfm26.cmd, so nothing needs a switch any more.
REM
REM  Two bugs fixed in the merge, both found by diffing the two files:
REM   1. The old launch line was literally `start "" /min powershell`
REM      with NO command after it.  It opened an empty PowerShell
REM      window and never started a server at all -- 8083 was dead.
REM      Launch is now foreground with WaitForExit, like lfm25/lfm26.
REM   2. The planner's KV downgrade was printed but never passed to
REM      llama-server.  RUN_K is now used for -ctk/-ctv.
REM  Plus: the planner -Only ERROR gate, the friendly "port in use"
REM  skip, and a banner that is printed BEFORE the launch (before, the
REM  whole banner vanished with the window).
REM
REM  The fastest thing on this box that can still drive a tool loop.
REM  Measured 2026-10-12, llama-bench b10743, -t 4 -ub 512 -p N -n 32:
REM
REM    context        tg32          pp
REM      2048       21.69 t/s     163.2 t/s
REM      8192       21.96 t/s        -        <- NO decay
REM     16384       22.12 t/s     112.5 t/s   <- NO decay
REM
REM  Zero decay from 2k to 16k is not a typo.  This is the whole point
REM  of the lfm2 architecture: only 6 of 16 layers are real GQA
REM  attention (8 kv_head x 64 dim), the other 10 are 3-token short
REM  convolutions that hold NO KV cache at all.  So:
REM
REM      KV/token = 6 x 8 x 64 x 2 (K+V) x 2 B = 12 KiB   (f16)
REM      16384 ctx = 0.188 GiB      32768 ctx = 0.375 GiB
REM
REM  Compare Qwen3-1.7B-Q4_K_M (qwen3 arch, 28 full-attn layers):
REM      KV/token = 224 KiB  -- 19x more expensive per token.
REM    measured tg:  2048 -> 15.19 t/s   8192 -> 14.02 t/s
REM    and at 16384 it does not complete on this box (aborted twice --
REM    16384 x 224 KiB = 3.5 GiB of KV, prefill is O(n^2)).  It is also
REM    a plain instruct model with no agentic post-training, so it is
REM    not a substitute for this one even ignoring speed.
REM
REM  WHY THIS ONE: 27% faster decode than the 8B-A1B MoE (21.7 vs
REM  13.8 tg32 at the same -t 4), 6x faster prefill (163 vs 26 t/s on
REM  IQ2_M / 74 on Q4_K_M), 694 MB of weights instead of 4.96 GB, and
REM  it returns NATIVE OpenAI tool_calls -- verified 3/3 by
REM  tool_test.py.  The 8B-A1B is the better pick only when you need
REM  its 8B-level knowledge; see README.md 4.8.
REM
REM  KNOWN LIMITATION (measured): 1.2B occasionally mis-transcribes
REM  paths -- asked for llama-plan.ps1 it answered "D:/AI/LLMLocal/
REM  models", a wrong directory.  For an agent that executes file
REM  paths, VALIDATE every path on the client side, or use
REM  agent-lfm26.cmd (the 2.6B writes exact paths).
REM
REM  ENDPOINT : http://127.0.0.1:8080/v1      MODEL : lfm25-1.2b
REM  WEB UI   : http://127.0.0.1:8080/      (use 127.0.0.1, not
REM             localhost -- the server binds IPv4 only and some
REM             browsers resolve localhost to ::1 first)
REM
REM  KEEP THIS WINDOW OPEN. Closing it, or Ctrl+C here, stops the
REM  server -- the launch is foreground on purpose.
REM ===============================================================
chcp 65001 >nul
title LFM2.5-1.2B Agent API - 127.0.0.1:8080

REM -------------------------- knobs --------------------------
set "MODEL=%~dp0models\LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
set "ALIAS=lfm25-1.2b"
REM 8080 because that is what ~/.pi/agent/models.json points at with
REM model id lfm25-1.2b -- do not move it without editing that file.
REM agent-lfm26.cmd (the 2.6B) also uses 8080; only one at a time.
REM The rest of the map: 8081 MiniCPM5, 8082 8B-A1B, 8084 SakThai.
REM 8083 was this script's old port and is now free (was dead anyway,
REM see bug 1 in the header).
set "PORT=8080"
REM CTX=16384.  Chosen from the measured curve, not from the model's
REM native 128k: decode is flat to 16k AND the real agent prefix is only
REM 754 tokens, so 16k is 20x headroom for a tool loop.  Because
REM --context-shift is on, a lower ceiling would slide the window even
REM sooner -- and since this model's kv is 12 KiB/token, there is no
REM speed reason to go higher, only memory.  32768 costs 0.19 GiB more.
REM Raise to 32768 only for one-shot long-document work.
set "CTX=16384"
REM f16 KV, NOT q8_0/q4_0.  Measured on the 8B-A1B: a quantized KV
REM makes llama.cpp dequantize K/V inside the O(n^2) prefill path --
REM prefill 70.4 -> 36.0 t/s (q4_0) for only +53% decode at the end.
REM An agent waits on the COLD prefill every turn, so f16 wins.
REM Here f16 is also nearly free: 16384 x 12 KiB = 0.188 GiB.
set "CTK=f16"
set "CTV=f16"
REM -t 4 measured fastest on this 6-core box and leaves 2 cores for
REM Windows.  -t 6 is actively WORSE here (Qwen3-1.7B: 15.19 -> 9.29).
REM Decode is memory-bus-bound, so more threads just fight the bus.
set "THREADS=4"
set "MAXT=5"
REM -tb 6: prefill wants every core while decode keeps -t 4.  -t and -tb
REM are independent knobs, so use both.  (llama-bench itself does NOT
REM accept -tb, so these two numbers cannot be reproduced with it.)
REM agent-lfm25.cmd used TB=4; 6 is the tuned value for this model.
set "TB=6"
REM ubatch 512 measured faster than 2048 on this CPU.  Note this is the
REM OPPOSITE of the MoE 8B, which needs ub 2048 (92 vs 78 t/s) so that
REM all 32 experts get a real GEMM.  Dense model -> ubatch is a wash,
REM and 512 measured best.
set "UB=512"
REM Working set at 16384/f16 is ~1.2 GiB (planner lfm12 table).  6 GiB
REM cap is generous and leaves the box responsive.
set "MAXRAM=6"
set "CACHEMB=2048"      rem --cache-ram MiB. This is the single biggest
REM                       CPU win: a 2275-token system prompt costs 82.6 s
REM                       cold and 0.67 s warm (99.2% reused).  This is a
REM                       CAP, not a reservation -- it commits nothing
REM                       until hit, so do NOT lower it to save RAM.
set "CTXCP=32"          rem --ctx-checkpoints per slot
set "CMS=256"           rem --checkpoint-min-step. Default 8192 is far
REM                       too coarse for an agent -- a 754-token prefix
REM                       could never build a checkpoint.
set "CPCREUSE=64"       rem --cache-reuse: min chunk for KV-shifting
REM parallel slots; more than 1 splits CTX between them.  NOTE: the
REM ">" of the old "rem >1" wording had to go: it sits OUTSIDE the
REM quotes of the set line below, so cmd parsed it as a redirect and
REM tried to write to a file named "1" in this directory.
set "NPAR=1"
set "LOWPRI=1"          rem BelowNormal so the desktop stays smooth
REM Sampling MUST match the model card: temp 0.1 / top_k 50 /
REM repeat 1.1.  LFM2.5 is trained in a low-entropy regime -- raising
REM the temperature to qwen-ish 0.7/0.95 makes the tool_call JSON
REM break.  Do not "tune" these.
set "TEMP=0.1"
set "TOPK=50"
set "REPEAT=1.1"
REM 1.2B-Instruct does NOT think (no reasoning_content), so 400 is
REM plenty -- that was MAXTOK in agent-lfm25.cmd's fast branch.  Do
REM not copy the 2.6B's 1400 here, that is the thinking model.
set "MAXTOK=400"
REM WARM=1 prefills the frozen agent prefix in the background at
REM start-up so the first real question skips the cold prefill.
REM Worth it here: cold is ~4 s, not minutes.
set "WARM=0"
set "SERVE=%~dp0Bonsai-demo\bin\cpu\llama-server.exe"
REM -----------------------------------------------------------

echo ================================================================
echo  LFM2.5-1.2B-Instruct  --  fastest agent-capable config on this box
echo  model %MODEL%
echo  port  http://127.0.0.1:%PORT%/v1    (alias %ALIAS%)
echo ================================================================

if not exist "%SERVE%" (
  echo [ERR] not found: %SERVE%
  echo       must use bin\cpu\ -- bin-vulkan\ tries the iGPU and dies
  echo       with "ggml_vulkan: Device memory allocation ... failed"
  pause & exit /b 1
)
if not exist "%MODEL%" (
  echo [ERR] not found: %MODEL%
  echo       bash dl.sh LiquidAI/LFM2.5-1.2B-Instruct-GGUF LFM2.5-1.2B-Instruct-Q4_K_M.gguf models
  pause & exit /b 1
)
if not exist "%~dp0llama-plan.ps1" (
  echo [ERR] not found: %~dp0llama-plan.ps1
  pause & exit /b 1
)

REM ---- ask the planner what we can afford right now ----
REM NOTE: arrow below MUST stay escaped as ^> -- a bare > is a shell
REM redirection and would dump the whole line into a file named "f16".
REM This file is deliberately pure ASCII: cmd.exe decodes UTF-8 .cmd on
REM codepage 936 and mis-pairs Chinese bytes into pipe characters, which
REM then get EXECUTED as commands.
REM RUN_K is seeded from CTK so a planner that prints nothing still
REM gives the server a valid kv type; the loop below overwrites it only
REM if the planner answered.
set "RUN_CTX=%CTX%" & set "RUN_CTK=%CTK%" & set "RUN_T=%THREADS%" & set "RUN_K=%CTK%"
set "PSPLAN=powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0llama-plan.ps1" -Size lfm12 -Ctx %RUN_CTX% -CtK %RUN_CTK% -Threads %RUN_T% -MaxRam %MAXRAM% -MaxThreads %MAXT% -HeadroomPct 10"
for /f %%c in ('%PSPLAN% -Only CTX')   do set "RUN_CTX=%%c"
for /f %%k in ('%PSPLAN% -Only CTK')   do set "RUN_K=%%k"
for /f %%t in ('%PSPLAN% -Only THREADS') do set "RUN_T=%%t"
echo [PLAN] ctx=%RUN_CTX%  kv=%RUN_K%  threads=%RUN_T%
for /f "tokens=*" %%v in ('%PSPLAN% -Only NOTE') do echo   [auto] %%v

REM ---- refuse to start when even the smallest plan does not fit ----
REM Took this from agent-lfm25.cmd, which had it and this script did
REM not.  Without it llama-server starts, thrashes and gets
REM OOM-killed a minute later, which reads as "the model is broken".
set "PLAN_ERR=0"
for /f %%e in ('%PSPLAN% -Only ERROR') do set "PLAN_ERR=%%e"
if "%PLAN_ERR%"=="1" (
  echo.
  echo  [ERR] Not enough RAM headroom for even the smallest plan.
  for /f "tokens=*" %%n in ('%PSPLAN% -Only NOTE') do echo        %%n
  echo        Close some apps and run again, or lower CTX in this file.
  pause & exit /b 1
)

REM ---- port already taken? do not start a second copy ----
REM The old line here only WARNed and went on to launch, so the new
REM server died on bind with a message that looks unrelated.
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

echo.
echo [RUN] %ALIAS%  ctx=%RUN_CTX%  KV=%RUN_K%  t=%RUN_T%  tb=%TB%  ub=%UB%  port=%PORT%
echo       decode  ~21.7 t/s, FLAT from 2k to 16k (12 KiB/token KV)
echo       prefill ~163 t/s at 2k, ~112 t/s at 16k
echo       memory ~1.2 GiB at 16384 f16.  prefix cache ON
echo       (cache-ram=%CACHEMB% MiB, ctx-checkpoints=%CTXCP%,
echo        checkpoint-min-step=%CMS%) -- cold prefix 82.6 s, warm 0.67 s
echo       no thinking model: max_tokens %MAXTOK% per turn is plenty
echo.
echo       health : curl http://127.0.0.1:%PORT%/health
echo       test   : cd /d %~dp0 ^& set AGENT_MAXTOK=%MAXTOK% ^& python tool_test.py %PORT% %ALIAS% %TEMP%
echo       files  : cd /d %~dp0 ^& set AGENT_MAXTOK=%MAXTOK% ^& python fs_probe.py %PORT% %ALIAS% %TEMP% --clean
echo       loop   : cd /d %~dp0 ^& set AGENT_MAXTOK=%MAXTOK% ^& python agent_loop.py 5 %PORT% %ALIAS% %TEMP%
echo       stop   : Ctrl+C in THIS window (or powershell -c "Stop-Process -Name llama-server -Force")
echo.

REM ---- optional: prefill the agent prefix in the background ----
REM It polls /health first, so it is safe to start before the server.
if "%WARM%"=="1" (
  echo [WARM] waiting for server, then prefilling the frozen agent prefix ...
  start "warm" /min powershell -NoProfile -Command "for($i=0;$i -lt 180;$i++){try{if((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:%PORT%/health').StatusCode -eq 200){break}}catch{};Start-Sleep 1}; $env:AGENT_MAXTOK='%MAXTOK%'; python '%~dp0agent_loop.py' 0 %PORT% %ALIAS% --warm"
) else (
  echo [WARM] off -- set WARM=1 to prefill the agent_loop.py prefix at startup
)

REM ---- go foreground: this blocks until the server exits ----
REM --jinja is MANDATORY, do not drop it: it loads the chat template
REM baked into the GGUF (supports_tools=true).  Without it this 1.2B
REM emits calls as plain prose and tool_calls comes back empty -- it
REM looks exactly like "the model has no tool support".
set "ARGS=-m %MODEL% --alias %ALIAS% -ngl 0 -fa on -c %RUN_CTX% -ctk %RUN_K% -ctv %RUN_K% -t %RUN_T% -tb %TB% -ub %UB% -n 0 --host 127.0.0.1 --port %PORT% --parallel %NPAR% --cache-ram %CACHEMB% --ctx-checkpoints %CTXCP% --checkpoint-min-step %CMS% --cache-reuse %CPCREUSE% --cache-idle-slots --context-shift --jinja -rea off --temp %TEMP% --top-k %TOPK% --repeat-penalty %REPEAT%"

if "%LOWPRI%"=="1" (
  powershell -NoProfile -Command "$p = Start-Process -FilePath '%SERVE%' -ArgumentList '%ARGS%' -NoNewWindow -PassThru; $p.PriorityClass = 'BelowNormal'; $p.WaitForExit()"
) else (
  "%SERVE%" %ARGS%
)
echo.
echo [done] server stopped.
pause
endlocal