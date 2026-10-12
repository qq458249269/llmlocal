@echo off
REM ===============================================================
REM  agent-lfm26.cmd  --  LFM2.5-2.6B Q4_K_M, CPU only
REM
REM  WHY THIS SCRIPT: the 1.2B is the fastest agent base on this box
REM  (21.7 t/s) but it cannot be trusted with file paths.  Measured
REM  inside pi on 2026-10-12: asked to create 1.txt it called
REM  write() four times and every single `path` was a DIRECTORY
REM  (D:/user/Desktop), so every call died with EEXIST/EISDIR and
REM  the model then invented "I cannot access the filesystem".
REM  tool_test.py still said 3/3 -- it only checks the tool NAME.
REM
REM  The 2.6B writes the full path correctly the moment you give it
REM  one (fs_probe.py: landed, exit 0).  It costs 10.3 t/s instead of
REM  21.7.  Use the 1.2B for read-only work, this one for anything
REM  that writes.
REM
REM  IT IS A THINKING MODEL -- max_tokens must be >= 1400.  At 600 it
REM  spends the whole budget on <think> and returns
REM  finish_reason=length with content='' , which looks exactly like
REM  "this model cannot call tools".  See README section 6.
REM
REM  ENDPOINT : http://127.0.0.1:8080/v1      MODEL : lfm25-2.6b
REM  WEB UI   : http://127.0.0.1:8080/      (use 127.0.0.1, not
REM             localhost -- the server binds IPv4 only and some
REM             browsers resolve localhost to ::1 first)
REM
REM  KEEP THIS WINDOW OPEN. Closing it stops the server.
REM ===============================================================
setlocal EnableDelayedExpansion
chcp 65001 >nul
title LFM2.5-2.6B Agent API - 127.0.0.1:8080

REM -------------------------- knobs --------------------------
set "MODEL=%~dp0models\LFM2.5-2.6B-Q4_K_M.gguf"
set "ALIAS=lfm25-2.6b"
REM 8080 is what ~/.pi/agent/models.json points at; 8081 MiniCPM5,
REM 8082 8B-A1B, 8084 SakThai.  The 1.2B (agent-lfm12-fast.cmd) also
REM uses 8080 -- one at a time.  Only one server at a time.
set "PORT=8080"
REM 16384, not 32768: --context-shift is on, so -c is the ceiling the
REM sliding window uses, and current KV length is what actually costs
REM decode speed.  16384 f16 is ~0.25 GiB of KV, still 20x the real
REM agent prefix.  2.6B native window is 128k, so this needs no RoPE
REM extension.  Raise to 32768 only for one-shot long-document work.
set "CTX=16384"
REM q8_0 KV (symmetric, one fp16 scale per 32-block). Only because the
REM engine is now ik_llama.cpp (see SERVE below): its IQK FlashAttention
REM CPU kernels read the quantized KV directly, so there is no dequant
REM pass in the O(n^2) path.  Measured with llama-bench -t 4 -p 2048
REM (same model, same box):  ik_llama.cpp f16 pp 84.8 / q8_0 pp 88.5,
REM old b10743 f16 pp 67.2 / q8_0 pp 55.2 t/s.  So on ik_llama.cpp
REM q8_0 is FASTER than f16 while halving KV RAM; on the old build it
REM was the other way round, which is why this line used to say f16.
REM q4_0 measured 83.6 -- slower than q8_0 here, so do not go lower.
set "CTK=q8_0"
set "CTV=q8_0"
REM -t 4 measured fastest on this 6-core box and leaves 2 cores for
REM Windows.  Decode is memory-bus-bound, so more threads just fight
REM for the bus (-t 6 measured 8.56 t/s vs 9.94 at t=4).
set "THREADS=4"
set "MAXT=5"
REM -tb 4: prefill wants every core while decode keeps -t 4.  -t and
REM -tb are independent knobs, so use both.  (llama-bench itself does
REM NOT accept -tb, so these two numbers cannot be reproduced with it.)
set "TB=4"
REM ubatch 512 measured faster than 2048 on this CPU.  Note this is
REM the OPPOSITE of the MoE 8B, which needs ub 2048 (92 vs 78 t/s) so
REM that all 32 experts get a real GEMM.  Dense model -> wash.
set "UB=512"
REM Working set at 16384/f16 is ~2.1 GiB (planner lfm25 table).  6 GiB
REM cap is generous and leaves the box responsive.
set "MAXRAM=6"
set "CACHEMB=2048"      rem --cache-ram MiB. This is the single biggest
REM                       CPU win: a 2275-token system prompt costs 82.6 s
REM                       cold and 0.67 s warm (99.2% reused).  This is a
REM                       CAP, not a reservation -- it commits nothing
REM                       until hit, so do NOT lower it to save RAM.
set "CTXCP=32"          rem --ctx-checkpoints per slot
set "CMS=256"           rem --ctx-checkpoints-interval. Default 512 is too
REM                       coarse for an agent -- a 754-token prefix could
REM                       never build a checkpoint.
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
REM Printed as a hint only -- the server does not enforce it.  Every
REM test script in this repo reads AGENT_MAXTOK from the env; export it
REM before calling them or the thinking will eat the budget.
set "MAXX=1400"
REM WARM=1 prefills the frozen agent prefix in the background at
REM start-up so the first real question skips the cold prefill.
set "WARM=0"
REM ik_llama.cpp built for AVX2 (see README). Its IQK FlashAttention
REM CPU kernels take the quantized KV above straight in the attention
REM kernel -- that is what makes q8_0 KV a prefill WIN here instead of
REM the loss it was on b10743.  Falls back to the old engine if you
REM have not built it yet.
set "SERVE=%~dp0ik_llama-src\build\bin\llama-server.exe"
if not exist "%SERVE%" (
  echo [WARN] ik build missing: %SERVE%
  echo        falling back to the old engine -- but rebuild it, the new
  echo        engine is the only reason q8_0 KV is a prefill win.
  set "SERVE=%~dp0Bonsai-demo\bin\cpu\llama-server.exe"
)
REM -----------------------------------------------------------

echo ================================================================
echo  LFM2.5-2.6B  --  the one to use when the agent writes files
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
  echo       bash dl.sh LiquidAI/LFM2.5-2.6B-GGUF LFM2.5-2.6B-Q4_K_M.gguf models
  pause & exit /b 1
)
if not exist "%~dp0llama-plan.ps1" (
  echo [ERR] not found: %~dp0llama-plan.ps1
  pause & exit /b 1
)

REM ---- ask the planner what we can afford right now ----
REM NOTE: arrow below MUST stay escaped as ^> -- a bare > is a shell
REM redirection and would dump the whole line into a file named "f16".
REM Every `>` in an `echo` line needs it too: an unescaped one here
REM silently created a 57-byte file literally named "1400".
REM This file is deliberately pure ASCII: cmd.exe decodes UTF-8 .cmd on
REM codepage 936 and mis-pairs non-ASCII bytes into pipe characters,
REM which then get EXECUTED as commands.
REM RUN_K is seeded from CTK so a planner that prints nothing still
REM gives the server a valid kv type.  Seeded before the loop on
REM purpose: the loop below overwrites it only if the planner answered.
set "RUN_CTX=%CTX%" & set "RUN_CTK=%CTK%" & set "RUN_T=%THREADS%" & set "RUN_K=%CTK%"
set "PSPLAN=powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0llama-plan.ps1" -Size lfm25 -Ctx %RUN_CTX% -CtK %RUN_CTK% -Threads %RUN_T% -MaxRam %MAXRAM% -MaxThreads %MAXT% -HeadroomPct 10"
for /f %%c in ('%PSPLAN% -Only CTX')   do set "RUN_CTX=%%c"
for /f %%k in ('%PSPLAN% -Only CTK')   do set "RUN_K=%%k"
for /f %%t in ('%PSPLAN% -Only THREADS') do set "RUN_T=%%t"
echo [PLAN] ctx=%RUN_CTX%  kv=^>%RUN_K%  threads=%RUN_T%
for /f "tokens=*" %%v in ('%PSPLAN% -Only NOTE') do echo   [auto] %%v

REM ---- refuse to start when even the smallest plan does not fit ----
REM agent-lfm25.cmd (now folded into agent-lfm12-fast.cmd) already had
REM this; lfm26 was the only script that
REM started anyway and let llama-server get OOM-killed two minutes
REM later, which reads as "the model is broken" rather than "no RAM".
set "PLAN_ERR=0"
for /f %%e in ('%PSPLAN% -Only ERROR') do set "PLAN_ERR=%%e"
if "%PLAN_ERR%"=="1" (
  echo.
  echo  [ERR] Not enough RAM headroom for even the smallest plan.
  for /f "tokens=*" %%n in ('%PSPLAN% -Only NOTE') do echo        %%n
  echo        Close some apps and run again, or lower CTX in this file.
  pause & exit /b 1
)

netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 echo [WARN] port %PORT% already in use -- stop that server first

REM ---- launch: FOREGROUND, BelowNormal so the desktop stays smooth ----
REM FOREGROUND on purpose (it used to be `start "" /min powershell`,
REM which detached the server and let this window close instantly).
REM Detaching cost two things: Ctrl+C in the window no longer stopped
REM the server -- you had to remember Stop-Process -- and every banner
REM line printed AFTER the launch line was invisible, because the window
REM was already gone.  Same shape as agent-lfm12-fast.cmd: the server
REM log lands in this window, Ctrl+C here stops it.
REM --jinja is MANDATORY, do not drop it: it loads the chat template
REM baked into the GGUF (supports_tools=true).  Without it this model
REM emits calls as plain prose and tool_calls comes back empty -- it
REM looks exactly like "the model has no tool support".
set "ARGS=-m %MODEL% --alias %ALIAS% -ngl 0 -fa 1 -c %RUN_CTX% -ctk %RUN_K% -ctv %RUN_K% -t %RUN_T% -tb %TB% -ub %UB% -n 0 --host 127.0.0.1 --port %PORT% --parallel %NPAR% --cache-ram %CACHEMB% --ctx-checkpoints %CTXCP% --ctx-checkpoints-interval %CMS% --context-shift on --jinja -rea off --temp %TEMP% --top-k %TOPK% --repeat-penalty %REPEAT%"

echo.
echo [RUN] %ALIAS%  ctx=%RUN_CTX%  KV=%RUN_K%  t=%RUN_T%  tb=%TB%  ub=%UB%  port=%PORT%
echo       decode  ~8.6 t/s, prefill ~88.5 t/s (llama-bench -t 4 -p 2048, ik engine)
echo       memory  ~1.1 GiB at 16384 q8_0 KV (half of f16).  prefix cache ON
echo       (cache-ram=%CACHEMB% MiB, ctx-checkpoints=%CTXCP%,
echo        ctx-checkpoints-interval=%CMS%)
echo       THINKING model -- give every request max_tokens ^>= %MAXX%,
echo       or it will finish_reason=length with an empty answer.
echo.
echo       test   : cd /d %~dp0 ^& set AGENT_MAXTOK=%MAXX% ^& python tool_test.py %PORT% %ALIAS% %TEMP%
echo       files  : cd /d %~dp0 ^& set AGENT_MAXTOK=%MAXX% ^& python fs_probe.py %PORT% %ALIAS% %TEMP% --clean
echo       loop   : cd /d %~dp0 ^& set AGENT_MAXTOK=%MAXX% ^& python agent_loop.py 5 %PORT% %ALIAS% %TEMP%
echo       stop   : Ctrl+C in THIS window (or powershell -c "Stop-Process -Name llama-server -Force")
echo.

REM ---- optional: prefill the agent prefix in the background ----
REM It polls /health first, so it is safe to start before the server.
if "%WARM%"=="1" (
  echo [WARM] waiting for server, then prefilling the frozen agent prefix ...
  start "warm" /min powershell -NoProfile -Command "for($i=0;$i -lt 180;$i++){try{if((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:%PORT%/health').StatusCode -eq 200){break}}catch{};Start-Sleep 1}; $env:AGENT_MAXTOK='%MAXX%'; python '%~dp0agent_loop.py' 0 %PORT% %ALIAS% --warm"
) else (
  echo [WARM] off -- set WARM=1 to prefill the agent_loop.py prefix at startup
)

REM ---- go foreground: this blocks until the server exits ----
if "%LOWPRI%"=="1" (
  powershell -NoProfile -Command "$p = Start-Process -FilePath '%SERVE%' -ArgumentList '%ARGS%' -NoNewWindow -PassThru; $p.PriorityClass = 'BelowNormal'; $p.WaitForExit()"
) else (
  "%SERVE%" %ARGS%
)
echo.
echo [done] server stopped.
pause
endlocal
