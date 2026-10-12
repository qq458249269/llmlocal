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
REM 8082 8B-A1B, 8083 the 1.2B fast script.  Only one at a time.
set "PORT=8080"
REM 16384, not 32768: --context-shift is on, so -c is the ceiling the
REM sliding window uses, and current KV length is what actually costs
REM decode speed.  16384 f16 is ~0.25 GiB of KV, still 20x the real
REM agent prefix.  2.6B native window is 128k, so this needs no RoPE
REM extension.  Raise to 32768 only for one-shot long-document work.
set "CTX=16384"
REM f16 KV, NOT q8_0/q4_0.  Quantized KV forces llama.cpp to
REM dequantize K/V inside the O(n^2) prefill path -- measured on the
REM 8B-A1B: prefill 70.4 -> 36.0 t/s for q4_0.  An agent waits on the
REM COLD prefill every single turn, so f16 wins.  Here f16 is also
REM cheap because LFM2.5 only stores KV on its 8 full-attention
REM layers (the other 22 are short convolutions): 12-16 KiB/token.
set "CTK=f16"
set "CTV=f16"
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
set "CMS=256"           rem --checkpoint-min-step. Default 8192 is far
REM                       too coarse for an agent -- a 754-token prefix
REM                       could never build a checkpoint.
set "CPCREUSE=64"       rem --cache-reuse: min chunk for KV-shifting
set "NPAR=1"            rem parallel slots; >1 splits CTX between them
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
set "SERVE=%~dp0Bonsai-demo\bin\cpu\llama-server.exe"
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
set "RUN_CTX=%CTX%" & set "RUN_CTK=%CTK%" & set "RUN_T=%THREADS%"
set "PSPLAN=powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0llama-plan.ps1" -Size lfm25 -Ctx %RUN_CTX% -CtK %RUN_CTK% -Threads %RUN_T% -MaxRam %MAXRAM% -MaxThreads %MAXT% -HeadroomPct 10"
for /f %%c in ('%PSPLAN% -Only CTX')   do set "RUN_CTX=%%c"
for /f %%k in ('%PSPLAN% -Only CTK')   do set "RUN_K=%%k"
for /f %%t in ('%PSPLAN% -Only THREADS') do set "RUN_T=%%t"
echo [PLAN] ctx=%RUN_CTX%  kv=^>%RUN_K%  threads=%RUN_T%
for /f "tokens=*" %%v in ('%PSPLAN% -Only NOTE') do echo   [auto] %%v

netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 echo [WARN] port %PORT% already in use -- stop that server first

REM ---- launch: detached, BelowNormal so the desktop stays smooth ----
REM --jinja is MANDATORY, do not drop it: it loads the chat template
REM baked into the GGUF (supports_tools=true).  Without it this model
REM emits calls as plain prose and tool_calls comes back empty -- it
REM looks exactly like "the model has no tool support".
start "" /min powershell -NoProfile -Command "$p=Start-Process -FilePath '%SERVE%' -ArgumentList '-m','%MODEL%','--alias','%ALIAS%','-ngl','0','-fa','on','-c','%RUN_CTX%','-ctk','%CTK%','-ctv','%CTV%','-t','%RUN_T%','-tb','%TB%','-ub','%UB%','-n','0','--host','127.0.0.1','--port','%PORT%','--parallel','%NPAR%','--cache-ram','%CACHEMB%','--ctx-checkpoints','%CTXCP%','--checkpoint-min-step','%CMS%','--cache-reuse','%CPCREUSE%','--cache-idle-slots','--context-shift','--jinja','-rea','off','--temp','%TEMP%','--top-k','%TOPK%','--repeat-penalty','%REPEAT%' -NoNewWindow -PassThru; $p.PriorityClass='BelowNormal'"

echo.
echo [RUN] %ALIAS%  ctx=%RUN_CTX%  KV=%CTK%  t=%RUN_T%  tb=%TB%  ub=%UB%  port=%PORT%
echo       decode  ~10.3 t/s, prefill ~65.5 t/s
echo       memory  ~2.1 GiB at 16384 f16.  prefix cache ON
echo       (cache-ram=%CACHEMB% MiB, ctx-checkpoints=%CTXCP%,
echo        checkpoint-min-step=%CMS%)
echo       THINKING model -- give every request max_tokens ^>= %MAXX%,
echo       or it will finish_reason=length with an empty answer.
echo.
echo       test   : cd /d %~dp0 ^& set AGENT_MAXTOK=%MAXX% ^& python tool_test.py %PORT% %ALIAS% %TEMP%
echo       files  : cd /d %~dp0 ^& set AGENT_MAXTOK=%MAXX% ^& python fs_probe.py %PORT% %ALIAS% %TEMP% --clean
echo       loop   : cd /d %~dp0 ^& set AGENT_MAXTOK=%MAXX% ^& python agent_loop.py 5 %PORT% %ALIAS% %TEMP%
echo       stop   : powershell -c "Stop-Process -Name llama-server -Force"
echo.

REM ---- optional: prefill the agent prefix in the background ----
if "%WARM%"=="1" (
  echo [WARM] waiting for server, then prefilling the frozen agent prefix ...
  start "warm" /min powershell -NoProfile -Command "for($i=0;$i -lt 180;$i++){try{if((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:%PORT%/health').StatusCode -eq 200){break}}catch{};Start-Sleep 1}; $env:AGENT_MAXTOK='%MAXX%'; python '%~dp0agent_loop.py' 0 %PORT% %ALIAS% --warm"
) else (
  echo [WARM] off -- set WARM=1 to prefill the agent_loop.py prefix at startup
)
echo.
echo [done] server stopped.
pause
endlocal
