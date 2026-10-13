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
REM f16 KV, back to what the upstream engine actually wants.  Measured on
REM this box, llama-bench -t 4 -p 2048, same 2.6B:  f16 pp 67.2 /
REM q8_0 pp 55.2 / q4_0 worse still.  q8_0 only won under the ik_llama.cpp
REM fork (its IQK FlashAttention reads quantized KV straight out of the
REM attention kernel); that fork is gone, so q8_0 here would just be a
REM 18% prefill tax.  At -c 16384 f16 KV is ~0.25 GiB -- RAM is not the
REM reason to quantize, so do not.
set "CTK=f16"
set "CTV=f16"
REM -t 4.  Re-measured on fork b10743 (llama-bench -p 1024 -n 64 -r 2, 2.6B
REM Q4_K_M, this box, 2026-10-20):
REM       threads    4       5       6
REM       decode   9.50   10.12    6.82   t/s
REM       prefill 71.10   83.68   87.64   t/s
REM -t 5 is still the fastest, so -t 4 is a DELIBERATE give-up of ~6% decode
REM and ~15% prefill to leave 2 of 6 cores to Windows and the desktop.
REM This is a wall-clock knob, not a scaling one -- put it back to 5 if the
REM desktop being responsive stops mattering.
REM MAXT MUST track THREADS: it is the planner's ceiling, so leaving MAXT=5
REM lets the planner quietly raise the server back to 5 threads when RAM
REM headroom allows, which defeats the point of asking for 4.
REM There is NO "BLAS acceleration" switch in this build and adding one
REM cannot help: bin\cpu has no ggml-blas.dll / OpenBLAS / MKL (libomp140
REM is OpenMP, not BLAS), and the only "blas" strings in the DLLs are a
REM dead name in the dynamic-backend registry, one error message, and a
REM benchmark enum.  Decode streams 1.55 GiB of weights per token, so it is
REM memory-bus-bound and an arithmetic backend is irrelevant; prefill runs
REM ggml's fused dequant+dot kernels for Q4_K, which no BLAS can express.
REM The real CPU wins here are already on: -fa 1, --cache-ram 2048 and
REM --ctx-checkpoints 32 (82.6 s cold -> 0.67 s warm on the agent prefix).
set "THREADS=4"
set "MAXT=4"
REM -tb 4 too: -tb is an INDEPENDENT knob from -t, so leaving it at 5 would
REM still pin 5 cores during every prefill and undo "4 cores".  Prefill is
REM the half that actually wants cores (71 t/s at 4 vs 88 at 6), so this is
REM the bigger half of the trade.  (llama-bench does NOT accept -tb, so the
REM 71.10 pp figure above is a -t 4 run and is the closest reproducible
REM proxy for what -tb 4 will do here.)
set "TB=4"
REM ubatch 512 measured faster than 2048 on this CPU.  Note this is
REM the OPPOSITE of the MoE 8B, which needs ub 2048 (92 vs 78 t/s) so
REM that all 32 experts get a real GEMM.  Dense model -> wash.
set "UB=512"
REM Working set at 16384/f16 is ~2.1 GiB (planner lfm25 table), so a 4 GiB
REM cap keeps ~1.9 GiB of slack.  4 is picked on purpose: this box is
REM ASYMMETRIC dual channel -- ChannelA-DIMM0 = 16 GB, ChannelB-DIMM0 = 4 GB
REM (both DDR4-2400, confirmed via Win32_PhysicalMemory).  The controller
REM interleaves both channels starting at address 0, so only roughly the
REM first 4 GiB of process address space gets full dual-channel bandwidth;
REM past that the tail is served by the 16 GB stick alone.
REM ponytail: that mechanism is textbook, NOT measured on this box -- and
REM today it costs nothing anyway, because a 2.6B at 16384/f16 only needs
REM 2.1 GiB and therefore already sits inside the interleaved window.
REM It is a guard rail, not a speed knob, and it starts to bite at 4B and
REM up (agent-qwen35-4b.cmd peaks at 3.8 GiB, right on the edge;
REM agent-qwen7b.cmd peaks at 6.3 GiB, already past it).
REM To measure instead of trusting: sweep tg96 at -t 4 while padding the
REM working set past 4 GiB and look for the bandwidth cliff.
REM Real fix for the asymmetry, not a script change: add a 4 GB stick to
REM ChannelA so the pair becomes 8+8 matched.
set "MAXRAM=4"
set "CACHEMB=2048"      rem --cache-ram MiB. This is the single biggest
REM                       CPU win: a 2275-token system prompt costs 82.6 s
REM                       cold and 0.67 s warm (99.2% reused).  This is a
REM                       CAP, not a reservation -- it commits nothing
REM                       until hit, so do NOT lower it to save RAM.
set "CTXCP=32"          rem --ctx-checkpoints per slot
REM --ctx-checkpoints-interval does NOT exist on the upstream engine --
REM ik_llama.cpp had it, this build does not, and passing it makes
REM llama-server exit with "invalid argument" before it loads anything.
REM Upstream's fixed interval (512) is coarse for an agent: a 754-token
REM prefix can never build a checkpoint.  --cache-ram still carries the
REM heavy lifting, so leave it alone unless a build with the interval
REM knob comes back.
REM parallel slots; more than 1 splits CTX between them.
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
REM Upstream llama.cpp (Bonsai-demo/bin/cpu).  This is the llamafile build:
REM one ggml-cpu-<microarch>.dll per ISA, runtime-dispatched, so it picks
REM the AVX2+FMA+F16C kernels for this i5-8500 without any local build step.
REM The ik_llama.cpp fork was dropped on 2026-10-13 -- its prefill win was
REM real (88 vs 65 t/s on llama-bench) but invisible in the agent, and its
REM specialized LFM2.5 parser defaulted parallel_tool_calls to FALSE, which
-- mangled every two-call turn.  Upstream derives the same flag from the
REM chat template (llama-cpp-src server-common.cpp:1297), so the original
REM engine does not have that bug.  See README 4.17.
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
REM --context-shift was dropped too: LFM2.5 is a recurrent/hybrid model and
REM llama-server prints "ctx_shift is not supported by recurrent model, it
REM will be disabled" every launch.  -c is therefore the real window here,
REM not a sliding-window ceiling.
REM -lm none: pin the parameters in ordinary private memory instead of
REM leaving them file-backed.  With the default mmap the 1.55 GiB of weights
REM land in the file cache, which Windows is free to reclaim under
REM pressure -- that is exactly the "works fine, then stalls" case, and it
REM is also why llama-plan.ps1 needs -CountMmap to tell the truth.
REM -lm none makes them a real anonymous commit: never evicted as cache,
REM never compressed, no read-back path.
REM Measured on this box, fork b10743, llama-bench -t 4 -p 0 -n 96 -r 3:
REM     mmap 9.53 +- 0.11   mlock 9.81 +- 0.43   none 9.95 +- 0.28  t/s
REM none beats mmap by ~4.4% AND is the semantics we actually want.
REM none and mlock are inside each other's noise, so pick on behaviour:
REM none = pinned commit, still swappable under real pressure;
REM mmap+mlock = VirtualLock, never paged out, but the server can fail to
REM start if the lock cannot be satisfied.  For an agent server that must
REM not stall mid-turn, take none.
REM Cost: load is slower (full copy, not lazy paging) and the weights are no
REM longer reclaimable, which is the trade MAXRAM=4 above now has to cover.
set "ARGS=-m %MODEL% --alias %ALIAS% -ngl 0 -fa 1 -lm none -c %RUN_CTX% -ctk %RUN_K% -ctv %RUN_K% -t %RUN_T% -tb %TB% -ub %UB% -n 0 --host 127.0.0.1 --port %PORT% --parallel %NPAR% --cache-ram %CACHEMB% --ctx-checkpoints %CTXCP% --jinja -rea off --temp %TEMP% --top-k %TOPK% --repeat-penalty %REPEAT%"

echo.
echo [RUN] %ALIAS%  ctx=%RUN_CTX%  KV=%RUN_K%  t=%RUN_T%  tb=%TB%  ub=%UB%  port=%PORT%
echo       decode  ~9.5 t/s, prefill ~71 t/s (llama-bench -t 4, fork b10743)
echo       memory  ~2.1 GiB at 16384 f16 KV, cap 4 GiB, -lm none (pinned).  prefix cache ON
echo       (cache-ram=%CACHEMB% MiB, ctx-checkpoints=%CTXCP%)
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
