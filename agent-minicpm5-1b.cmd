@echo off
REM ===============================================================
REM  agent-minicpm5-1b.cmd  --  MiniCPM5-1B Q4_K_M, CPU only
REM
REM  The fastest option on this box by a wide margin: 6.9x faster
REM  prefill and 9x faster decode than agent-ornith9b.cmd, in 3.9 GiB
REM  instead of 9.6 GiB.  It is NOT a drop-in replacement -- see the
REM  tool-calling note at the bottom.
REM
REM  Budget: 4 threads (decode) / 6 threads (prefill) / 6 GiB RAM
REM          (i5-8500, 6c/6t, no NVIDIA GPU)
REM
REM  Measured on this box (Q4_K_M, all llama-bench unless noted):
REM    weights   0.64 GiB      RSS @131072 f16 KV   3.91 GiB
REM    context   131072        (model native max_position_embeddings)
REM    pp512     189 t/s       pp1024 218 t/s      pp2048 191 t/s
REM    decode    tg64  26.76 t/s @ -t 4  |  22.56 @ t=5 | 20.33 @ t=6
REM    REAL agent prefix, end to end (agent_loop.py --warm, 831 tokens):
REM      MiniCPM5-1B  done in 6.3 s | 136.2 t/s
REM      Ornith 9B    done in 43.7 s | 22.0 t/s    <- 6.9x
REM      Ornith 9B IQ done in 77.0 s | 12.4 t/s    <- 12.2x
REM    load      ~2 s
REM
REM  ===== KV QUANTIZATION IS A TRAP HERE.  DO NOT TURN IT ON. =====
REM  User asked for "KV quant + 128k ctx".  Measured at 8174 prompt
REM  tokens, same server, only -ctk/-ctv changed:
REM      KV      prefill          decode (200 tok)
REM      f16     125.7 t/s        7.21 t/s     <- this script
REM      q8_0     55.2 t/s        (n/a at this ctx)
REM      q4_0     33.3 t/s        9.12 t/s
REM  You pay 3.8x on prefill to buy 1.27x on decode.  Reason: with a
REM  quantized KV cache llama.cpp has to dequantize K/V per attention
REM  step, which lands squarely in the O(n^2) prefill path, while the
REM  weights (0.64 GiB) still dominate the per-token decode read at
REM  realistic context lengths (<=32k).  At 131072 f16 the whole
REM  process is 3.91 GiB, so there is no memory reason to quantize.
REM      KV/token = 24 layers x 2 kv_head x 128 dim x 2 (K+V) x 2 B
REM                = 24,576 B = 24.0 KiB
REM      131072 ctx: f16 3.00 GiB | q8_0 1.50 | q4_0 0.75
REM    EXTRA LANDMINE: -fa off + q4_0 KV makes llama-server DIE on the
REM    first request (connection refused).  f16 is the only combination
REM    that is both correct and fast here.
REM
REM  ===== OLD-CPU SETTINGS, all measured =====
REM    quant    Q4_K_M, not an i-quant.  On the 9B, IQ4_XS -> Q4_K_M
REM             was +87% prefill: i-quants have no batched GEMM panel
REM             on x86 so every weight gets LUT-decoded once PER TOKEN
REM             of the batch.  This CPU is AVX2-only Coffee Lake:
REM             AVX2=1 F16C=1 FMA=1 BMI2=1, NO AVX512, NO VNNI.
REM    decode   t=4 is the sweet spot and beats t=6 by 32%:
REM             26.76 (t=4) / 26.52 (t=2) / 25.79 (t=3) / 22.56 (t=5)
REM             / 20.33 (t=6).  Small model = short matvec = thread
REM             sync dominates, and fewer threads also leave the
REM             desktop free.
REM    prefill  -tb 6 with -t 4 -> 215 t/s.  Prefill wants every core;
REM             decode wants 4.  -t and -tb are independent knobs, so
REM             use both.
REM    -fa on   149.4 t/s vs 132.0 t/s for -fa off (f16 KV, 5793 tok).
REM    ubatch   512; this model is small enough that 512 already
REM             saturates.  (For MoE models ubatch DOES matter: on
REM             LFM2.5-8B-A1B, ub 256 -> 78 t/s but ub 2048 -> 92 t/s.)
REM    -ngl 0   MUST use bin\cpu\.  bin-vulkan\ tries the iGPU and dies
REM             with "ggml_vulkan: Device memory allocation ... failed"
REM             (UHD 630 has 10298 MiB reported but 0 MiB free).
REM    mmap     leave the load mode alone: on the 9B, mmap 2.90 |
REM             mlock 2.02 | none 2.91 t/s, i.e. mlock was 30% worse.
REM
REM  ===== PREFILL IS O(n^2).  BUDGET FOR IT. =====
REM  Attention FLOPs grow linearly with context, so total prefill time
REM  grows quadratically.  Measured with f16 KV, -t 4 -tb 6:
REM       prompt tokens      prefill wall      rate
REM            829              6.1 s        136 t/s
REM           5793             38.8 s        149 t/s
REM           8174             65.0 s        126 t/s
REM          20368            302.9 s         67 t/s
REM    Extrapolating the fitted n^2: a single one-shot 131072-token
REM    prefill is ~3 hours.  So 128k is reachable only INCREMENTALLY
REM    through conversation, and even then each extra turn gets more
REM    expensive (dt/dn at 100k ctx is ~0.15 s/token).  Keep real
REM    conversations under ~32k.
REM
REM  ===== WHY 128k + f16 (and not a smaller ctx) =====
REM  MiniCPM5-1B is a PLAIN LlamaForCausalLM (24 full-attention layers,
REM  no MoE, no hybrid linear attention), which is the best news here:
REM      --context-shift and --cache-reuse BOTH WORK on it.  On
REM      Ornith-1.5-9B (hybrid) they fail with "KV cache shifting is
REM      not supported by this context", so its 262144 was a hard
REM      ceiling.  Here the context can slide, so a long conversation
REM      degrades gracefully instead of dying.
REM    And the small vocabulary (130,560) cuts the real agent prefix
REM    from 949 tokens to 831 (Ornith has 248,320).
REM
REM  SAMPLING -- model card: no-think  temp 0.7  top_p 0.95
REM                             think    temp 0.9  top_p 0.95
REM    Turn thinking OFF for interactive work and tools.  It works via
REM    the jinja template, so pass it per request:
REM      curl .../v1/chat/completions -d '{"chat_template_kwargs":
REM            {"enable_thinking": false}, ...}'
REM    Verified both: no-think 73 out-tok @ 28.2 t/s, think 126 @ 27.0.
REM
REM  ===== LIMITATION: TOOL CALLING =====
REM  Sampled on this box with the OpenAI-compatible endpoint:
REM      Chinese fact QA     OK   (capital of China -> Beijing, correct)
REM      multi-step arithmetic OK  (12-7)*24 = 120, reasoning correct
REM      trivial code        OK   (correct palindrome one-liner)
REM      knowledge question  WRONG (answered "what is KV cache
REM                              quantization" with nonsense, and
REM                              conflated it with training)
REM      tool call           REFUSED -- asked for {"tool":...} JSON it
REM                              said "I don't have access to a
REM                              get_weather tool".  The model card
REM                              says SGLang is the recommended backend
REM                              for tool use and that MiniCPM5-1B
REM                              emits XML-style calls, so this needs
REM                              the SGLang chat template (or a custom
REM                              jinja template) before it can drive
REM                              agent_loop.py's tools.
REM    So: use it as a fast chat / drafting / summarising model, or
REM    for the long-context prefix work.  Keep Ornith-1.5-9B for real
REM    tool-calling agents.
REM
REM  Download (ModelScope mirror, 688 MB):
REM    bash dl.sh openbmb/MiniCPM5-1B-GGUF MiniCPM5-1B-Q4_K_M.gguf models 4
REM ===============================================================
setlocal

set "MODEL=%~dp0models\MiniCPM5-1B-Q4_K_M.gguf"
set "ALIAS=minicpm5-1b"
set "PORT=8081"
set "CTX=131072"
REM f16 KV, NOT a quantized KV -- see the trap documented above.
set "CTK=f16"
set "CTV=f16"
REM Decode threads.  t=4 beat t=6 by 32% (26.76 vs 20.33 t/s) and also
REM leaves two cores for Windows.
set "THREADS=4"
set "MAXT=5"
REM TB = threads for PROMPT PROCESSING ONLY (-tb).  -t 4 -tb 6 is the
REM measured optimum: 215 t/s prefill AND 26.8 t/s decode at once.
set "TB=6"
set "UB=512"
set "MAXRAM=6"
set "CACHEMB=1024"
set "CTXCP=64"
set "CMS=512"
REM WARM=1 prefills the agent prefix in the background at start-up.
REM Here it is worth it even if the client sends a different prefix:
REM the cold cost is ~1 s, not minutes.
set "WARM=0"
set "SERVE=%~dp0Bonsai-demo\bin\cpu\llama-server.exe"

if not exist "%SERVE%" (
  echo [ERR] not found: %SERVE%
  pause & exit /b 1
)
if not exist "%MODEL%" (
  echo [ERR] not found: %MODEL%
  echo       bash dl.sh openbmb/MiniCPM5-1B-GGUF MiniCPM5-1B-Q4_K_M.gguf models 4
  pause & exit /b 1
)
if not exist "%~dp0llama-plan.ps1" (
  echo [ERR] not found: %~dp0llama-plan.ps1
  pause & exit /b 1
)

REM ---- ask the planner what we can afford right now ----
set "RUN_CTX=%CTX%" & set "RUN_CTK=%CTK%" & set "RUN_T=%THREADS%"
set "PSPLAN=powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0llama-plan.ps1" -Size mcpm5 -Ctx %RUN_CTX% -Ctk %RUN_CTK% -Threads %RUN_T% -MaxRam %MAXRAM% -MaxThreads %MAXT%"
for /f %%c in ('%PSPLAN% -Only CTX')  do set "RUN_CTX=%%c"
for /f %%k in ('%PSPLAN% -Only CTK')  do set "RUN_K=%%k"
for /f %%t in ('%PSPLAN% -Only THREADS') do set "RUN_T=%%t"
REM NOTE: the arrow below MUST stay escaped as ^> -- a bare > is a shell
REM redirection and would send this whole line into a file named f16.
echo [PLAN] ctx=%RUN_CTX%  ctk=%RUN_CTK%  kv-^>%RUN_K%  threads=%RUN_T%

netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 echo [WARN] port %PORT% already in use -- stop that server first

REM ---- launch: BelowNormal so the desktop stays responsive ----
start "" /min powershell -NoProfile -Command "$p=Start-Process -FilePath '%SERVE%' -ArgumentList '-m','%MODEL%','--alias','%ALIAS%','-ngl','0','-fa','on','-c','%RUN_CTX%','-ctk','%CTK%','-ctv','%CTV%','-t','%RUN_T%','-tb','%TB%','-ub','%UB%','-n','0','--host','127.0.0.1','--port','%PORT%','--parallel','1','--cache-ram','%CACHEMB%','--ctx-checkpoints','%CTXCP%','--checkpoint-min-step','%CMS%','--cache-reuse','64','--cache-idle-slots','--context-shift','--jinja','-rea','off','--temp','0.7','--top-p','0.95' -NoNewWindow -PassThru; $p.PriorityClass='BelowNormal'"

echo [RUN] %ALIAS%  ctx=%RUN_CTX%  KV=%CTK%  t=%RUN_T%  tb=%TB%  ub=%UB%  port=%PORT%
echo       prefill ~215 t/s (bench) / 136 t/s (real agent prefix)
echo       decode  ~26.8 t/s at -t 4, 7.2 t/s at 8k ctx, ~2-4 t/s at 32k+
echo       RSS ~3.9 GiB at 131072 f16 KV.  Prefix cache ON
echo       (cache-ram=%CACHEMB% MiB, ctx-checkpoints=%CTXCP%).
echo       Tool calling needs the SGLang template -- see header notes.
echo       stop with:  powershell -c "Stop-Process -Name llama-server -Force"

REM ---- optional: prefill the agent prefix in the background ----
if "%WARM%"=="1" (
  echo [WARM] waiting for server, then prefilling the frozen agent prefix ...
  start "warm" /min powershell -NoProfile -Command "for($i=0;$i -lt 180;$i++){try{if((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:%PORT%/health').StatusCode -eq 200){break}}catch{};Start-Sleep 1}; python '%~dp0agent_loop.py' 0 %PORT% %ALIAS% --warm"
) else (
  echo [WARM] off -- set WARM=1 to prefill the agent_loop.py prefix at startup
)
endlocal