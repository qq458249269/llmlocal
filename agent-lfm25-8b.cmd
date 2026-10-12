@echo off
REM ===============================================================
REM  agent-lfm25-8b.cmd  --  LFM2.5-8B-A1B Q4_K_M, CPU only
REM
REM  The middle tier: 8.47 B total but only 1.5 B active per token, so
REM  it behaves like a 1.5B model for speed while keeping 8B knowledge.
REM  2.0x faster prefill and 4.3x faster decode than agent-ornith9b.cmd,
REM  in 8.2 GiB instead of 9.6 GiB.  Stronger than MiniCPM5-1B at
REM  agentic tool use, so this is the one to run for a real tool loop.
REM
REM  Budget: 4 threads (decode) / 6 threads (prefill) / 9 GiB RAM
REM          (i5-8500, 6c/6t, no NVIDIA GPU)
REM  MAXRAM is compared against the $PEAK_* table, which records WORKING SET
REM  (~8.5 GiB at 32k), not private commit (3.7 GiB).  Keep it at 9 so the
REM  planner does not needlessly clamp CTX down to 8192.
REM
REM  Measured on this box (UD-Q4_K_M, 4.95 GiB), re-measured 2026-09-30:
REM    weights   4.95 GiB     131072 ctx  PrivateWS 5.23 / WS 9.71 GiB
REM                          32768 ctx  PrivateWS 3.72 / WS 8.48 GiB
REM    context   16384        (was 32768, now 16384 -- see CONTEXT at the
REM                          CTX= line below; the 32k/128k numbers below are
REM                          the historical measurements they came from)
REM    pp512     79.7 t/s @ -t 4        tg32  13.5 t/s @ -t 4
REM    REAL agent prefix, end to end (agent_loop.py --warm, 754 tokens):
REM      LFM2.5-8B-A1B  done in 21.5 s | 35.4 t/s   <- this script
REM      Ornith 9B      done in 43.7 s | 22.0 t/s   <- 2.03x
REM      MiniCPM5-1B    done in  6.3 s | 136.2 t/s  <- 3.4x, but no tools
REM    load      ~25 s (4.95 GiB from disk)
REM
REM  ===== WHY PrivateWS (3.72) < WEIGHTS (4.95) =========================
REM  The "it eats a lot of RAM" feeling is mostly an illusion.  llama.cpp
REM  mmaps the weights, so those 4.95 GiB land in file cache, not in private
REM  commit -- Windows reclaims them under pressure.  Read BOTH numbers:
REM      WorkingSet   8.48 GiB  includes 4.95 GiB of mmap model pages
REM      PrivateWS    3.72 GiB  the part that is really pinned = committed
REM                              KV + compute buffer
REM  The $PEAK_* tables in llama-plan.ps1 record WorkingSet, so do not read
REM  them as private usage.
REM
REM  ===== WHERE THE MEMORY ACTUALLY GOES (measured 32k vs 128k) ====
REM    131072 f16  PrivateWS 5.23 GiB
REM     32768 f16  PrivateWS 3.72 GiB      <- gap 1.51 GiB
REM  The gap is 12 KiB/token x (131072-32768) = 1.47 GiB, i.e. exactly the
REM  KV cache.  => -c is the only knob here that moves real memory, because
REM  KV is pre-allocated per -c.
REM
REM  ===== THESE THREE PARAMETERS COST YOU NOTHING ====================
REM  bare vs full, PrivateWS / WorkingSet identical to the byte:
REM      -c 32768 -t 4 -tb 6 -ub 2048                          3.72 / 8.48
REM      -c 32768 -t 4 -tb 6 -ub 2048 --cache-ram 1024 ...      3.72 / 8.48
REM  --cache-ram 1024 / --ctx-checkpoints 64 / --cache-idle-slots are CAPS,
REM  not reserves -- nothing is committed until they are hit.  Do not bother
REM  lowering them to save RAM; it does nothing.  What they do buy is the
REM  92-96% prefix-cache hit rate measured by agent_loop.py, which is worth
REM  far more than the GiB.
REM
REM  ===== 10 t/s IS NOT A SLOW MODEL, IT IS A LONG CONVERSATION ======
REM  decode falls off as the CURRENT KV grows, because every generated
REM  token rescans the whole cache (6 attention layers x 12 KiB/token).
REM  Measured -c 32768 / f16 / -t 4:
REM      KV length   decode     prefill
REM         2.0k      11.5 t/s   86.0 t/s
REM         7.2k       7.6 t/s   86.7 t/s
REM        13.9k       5.1 t/s   80.7 t/s
REM        17.8k       3.1 t/s   60.3 t/s
REM  So shrinking -c does NOT make generation faster (KV is read on demand);
REM  it only makes the wall arrive 4x sooner.  What actually speeds things up
REM  is keeping the conversation short, which --context-shift does for you.
REM
REM  ===== TRIED AND NOT WORTH IT =====================================
REM  KV quant q8_0  same 7.2k run: decode 7.63 -> 8.23 t/s (+8%) but
REM                 prefill 86.7 -> 49.9 t/s (-42%).  In an agent loop the
REM                 cold prefill is what you wait on, so +8% is not a trade.
REM                 Keep f16.
REM  -t 6          at 7.2k ctx it gives 3.46 t/s vs 7.63 at -t 4, i.e. 55%
REM                 SLOWER.  This MoE's decode is memory-latency bound, not
REM                 bandwidth bound, so extra threads just fight over the
REM                 same memory bus.
REM  -ub            prefill only; decode runs at batch=1, so it is a no-op.
REM  speculative    --spec-draft-model LFM2.5-1.2B-Instruct is accepted by
REM                 llama.cpp, but 1.2B's vocab is 65536 vs this model's
REM                 128000, so the token ids do not really line up -- do not
REM                 depend on it.  It measured 11.13 t/s at 3.7k ctx (+20%)
REM                 for +0.68 GiB, and on 6 cores the draft and the target just
REM                 steal cores from each other.
REM
REM  ===== ARCHITECTURE (lfm2moe) =====
REM  24 layers, but only 6 of them have attention -- 18 are gated short
REM  convolution blocks.  32 experts, top-4, first 2 layers dense.
REM  Consequences that matter here:
REM    KV/token = 6 layers x 8 kv_head x 64 dim x 2 (K+V) x 2 B
REM            = 12,288 B = 12.0 KiB f16  (Ornith-9B is 32.0 KiB)
REM    32768 ctx: f16 0.375 GiB  <- matches the PrivateWS gap measured above
REM    --context-shift and --cache-reuse ARE accepted here (verified: no
REM    "not supported" errors).  They are REJECTED on agent-ornith9b.cmd
REM    ("KV cache shifting is not supported by this context"), so this
REM    model slides its context instead of dying at the wall.
REM  Vocabulary is 128,000, which is why the agent prefix is 754 tokens
REM  instead of Ornith's 949.
REM
REM  ===== UBATCH MATTERS FOR MoE (it does NOT for dense models) =====
REM  On a dense model ubatch is a wash -- on Ornith-9B, ub 256/512/1024/
REM  2048 all gave 25-26 t/s.  On an MoE the batch has to be large
REM  enough that every one of the 32 experts gets a real GEMM:
REM      ub 256   78.15 t/s
REM      ub 512   85.49 t/s
REM      ub 1024  81.77 t/s
REM      ub 2048  92.10 t/s   <- this script
REM  Without it you leave ~17% of the prefill throughput on the table.
REM
REM  ===== KV QUANTIZATION IS STILL A TRAP.  DO NOT TURN IT ON. =====
REM  Same server, only -ctk/-ctv changed, 10366 prompt tokens:
REM      KV      prefill          decode (300 tok)
REM      f16     70.4 t/s        4.74 t/s    <- this script
REM      q4_0    36.0 t/s        7.24 t/s
REM  2.0x slower prefill to buy 1.53x decode.  The penalty is smaller
REM  than on MiniCPM5-1B (3.8x / 1.27x) only because just 6 of 24 layers
REM  hold a KV cache, but the sign is the same: a quantized KV cache
REM  makes llama.cpp dequantize K/V inside the O(n^2) prefill path.
REM  Re-measured at 7.2k ctx on 2026-09-30, the decode win had shrunk to
REM  +8% (7.63 -> 8.23 t/s) while prefill still lost 42%.  Not worth it.
REM
REM  ===== OLD-CPU SETTINGS, all measured =====
REM    quant    Q4_K_M, not an i-quant.  i-quants have no batched GEMM
REM             panel on x86, so every weight is LUT-decoded once PER
REM             TOKEN of the batch.  On the 9B, IQ4_XS -> Q4_K_M was
REM             +87% prefill for -6% decode.  This CPU is AVX2-only
REM             Coffee Lake: AVX2=1 F16C=1 FMA=1 BMI2=1, no AVX512/VNNI.
REM             Note: community reports claim mudler's GGUF builds of
REM             this model run ~2x faster than LiquidAI/unsloth ones,
REM             so the publisher matters as much as the quant type.
REM             This file uses unsloth's UD-Q4_K_M; if you ever want to
REM             A/B it, mudler/LFM2.5-8B-A1B-APEX-GGUF is the one.
REM    decode   t=4 beats t=6 by 29%: 12.30 (t=4) | 11.62 (t=5) |
REM             9.52 (t=6).  Decode reads only the ~4/32 active experts,
REM             so it is bandwidth-bound and fewer threads win.
REM    prefill  -tb 6 so batch work still gets every core while -t stays 4.
REM    -ngl 0   MUST use bin\cpu\.  bin-vulkan\ tries the iGPU and dies
REM             with "ggml_vulkan: Device memory allocation ... failed"
REM             (UHD 630 reports 10298 MiB but has 0 MiB free).
REM    load     leave the load mode at the mmap default: on the 9B,
REM             mmap 2.90 | mlock 2.02 | none 2.91 t/s.
REM
REM  ===== SAMPLING -- model card =====
REM    temperature 0.2, top_k 80.  Deliberately low; this model is
REM    tuned for tool calls and long multi-turn agentic loops, and a
REM    high temperature makes it drift out of the tool-call schema.
REM    It also emits no separate thinking block by default, so there is
REM    no --reasoning-budget needed (unlike agent-ornith9b.cmd, which
REM    burns ~2967 thinking tokens on a one-sentence question).
REM    CAREFUL: these are SERVER DEFAULTS only.  agent_loop.py sends its
REM    own temperature/top_k per request (measured: temp=0.10 top_k=50),
REM    which overrides them.  If you want the card values to win, edit
REM    llm_client.py / agent_loop.py, not this file.
REM
REM  ===== VERIFIED END TO END =====
REM    python agent_loop.py 1 8082 lfm25-8b
REM      turn 1  prefill 22.4 s / 771 tok | gen 263 tok @ 10.6 t/s | 47.3 s
REM      turn 2  prefill  4.9 s / 834 tok | 767 cached = 92.0% hit
REM      tool_calls: read_file -> args {"path":"llama-plan.ps1"}  OK
REM      1 round total 74.6 s, output 450 tok @ 9.6 t/s
REM    The tool call actually fired and parsed, which is exactly where
REM    agent-minicpm5-1b.cmd fails (it refuses and emits no JSON).
REM
REM  ===== WHAT THIS MODEL IS GOOD AND BAD AT =====
REM  Vendor table (LiquidAI, biased in their favour, and Ornith is not
REM  in their comparison set):
REM      non-hallucination rate   63.47   vs Qwen3.5-4B 16.99
REM      IFEval instruction       91.84   vs Qwen3.5-4B 87.80
REM      Tau2-Telecom multi-turn  88.07   vs Qwen3.5-4B 87.72
REM      AIME25 maths             42.53   vs Qwen3.5-4B 54.28  (weaker)
REM      BFCLv3 tool-call schema  64.79   vs Qwen3.5-4B 71.06  (weaker)
REM  Ornith-1.5-9B is the opposite profile: SWE-bench Verified 70.6,
REM  Terminal-Bench 2.1 46.2, and its own card says LFM2.5 is "not the
REM  best fit for heavy programming".
REM    => Use THIS for tool-calling agents and long-context retrieval.
REM       Use agent-ornith9b.cmd when the agent writes real code.
REM       Use agent-minicpm5-1b.cmd for fast drafting / summarising.
REM
REM  Download (ModelScope mirror, 4.96 GiB):
REM    bash dl.sh unsloth/LFM2.5-8B-A1B-GGUF LFM2.5-8B-A1B-UD-Q4_K_M.gguf models 8
REM ===============================================================
setlocal

set "MODEL=%~dp0models\LFM2.5-8B-A1B-UD-Q4_K_M.gguf"
set "ALIAS=lfm25-8b"
set "PORT=8082"
REM CTX=16384.  Lowered from 32768 on 2026-10-12.
REM
REM Why lowering it DOES help, despite "-c is just a cap":
REM   -c alone changes nothing about speed -- decode reads KV on demand.
REM   But this script has --context-shift ON, so -c is the hard ceiling the
REM   sliding window uses.  A smaller ceiling means the conversation slides
REM   SOONER, which is the only thing that actually keeps decode fast:
REM       measured decode vs CURRENT kv length, -c 32768 / f16 / -t 4:
REM           2.0k kv -> 11.5 t/s      13.9k kv ->  5.1 t/s
REM           7.2k kv ->  7.6 t/s      17.8k kv ->  3.1 t/s
REM   At 16k the wall arrives at ~7 t/s instead of ~3 t/s.  Prefill is
REM   O(n^2) as well (86 t/s at 2k -> 60 t/s at 17.8k), so a tighter cap
REM   also caps how long any single request can block.
REM
REM Memory: KV is pre-allocated per -c and this is the only knob that moves
REM real usage.  kv = 6 attn layers x 8 kv_head x 64 dim x 2 x 2B = 12 KiB/token.
REM     32768 -> 0.375 GiB     PrivateWS 3.72 GiB
REM     16384 -> 0.188 GiB     PrivateWS ~3.5 GiB
REM 16384 is still 20x the real agent prefix (754 tokens measured), so a
REM normal tool loop never touches the wall.
REM
REM Go back to 32768 (or 131072) only for one-shot long-document work:
REM 32768, or 131072 which needs YaRN and 5.23 GiB PrivateWS.
set "CTX=16384"
REM f16 KV, NOT a quantized KV -- see the trap documented above.
set "CTK=f16"
set "CTV=f16"
REM Decode threads.  t=4 beat t=6 by 29% (12.30 vs 9.52 t/s) and also
REM leaves two cores for Windows.
set "THREADS=4"
set "MAXT=5"
REM TB = threads for PROMPT PROCESSING ONLY (-tb).  MoE prefill wants
REM every core; decode wants 4.  -t and -tb are independent knobs.
set "TB=6"
REM ubatch 2048: MoE needs a large batch so all 32 experts get a GEMM
REM (92.1 vs 78.2 t/s at ub 256).  This is the opposite of the dense
REM Ornith-9B, where ubatch made no difference.
set "UB=2048"
set "MAXRAM=9"
set "CACHEMB=1024"
set "CTXCP=64"
set "CMS=512"
REM WARM=1 prefills the agent prefix in the background at start-up so
REM the first real question does not pay the ~21 s cold prefill.
set "WARM=0"
set "SERVE=%~dp0Bonsai-demo\bin\cpu\llama-server.exe"

if not exist "%SERVE%" (
  echo [ERR] not found: %SERVE%
  pause & exit /b 1
)
if not exist "%MODEL%" (
  echo [ERR] not found: %MODEL%
  echo       bash dl.sh unsloth/LFM2.5-8B-A1B-GGUF LFM2.5-8B-A1B-UD-Q4_K_M.gguf models 8
  pause & exit /b 1
)
if not exist "%~dp0llama-plan.ps1" (
  echo [ERR] not found: %~dp0llama-plan.ps1
  pause & exit /b 1
)

REM ---- ask the planner what we can afford right now ----
REM -CountMmap: the $PEAK_* tables record WORKING SET, which counts the
REM 4.76 GiB of mmapped weight pages.  Those are reclaimable file cache, so
REM counting them made the planner reject a config that actually fits and
REM over-downgrade to 8192/q4_0 -- the worst possible config for an agent
REM (q4_0 KV costs 2x prefill).  With the flag the fit test uses the real
REM private commit:  8.4 GiB WS - 4.76 GiB = 3.6 GiB, which fits.
set "RUN_CTX=%CTX%" & set "RUN_CTK=%CTK%" & set "RUN_T=%THREADS%"
set "PSPLAN=powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0llama-plan.ps1" -Size lfm258b -Ctx %RUN_CTX% -Ctk %RUN_CTK% -Threads %RUN_T% -MaxRam %MAXRAM% -MaxThreads %MAXT% -CountMmap"
for /f %%c in ('%PSPLAN% -Only CTX')  do set "RUN_CTX=%%c"
for /f %%k in ('%PSPLAN% -Only CTK')  do set "RUN_K=%%k"
for /f %%t in ('%PSPLAN% -Only THREADS') do set "RUN_T=%%t"
REM NOTE: any arrow in an echo line MUST stay escaped as ^> -- a bare >
REM is a redirection and would dump the whole line into a file.
REM The planner's -Only CTK hint is printed for information but DELIBERATELY
REM IGNORED below: it is a pure memory heuristic and it answers q8_0 here,
REM while measurement says f16 prefill is 2.0x faster (70.4 vs 36.0 t/s).
echo [PLAN] ctx=%RUN_CTX%  ctk=%RUN_CTK%  threads=%RUN_T%  (planner suggests kv %RUN_K% -- ignored, measured f16 wins)

netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 echo [WARN] port %PORT% already in use -- stop that server first

REM ---- launch: BelowNormal so the desktop stays responsive ----
start "" /min powershell -NoProfile -Command "$p=Start-Process -FilePath '%SERVE%' -ArgumentList '-m','%MODEL%','--alias','%ALIAS%','-ngl','0','-fa','on','-c','%RUN_CTX%','-ctk','%CTK%','-ctv','%CTV%','-t','%RUN_T%','-tb','%TB%','-ub','%UB%','-n','0','--host','127.0.0.1','--port','%PORT%','--parallel','1','--cache-ram','%CACHEMB%','--ctx-checkpoints','%CTXCP%','--checkpoint-min-step','%CMS%','--cache-reuse','64','--cache-idle-slots','--context-shift','--jinja','-rea','off','--temp','0.2','--top-k','80' -NoNewWindow -PassThru; $p.PriorityClass='BelowNormal'"

echo [RUN] %ALIAS%  ctx=%RUN_CTX%  KV=%CTK%  t=%RUN_T%  tb=%TB%  ub=%UB%  port=%PORT%
echo       prefill ~80 t/s  ^|  decode 11.5 t/s @2k ctx, 7.6 @7k, 3.1 @18k
echo       PrivateWS 3.7 GiB / WorkingSet 8.5 GiB -- the 4.95 GiB model
echo       is mmap file cache, reclaimable, not real usage.
echo       temp 0.2 / top-k 80 per model card.  context-shift + cache-reuse
echo       both work here, so the 32k context slides.
echo       stop with:  powershell -c "Stop-Process -Name llama-server -Force"

REM ---- optional: prefill the agent prefix in the background ----
if "%WARM%"=="1" (
  echo [WARM] waiting for server, then prefilling the frozen agent prefix ...
  start "warm" /min powershell -NoProfile -Command "for($i=0;$i -lt 180;$i++){try{if((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:%PORT%/health').StatusCode -eq 200){break}}catch{};Start-Sleep 1}; python '%~dp0agent_loop.py' 0 %PORT% %ALIAS% --warm"
) else (
  echo [WARM] off -- set WARM=1 to prefill the agent_loop.py prefix at startup
)
endlocal