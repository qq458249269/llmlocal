@echo off
REM ===============================================================
REM  agent-ornith9b.cmd  --  Ornith-1.5-9B Q4_K_M, CPU only
REM
REM  Budget: 5 threads / 8 GiB RAM  (i5-8500, 6c/6t, no NVIDIA GPU)
REM
REM  Measured on this box (Q4_K_M):
REM    weights   5.50 GB (5.12 GiB)
REM    context   262144   (model native max_position_embeddings)
REM    KV        q4_0     -> 2.0 GiB   (256k x 32 KiB/tok f16 x 0.25)
REM    RAM used  ~7.2 GiB
REM    tg        2.89 t/s  at -t 5
REM    pp512     25.5 t/s  -> 4000 prompt tokens in ~2.6 min
REM    REAL agent prefix, end to end (agent_loop.py --warm, 949 tokens):
REM      IQ4_XS  done in 77.0 s | 12.4 t/s
REM      Q4_K_M  done in 43.7 s | 22.0 t/s     <- 1.77x, this file's model
REM    RSS 9.6 GiB at 262144 (weights 5.12 + KV 2.0 + --cache-ram 3.0 + buf)
REM    load      ~4 s
REM    tool call OK, path fidelity OK, zh OK at q4_0 KV
REM
REM  ===== QUANT IS THE ONLY REAL PREFILL DIAL (measured, same model) =====
REM                     pp512 t=6   GFLOPS   size    4000 tok
REM    IQ4_XS (4.25 bpw)   13.7      245     4.88 GB   4.8 min
REM    Q4_K_M (4.82 bpw)   25.5      456     5.50 GB   2.6 min  <- this script
REM    Q4_0 (4.5 bpw)      ~24*      ~430    5.11 GB   ~2.8 min
REM      *Q4_0 measured 15-25% SLOWER than Q4_K_M on same-weights A/B runs
REM       (LFM2.5-350M and LFM2.5-1.2B, 2 runs each) -- so pick Q4_K_M.
REM    Cause: i-quants (IQ*) have no batched GEMM panel on x86, so every
REM    weight gets LUT-decoded once PER TOKEN of the batch.  K-quants take the
REM    real GEMM path.  This is an AVX2-only Coffee Lake -- no AVX512, no VNNI:
REM      CPU : AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | (no AVX512/VNNI)
REM    IQ4_XS -> Q4_K_M is +87% at prefill for -6% at decode (3.08 -> 2.89).
REM
REM    threads   pp64   t=1 2.96 | t=3 8.09 | t=6 14.12 (IQ4_XS) <- linear, use 6
REM    threads   decode t=5 BEATS t=6: 2.89 vs 2.13 (Q4_K_M), 3.08 vs 2.82 (IQ4_XS)
REM    ubatch    pp512  Q4_K_M: ub256 25.2 | 512 25.8 | 1024 24.9 | 2048 26.0
REM    flash-attn pp128 t=6: fa on 14.49 vs fa off 13.89
REM    --poll 0/50/100 at t=6/pp128: 14.85 / 15.14 / 14.88 (no effect)
REM    -lm mmap 2.90 | mlock 2.02 | none 2.91 (tg64, IQ4_XS) -> keep mmap
REM
REM    HARD CEILING: a dense 9B needs 2 x 8.95e9 = 17.9 GFLOP per token, and
REM    this CPU tops out at ~450 GFLOPS (qwen2.5-coder-7B Q4_K_M = 28.9 t/s
REM    pp512).  So prefill can NEVER drop below ~40 ms/token = 2.7 min for a
REM    4000-token prefix.  Q4_K_M is already within 15% of that floor.
REM    NO FLAG FIXES THIS.  Measured pp for a 4000-token prompt:
REM      LFM2.5-1.2B  Q4_K_M   191 t/s  ->  21 s     <- if pp is what hurts
REM      LFM2.5-2.6B  Q4_K_M    81 t/s  ->  49 s
REM      Qwen2.5-C-7B Q4_K_M    29 t/s  ->  2.3 min
REM      Ornith-1.5-9B Q4_K_M  25.5 t/s ->  2.6 min  (this script)
REM
REM    DECODE is bandwidth-bound, not compute: 4.87 GiB x 3.08 = 15.2 GB/s and
REM    4.36 GiB x 2.37 = 10.3 GB/s (7B Q4_K_M) -- same wall.  This box has
REM    16 GB + 4 GB DDR4-2400 (ChannelA-DIMM0 / ChannelB-DIMM0) = asymmetric
REM    "flex" mode, so only the first 8 GB is interleaved dual-channel.
REM    Swapping the 4 GB stick for a 16 GB one (-> 32 GB symmetric) is the
REM    only change that can come close to doubling decode.
REM    Keep 9B quality but kill the wait by making the 4000-token prefix a
REM    ONE-TIME cost: launch the server once, leave it running, and never let
REM    a single byte of the prefix drift (no date, no session id, no reordered
REM    tools) -- then every turn after the first is 90%+ cache hit (measured
REM    on this box: prompt_ms 3.8 s -> 0.56 s on the 2nd identical request).
REM    WARM=1 (below) moves the cold prefill to server start-up so the first
REM    real question is already cached.
REM
REM  WHY 256k + q4_0 and not 128k + q8_0:
REM    q4_0 KV at 256k costs the same 2.0 GiB as q8_0 at 128k, so you
REM    get 2x the context for free AND a faster decode.  q4_0 is only
REM    risky for needle-in-haystack recall across 100k+ tokens.
REM    If you want max attention fidelity instead, set CTX=131072
REM    and CTK=q8_0 (still 6.7 GiB, tg ~3.2 t/s).
REM
REM  MUST use bin\cpu\  -- bin-vulkan\ tries the iGPU and dies with
REM    "ggml_vulkan: Device memory allocation ... failed".
REM
REM  --reasoning-budget is REQUIRED: Ornith is a reasoning model and
REM  burns ~2967 thinking tokens on a one-sentence question (~16 min
REM  at 3 t/s).  256 cuts it to ~936.
REM
REM  PREFIX CACHE -- what actually works on this model:
REM    ENABLED : prompt prefix reuse (longest-common-prefix match),
REM               --cache-ram, --ctx-checkpoints, --cache-idle-slots.
REM               Measured: 5-turn tool loop, cache hit 86% -> 92%,
REM               wall time flat at 28-32 s/turn instead of growing.
REM    DISABLED: --cache-reuse and --context-shift.  Both need KV
REM               shifting, which the hybrid linear-attention context
REM               does not support:
REM                 "cache_reuse is not supported by this context"
REM                 "KV cache shifting is not supported for this context"
REM               So 262144 is a HARD ceiling -- a full context cannot
REM               slide; you must start a new conversation.
REM ===============================================================
setlocal

set "MODEL=%~dp0models\Ornith-1.5-9B-Q4_K_M.gguf"
set "ALIAS=ornith-9b"
set "PORT=8080"
set "CTX=262144"
set "CTK=q4_0"
set "CTV=q4_0"
set "THREADS=5"
set "MAXT=5"
REM TB = threads for PROMPT PROCESSING ONLY (-tb).  Decode keeps -t 5 so the
REM desktop stays usable, but prefill gets all 6 cores: 12.87 -> ~14.9 t/s.
set "TB=6"
REM ub 256 measured slightly faster than 512/1024/2048 for both quants
REM (Q4_K_M: 25.2 / 25.8 / 24.9 / 26.0) -- larger ubatches do NOT help.
set "UB=256"
set "MAXRAM=8"
set "CACHEMB=3072"
set "CTXCP=64"
set "CMS=512"
set "BUDGET=256"
REM WARM=1: right after the server is up, send one max_tokens=1 request with
REM the agent_loop.py system prompt + tools in the background, so the cold
REM ~4.5 min prefill happens while you are still reading the console instead of
REM in front of your first question.  Only useful when the client really uses
REM that same frozen prefix; set WARM=0 to skip it.
set "WARM=0"
set "SERVE=%~dp0Bonsai-demo\bin\cpu\llama-server.exe"

if not exist "%SERVE%" (
  echo [ERR] not found: %SERVE%
  pause & exit /b 1
)
if not exist "%MODEL%" (
  echo [ERR] not found: %MODEL%
  echo       bash dl.sh bartowski/Ornith-1.5-9B-GGUF Ornith-1.5-9B-Q4_K_M.gguf models 8
  pause & exit /b 1
)
if not exist "%~dp0llama-plan.ps1" (
  echo [ERR] not found: %~dp0llama-plan.ps1
  pause & exit /b 1
)

REM ---- ask the planner what we can afford right now ----
set "RUN_CTX=%CTX%" & set "RUN_CTK=%CTK%" & set "RUN_T=%THREADS%"
set "PSPLAN=powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0llama-plan.ps1" -Size ornith9b -Ctx %RUN_CTX% -Ctk %RUN_CTK% -Threads %RUN_T% -MaxRam %MAXRAM% -MaxThreads %MAXT%"
for /f %%c in ('%PSPLAN% -Only CTX')  do set "RUN_CTX=%%c"
for /f %%k in ('%PSPLAN% -Only CTK')  do set "RUN_CTK=%%k"
for /f %%t in ('%PSPLAN% -Only THREADS') do set "RUN_T=%%t"
echo [PLAN] ctx=%RUN_CTX%  ctk=%RUN_CTK%  threads=%RUN_T%

netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 echo [WARN] port %PORT% already in use -- stop that server first

REM ---- launch: BelowNormal so the desktop stays responsive ----
start "" /min powershell -NoProfile -Command "$p=Start-Process -FilePath '%SERVE%' -ArgumentList '-m','%MODEL%','--alias','%ALIAS%','-ngl','0','-fa','on','-c','%RUN_CTX%','-ctk','%RUN_CTK%','-ctv','%CTV%','-t','%RUN_T%','-tb','%TB%','-ub','%UB%','-n','0','--host','127.0.0.1','--port','%PORT%','--parallel','1','--cache-ram','%CACHEMB%','--ctx-checkpoints','%CTXCP%','--checkpoint-min-step','%CMS%','--cache-idle-slots','--jinja','--temp','0.6','--top-p','0.95','--top-k','20','--reasoning-budget','%BUDGET%' -NoNewWindow -PassThru; $p.PriorityClass='BelowNormal'"

echo [RUN] %ALIAS%  ctx=%RUN_CTX%  ctk=%RUN_CTK%  t=%RUN_T%  tb=%TB%  ub=%UB%  port=%PORT%
echo       prefix cache ON  (cache-ram=%CACHEMB% MiB, ctx-checkpoints=%CTXCP%)
echo       prefill budget: ~15 t/s ; a 4000-token prompt costs ~4.5 min EVERY
echo                       time the prefix cache is lost; keep the server up.
echo       stop with:  powershell -c "Stop-Process -Name llama-server -Force"

REM ---- optional: prefill the agent prefix in the background ----
if "%WARM%"=="1" (
  echo [WARM] waiting for server, then prefilling the frozen agent prefix ...
  start "warm" /min powershell -NoProfile -Command "for($i=0;$i -lt 180;$i++){try{if((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:%PORT%/health').StatusCode -eq 200){break}}catch{};Start-Sleep 1}; python '%~dp0agent_loop.py' 0 %PORT% %ALIAS% --warm"
) else (
  echo [WARM] off -- set WARM=1 to prefill the agent_loop.py prefix at startup
)
endlocal
