@echo off
REM ===============================================================
REM  agent-sakthai.cmd  --  SakThai Context 0.5B Q4_K_M, CPU only
REM
REM  PURE ASCII ON PURPOSE. cmd.exe decodes UTF-8 .cmd on codepage 936
REM  and mis-pairs non-ASCII bytes into pipe characters, which then get
REM  EXECUTED as commands.  Every other .cmd in this repo is ASCII too.
REM
REM  WHAT THIS IS:  Qwen2.5-0.5B-Instruct + 2309 rows of tool-calling
REM  SFT.  397 MB weights, decode 34-38 t/s -- the FASTEST agent base on
REM  this box (3.4x LFM2.5-2.6B, 1.5x LFM2.5-1.2B) and by far the
REM  weakest one.
REM
REM  MEASURED 2026-10-12 (i5-8500 / DDR4-2400):
REM    decode  0.6k:34.2  2.0k:32.1  7.5k:16.3  14.8k:9.7 t/s
REM    prefill  0.6k:206  2.0k:199  7.5k:142  14.8k:90 t/s
REM    llama-bench tg256 ceiling = 38.3 t/s at -t 4 (t=6 -> 29.5)
REM    KV 12 KiB/token -- all 24 layers store KV (plain qwen2 attention,
REM    no conv/SSM layers), L_half = 32k, so ctx is cheap up to ~8k.
REM
REM  *** TOOL CALLING: USE PORT 8085 (sakfix.py PROXY), NOT 8084 ***
REM  This model wraps its tool calls in a DIFFERENT tag every run
REM  (measured, 8 runs each): <tools>{...}</tools> / tools[name:...]
REM  / FUNCTION:x PARAMETERS:{...} / <tool>x</tool>{...} -- llama.cpp
REM  parses NONE of them -> tool_calls=null.  Only the plain
REM  <tool_call> block is understood, and even the strict
REM  sak_template.jinja does not make the model emit it reliably (0/8).
REM  So the fix is CLIENT SIDE: sakfix.py normalizes all those
REM  shapes into OpenAI tool_calls, cuts the self-Q&A rambling, and
REM  repairs historical tool_calls whose arguments got truncated
REM  (llama.cpp answers HTTP 500 for those, and no server flag
REM  avoids it).  See README 4.16.
REM
REM  TOOL CALLING (measured 2026-10-12, through the 8085 proxy):
REM  sak_oneshot.py 4/4 (3 correct calls + the "should NOT call"
REM  case); agent_loop.py -> 5/5 rounds, 4/4 tool calls, 30.1 t/s
REM  average, prefix cache 91.7%.  It still invents POSIX paths
REM  (/home/user/notes.txt), so keep it read-only / QA -- but the
REM  "tool_calls is always null" problem is gone.
REM
REM  IT DOES NOT STOP: finish_reason is nearly always "length" and it
REM  hallucinates further user/assistant turns.  The client MUST send
REM  stop=["<|im_end|>","\nuser","\nsystem"] and cap max_tokens.
REM  The proxy strips the extra turns anyway, but capping tokens still
REM  saves 10-20 s per turn.
REM
REM  ENDPOINT : http://127.0.0.1:8085/v1  (sakfix proxy -- USE THIS)
REM             http://127.0.0.1:8084/v1  (raw llama-server, unfixed)
REM  MODEL    : sakthai-0.5b
REM  KEEP THIS WINDOW OPEN. Closing it stops the server.
REM ===============================================================
setlocal EnableDelayedExpansion
chcp 65001 >nul
title SakThai Context 0.5B API - 127.0.0.1:8084

REM -------------------------- knobs --------------------------
set "MODEL=%~dp0models\sakthai-0.5b-Q4_K_M.gguf"
set "TPL=%~dp0sak_template.jinja"
set "ALIAS=sakthai-0.5b"
REM 8080 LFM2.5-2.6B / 8080 LFM2.5-1.2B / 8082 8B-A1B / 8084 this script
set "PORT=8084"
REM 8085 = sakfix.py 归一化代理的监听端口（客户端一律指这个）
set "FIXPORT=8085"
REM 8192 is not arbitrary: decode still 16.3 t/s at 7.5k but drops to
REM 9.7 at 14.8k (KV is 3x dearer here than on LFM2.5-2.6B -- 24/24
REM layers vs 8/30).  Native window is 32768 if you need long docs.
set "CTX=8192"
REM f16 KV: only 0.12 GiB at 8k, quantizing saves nothing real and
REM slows the cold prefill the agent waits on every turn
set "CTK=f16"
set "CTV=f16"
REM t=4 fastest, same story as 1.2B/2.6B: t=6 measured 29.5 t/s
set "THREADS=4"
set "TB=4"
set "UB=512"
set "NPAR=1"
REM 0.40 GB weights + 0.12 GB KV at 8k -> under 1 GiB working set
set "CACHEMB=1024"
set "CTXCP=32"
set "CMS=256"
set "CPCREUSE=64"
set "MAXRAM=6"
REM Sampling: Qwen2.5 family defaults, but measured down at 0.1.
REM temp 0.7 -> 2-3 duplicated tool_calls, temp 0.3 -> rambles when no
REM tool fits.  A 0.5B has no room for creativity.
set "TEMP=0.1"
set "TOPK=20"
set "TOPP=0.8"
set "REPEAT=1.05"
REM 0.5B never stops on its own; keep the budget small
set "MAXX=200"
set "SERVE=%~dp0Bonsai-demo\bin\cpu\llama-server.exe"
REM -----------------------------------------------------------

echo ================================================================
echo  SakThai Context 0.5B  --  fastest base here, read-only / QA only
echo  model %MODEL%
echo  port  http://127.0.0.1:%PORT%/v1    (alias %ALIAS%)
echo ================================================================

if not exist "%SERVE%" (
  echo [ERR] not found: %SERVE%
  pause & exit /b 1
)
if not exist "%MODEL%" (
  echo [ERR] not found: %MODEL%
  echo       bash dl.sh Nanthasit/sakthai-context-0.5b-merged sakthai-0.5b-Q4_K_M.gguf models
  pause & exit /b 1
)
if not exist "%TPL%" (
  echo [ERR] not found: %TPL%
  echo       without it tool-calling is dead, see header
  pause & exit /b 1
)

netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 echo [WARN] port %PORT% already in use -- stop that server first
netstat -ano | findstr /R /C:":%FIXPORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 echo [WARN] port %FIXPORT% already in use -- stop the old sakfix first

REM ---- launch: detached, BelowNormal so the desktop stays smooth ----
REM --jinja is mandatory (parses the tool_call block).  --chat-template-file
REM takes a FILE PATH; --chat-template takes an INLINE STRING -- passing
REM the filename to --chat-template makes the model ramble release notes
REM about "sak_template 2.0".  Verified both ways.
start "" /min powershell -NoProfile -Command "$p=Start-Process -FilePath '%SERVE%' -ArgumentList '-m','%MODEL%','--alias','%ALIAS%','-ngl','0','-fa','on','-c','%CTX%','-ctk','%CTK%','-ctv','%CTV%','-t','%THREADS%','-tb','%TB%','-ub','%UB%','-n','0','--host','127.0.0.1','--port','%PORT%','--parallel','%NPAR%','--cache-ram','%CACHEMB%','--ctx-checkpoints','%CTXCP%','--checkpoint-min-step','%CMS%','--cache-reuse','%CPCREUSE%','--cache-idle-slots','--context-shift','--jinja','--chat-template-file','%TPL%','-rea','off','--temp','%TEMP%','--top-k','%TOPK%','--top-p','%TOPP%','--repeat-penalty','%REPEAT%' -NoNewWindow -PassThru; $p.PriorityClass='BelowNormal'

REM ---- sakfix 归一化代理：0.5B 的 tool_call 包装每次都不一样，只能客户端归一化 ----
start "" /min powershell -NoProfile -Command "Start-Process -FilePath 'python' -ArgumentList '%~dp0sakfix.py','%PORT%','%FIXPORT%' -WorkingDirectory '%~dp0' -NoNewWindow"
"

echo.
echo [RUN] %ALIAS%  ctx=%CTX%  KV=%CTK%  t=%THREADS%  tb=%TB%  ub=%UB%  raw=%PORT%  proxy=%FIXPORT%
echo       decode ~34 t/s (short) / ~16 t/s at 8k / ~10 t/s at 16k
echo       prefill ~200 t/s (short) / ~142 t/s at 8k
echo       memory ~0.55 GiB.  sakfix proxy is doing the tool-call normalization
echo       client MUST send stop=[im_end, \nuser, \nsystem]
echo       keep max_tokens ^<= %MAXX%
echo.
echo       ENDPOINT http://127.0.0.1:%FIXPORT%/v1  <- use this (sakfix proxy)
echo       unit   : cd /d %~dp0 ^& python sak_unit.py
echo       loop   : cd /d %~dp0 ^& python agent_loop.py 5 %FIXPORT% %ALIAS% %TEMP%
echo       proxy  : cd /d %~dp0 ^& python sakfix.py %PORT% %FIXPORT%  (manual restart)
echo       raw    : cd /d %~dp0 ^& python tool_test.py %PORT% %ALIAS% %TEMP%  (unfixed, expect null)
echo       qa     : cd /d %~dp0 ^& python sak_qa.py %PORT% %ALIAS% %TEMP%  (no tools: proxy passes through)
echo       oneshot: cd /d %~dp0 ^& python sak_oneshot.py %FIXPORT% %ALIAS% %TEMP%  (4/4 expected)
echo       curve  : cd /d %~dp0 ^& python bench_ctx2.py %PORT% %ALIAS% "512,2048,8192,16384"
echo       stop   : powershell -c "Stop-Process -Name llama-server -Force"
echo.
echo [done] server stopped (close this window to also drop the sakfix proxy).
pause
endlocal