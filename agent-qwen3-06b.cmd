@echo off
rem ====================================================================
rem  Qwen3-0.6B-tools Q8_0 - tool-calling agent API (CPU only)
rem
rem  WHAT:  Qwen3-0.6B (596M) + LoRA SFT for OpenAI-style function calling.
rem         Emits the standard <tool_call> JSON block, so llama.cpp
rem         parses tool_calls natively -- NO sakfix.py proxy needed
rem         (unlike sakthai-0.5b, which wraps calls in random tags).
rem         tool_test.py: 3/3 (both tool cases correct + no false call).
rem
rem  PI: NOT USABLE.  The provider "qwen" (8086) was REMOVED from
rem  ~/.pi/agent/models.json on2026-10-13.  With pi's 1659-token agent
rem  system prompt this model emits broken prose and ZERO tool calls;
rem  with maxTokens 256 it dies at stopReason=length every turn.  Keep
rem  it for tool_test.py / short-prompt QA only.  Agent work goes to
rem  agent-lfm12-fast.cmd on 8080.  Full numbers in README 4.18.
rem
rem  CEILING: it is still a 0.6B.
rem  one can ramble INSIDE the arguments and still parse ("/etc/hosts" +
rem  a hundred digits). temp 0.1 + stop strings + a tight max_tokens is the
rem  client contract; treat any emitted path as unverified input.
rem
rem  *** USE Q8_0 OR BF16. THE K-QUANTS ARE BROKEN. ***
rem  Measured on this box, same prompt/template/server, greedy:
rem     Q8_0      correct <tool_call> {"city":"Shanghai"}
rem     BF16      same
rem     Q4_K_M    degenerates ("sh, sh, h, sh"), 1/3 on tool_test.py
rem     UD-Q4_K_XL degenerates into prose loops, 0/3
rem  Upstream's quants are broken, not a local problem. Do not "optimize"
rem  this model down to Q4 without re-running tool_test.py.
rem
rem  MEASURED (llama-bench b10743, i5-8500 6c / DDR4-2400):
rem     Q8_0  604 MiB  tg128 25.1 t/s @ -t 4   (28.6 @ -t 6 for Q4)
rem     KV    112 KiB/token f16 (28 layers x 8 kv heads x 128 dim x2 x2B)
rem            -> 8192 ctx = 0.88 GiB f16 / 0.22 GiB q4_0
rem     native context_length = 40960 -> 8k needs NO RoPE extension.
rem
rem  TEMPLATE: qwen06t_template.jinja = the GGUF's embedded template with
rem  the empty-think prefix forced on. Without it the model starts a
rem  chain, spills a stray </think> into the answer and never stops.
rem  (Passing {"enable_thinking":false} via --chat-template-kwargs is the
rem  alternative but quoting that JSON through cmd->PowerShell is a trap.)
rem
rem  This file is deliberately pure ASCII - cmd.exe mis-decodes UTF-8
rem  .cmd files on codepage 936 (see AGENTS.md / README-qwen.md).
rem ====================================================================
setlocal EnableDelayedExpansion
chcp 65001 >nul
title Qwen3-0.6B-tools Agent API - 127.0.0.1:8086

rem -------------------------- knobs --------------------------
set "MODEL=%~dp0models\Qwen3-0.6B-tools-Q8_0.gguf"
set "TPL=%~dp0qwen06t_template.jinja"
set "ALIAS=qwen3-0.6b-tools"
set "PORT=8086"          rem 8080 others / 8081 minicpm / 8082 8B-A1B / 8084-85 sakthai
set "CTX=8192"           rem agent turns, not document QA; 0.22 GiB KV at q4_0
set "CTK=q4_0"           rem f16 KV = 2x slower prefill here and saves only 0.65 GiB
set "THREADS=4"
set "TBATCH=4"
set "UBATCH=512"
set "NPAR=1"
set "CACHEMB=2048"       rem 0.60 GiB weights + 0.22 GiB KV
set "CTXCP=32"
set "CPMINSTEP=256"
set "CPCREUSE=256"       rem tool loops replay the same prefix -> reuse pays here
set "LOWPRI=1"
set "TEMP=0.1"           rem tool JSON must be exact; validation used greedy
set "TOPP=0.8"
set "TOPK=20"
set "REPEAT=1.15"        rem 1.05 was NOT enough: pi + no stop -> 2000-token degenerate loop
set "MAXTOK=512"         rem client-side cap; a 0.6B never stops on its own
set "SERVE=%~dp0Bonsai-demo\bin\cpu\llama-server.exe"
rem -----------------------------------------------------------

echo ================================================================
echo  Qwen3-0.6B-tools  Q8_0  --  native tool calling, CPU only
echo  model %MODEL%
echo  port  http://127.0.0.1:%PORT%/v1   (alias %ALIAS%)
echo  NOTE  Q4_K_M / UD-Q4_K_XL of this model are broken - do not use.
echo ================================================================

if not exist "%SERVE%" (
    echo [ERR] not found: %SERVE%    & pause & exit /b 1
)
if not exist "%MODEL%" (
    echo [ERR] not found: %MODEL%
    echo       bash dl.sh iromu/Qwen3-0.6B-tools-GGUF Qwen3-0.6B-tools-Q8_0.gguf models 8
    pause & exit /b 1
)
if not exist "%TPL%" (
    echo [ERR] not found: %TPL%    & pause & exit /b 1
)

netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 echo [WARN] port %PORT% already in use -- stop that server first

rem --jinja renders TPL (it carries the GGUF's <tools> contract and the
rem empty-think prefix). --chat-template takes an INLINE string;
rem --chat-template-file takes the path.
set "ARGS=-m %MODEL% --alias %ALIAS% -ngl 0 -fa on -c %CTX% -ctk %CTK% -ctv %CTK% -t %THREADS% -tb %TBATCH% -ub %UBATCH% -n 0 --host 127.0.0.1 --port %PORT% --parallel %NPAR% --cache-ram %CACHEMB% --ctx-checkpoints %CTXCP% --checkpoint-min-step %CPMINSTEP% --cache-reuse %CPCREUSE% --cache-idle-slots --context-shift --jinja --chat-template-file %TPL% -rea off --temp %TEMP% --top-p %TOPP% --top-k %TOPK% --repeat-penalty %REPEAT% --repeat-last-n 128"

if "%LOWPRI%"=="1" (
    powershell -NoProfile -Command "$p=Start-Process -FilePath '%SERVE%' -ArgumentList '%ARGS%' -NoNewWindow -PassThru; $p.PriorityClass='BelowNormal'; $p.WaitForExit()"
) else (
    "%SERVE%" %ARGS%
)

echo.
echo [RUN] %ALIAS%  ctx=%CTX%  KV=%CTK%  t=%THREADS%  tb=%TBATCH%  ub=%UBATCH%
echo       ~25 t/s generate, ~250 t/s prompt (measured Q8_0, b10743)
echo       ENDPOINT http://127.0.0.1:%PORT%/v1
echo       tools  : cd /d %~dp0 ^& python tool_test.py %PORT% %ALIAS% %TEMP%
echo       loop   : cd /d %~dp0 ^& python agent_loop.py 5 %PORT% %ALIAS% %TEMP%
echo       client : send stop=[im_end,\nuser,\nsystem], max_tokens ^<= %MAXTOK%
echo       stop   : powershell -c "Stop-Process -Name llama-server -Force"
echo.
echo [done] server stopped
pause
endlocal