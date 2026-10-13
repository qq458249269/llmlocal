param(
  [int]$Ctx = 65536,
  [string]$CtK = "q8_0",
  [int]$Threads = 4,
  [int]$HeadroomPct = 10,         # keep this much RAM + CPU free
  [int]$MaxRam = 8,               # HARD cap for the model process, GiB
  [int]$MaxThreads = 4,           # HARD cap for llama threads
[string]$Size = "8b",           # peak table: 8b | 17b | lfm25 | lfm12 | q7b | q30b | q35_4b | ornith9b | ornith35b | lfm258b
  [string]$Only = "",             # print just this key's value (for .cmd parsing)
  [switch]$CountMmap              # treat the mmapped weights as reclaimable (see note)
)
# NOTE ON THE $PEAK_* TABLES -- WorkingSet vs PrivateWS
#
# These tables record WORKING SET, which counts the mmapped weight pages.
# Those pages are FILE-BACKED and Windows reclaims them under pressure, so
# WorkingSet badly overstates the memory the model really holds.
# Measured on this box (LFM2.5-8B-A1B, same server, only -c changed):
#       ctx|kv        WorkingSet   PrivateWS
#       32768|f16       8.48 GiB     3.72 GiB
#       131072|f16      9.71 GiB     5.23 GiB
# The gap is a CONSTANT 4.76 GiB = the 4.95 GiB of mmap'd weights.  It does
# NOT move with -c or KV type, because KV is pre-allocated private memory.
#
# So when you are RAM-bound and the planner refuses a config that would
# actually fit, pass -CountMmap: the estimate drops by the mmap portion and
# the comparison against free RAM becomes honest.  WITHOUT this flag the
# planner over-downgrades -- it turned a working 16384/f16 into 8192/q4_0,
# which is the single worst config for an agent (q4_0 KV costs 2x prefill).
#
# $MMAP_RECLAIM = GiB of the estimate that is reclaimable file cache, per size.
# Only measured values are listed; anything else gets 0 (i.e. current behaviour).
# Measured peak working set (GiB) on this box, PrismML llama.cpp CPU fork.
# $WEIGHTS = size of the GGUF on disk; the rest of the peak is KV cache + buffers.
$PEAK_8B = @{
  "8192|q8_0"    = 2.6;  "8192|q4_0"    = 2.4
  "16384|q8_0"   = 3.2;  "16384|q4_0"   = 2.7;  "16384|f16" = 3.9
  "32768|q8_0"   = 4.34; "32768|q4_0"   = 3.0;  "32768|f16" = 6.44
  "65536|q8_0"   = 6.73; "65536|q4_0"   = 4.01; "65536|f16" = 10.86
  "131072|q8_0"  = 11.51;"131072|q4_0"  = 7.01; "131072|f16"= 13.63
}
# Bonsai-1.7B / Ternary-Bonsai-1.7B (qwen3 arch, 28 layers, 8 KV heads)
$PEAK_17B = @{
  "8192|q8_0"    = 1.3;  "8192|q4_0"    = 1.1
  "16384|q8_0"   = 1.6;  "16384|q4_0"   = 1.4;  "16384|f16" = 2.2
  "32768|q8_0"   = 2.2;  "32768|q4_0"   = 1.7;  "32768|f16" = 3.3
  "65536|q8_0"   = 3.29; "65536|q4_0"   = 2.46; "65536|f16" = 7.50
  "131072|q8_0"  = 7.94; "131072|q4_0"  = 4.42; "131072|f16"= 10.46
}
# LFM2.5-2.6B Q4_K_M (1.67 GiB weights, lfm2 arch, 30 layers of which only 8 are
# GQA attention - the other 22 are 3-token short convolutions and use NO KV cache).
# KV cost = 8 layers x 8 kv-heads x 64 dim x 2 (K+V) = 16 KiB/token f16,
# i.e. ~9x cheaper per token than the qwen3 arch, and ~0 at the conv layers.
#   f16: 16 KiB/tok -> 0.13 / 0.25 / 0.50 / 1.00 / 2.00 GiB at 8k/16k/32k/64k/128k
#   q8_0 and q4_0 halve that again.
$PEAK_LFM25 = @{
  "8192|f16"     = 2.0;  "8192|q8_0"    = 1.9;  "8192|q4_0"    = 1.9
  "16384|f16"    = 2.1;  "16384|q8_0"   = 2.0;  "16384|q4_0"   = 2.0
  "32768|f16"    = 2.4;  "32768|q8_0"   = 2.2;  "32768|q4_0"   = 2.0
  "65536|f16"    = 2.9;  "65536|q8_0"   = 2.4;  "65536|q4_0"   = 2.2
  "131072|f16"   = 3.9;  "131072|q8_0"  = 2.9;  "131072|q4_0"  = 2.5
}
# LFM2.5-8B-A1B: 4.95 GiB weights + KV.  Only 6 of 24 layers have attention
# (8 kv_head x 64 dim x 2 x 2 B = 12 KiB/token f16), so KV is unusually small.
#
# !! THESE ARE WORKING SET, NOT PRIVATE COMMIT.  mmap puts the 4.95 GiB of
# !! weights in reclaimable file cache, so WorkingSet overstates real usage.
# !! Re-measured 2026-09-30 (same server, only -c changed):
# !!      ctx|kv      WorkingSet   PrivateWS
# !!      131072|f16    9.71 GiB     5.23 GiB
# !!       32768|f16    8.48 GiB     3.72 GiB
# !!      131072-32768 = 98304 x 12 KiB = 1.47 GiB, and the PrivateWS gap is
# !!      1.51 GiB -- so -c is the only knob that moves real memory.
# !! Bare vs full (--cache-ram 1024 --ctx-checkpoints 64 --cache-idle-slots)
# !! measured identical: 3.72 / 8.48 both ways.  Those are caps, not reserves.
$PEAK_LFM25_8B = @{
  "8192|f16"     = 8.3;  "8192|q8_0"    = 8.2;  "8192|q4_0"    = 8.1
  "16384|f16"    = 8.4;  "16384|q8_0"   = 8.3;  "16384|q4_0"   = 8.2
  "32768|f16"    = 8.5;  "32768|q8_0"   = 8.4;  "32768|q4_0"   = 8.2
  "65536|f16"    = 8.7;  "65536|q8_0"   = 8.5;  "65536|q4_0"   = 8.3
  "131072|f16"   = 9.7;  "131072|q8_0"  = 8.8;  "131072|q4_0"  = 8.5
}
# decode t/s on this model tracks the CURRENT KV length, not -c (KV is read on
# demand).  Measured -c 32768 / f16 / -t 4:
#     KV 2.0k 11.5 t/s | 7.2k 7.6 | 13.9k 5.1 | 17.8k 3.1
# So "shrink -c" does not make generation faster; keeping the conversation
# short does.  -t 6 is WORSE at 7.2k (3.46 vs 7.63 t/s).
# LFM2.5-1.2B-Instruct Q4_K_M (0.68 GiB weights, same lfm2 arch family).
# Same 16 KiB/token KV, but the weights are 2.3x smaller, so this is the
# throughput choice: measured 19.5 t/s end-to-end vs 10.4 t/s for the 2.6B.
$PEAK_MCPM5 = @{
  "8192|f16"     = 1.2;  "8192|q8_0"    = 1.1;  "8192|q4_0"    = 1.0
  "16384|f16"    = 1.4;  "16384|q8_0"   = 1.2;  "16384|q4_0"   = 1.1
  "32768|f16"    = 1.7;  "32768|q8_0"   = 1.4;  "32768|q4_0"   = 1.2
  "65536|f16"    = 2.5;  "65536|q8_0"   = 1.7;  "65536|q4_0"   = 1.4
  "131072|f16"   = 3.9;  "131072|q8_0"  = 2.5;  "131072|q4_0"  = 1.75
}
$PEAK_LFM12 = @{
  "8192|f16"     = 1.1;  "8192|q8_0"    = 1.0;  "8192|q4_0"    = 1.0
  "16384|f16"    = 1.2;  "16384|q8_0"   = 1.1;  "16384|q4_0"   = 1.0
  "32768|f16"    = 1.4;  "32768|q8_0"   = 1.2;  "32768|q4_0"   = 1.1
  "65536|f16"    = 1.9;  "65536|q8_0"   = 1.5;  "65536|q4_0"   = 1.3
  "131072|f16"   = 2.9;  "131072|q8_0"  = 2.0;  "131072|q4_0"  = 1.7
}
# Qwen2.5-Coder-7B-Instruct Q4_K_M (4.36 GiB weights, qwen2 arch, 28 layers,
# 4 KV heads x 128 dim -> 56 KiB/token f16, same order as the 8B Bonsai but the
# weights are 2x bigger).  f16: 0.44/0.88/1.75/3.50/7.00 GiB at 8k..128k
$PEAK_Q7B = @{
  "8192|f16"     = 4.9;  "8192|q8_0"    = 4.7;  "8192|q4_0"    = 4.5
  "16384|f16"    = 5.4;  "16384|q8_0"   = 5.1;  "16384|q4_0"   = 4.7
  "32768|f16"    = 6.3;  "32768|q8_0"   = 5.6;  "32768|q4_0"   = 5.0
  "65536|f16"    = 8.1;  "65536|q8_0"   = 6.8;  "65536|q4_0"   = 5.6
  "131072|f16"   = 11.6; "131072|q8_0"  = 9.2;  "131072|q4_0"  = 6.8
}
# Qwen3-Coder-30B-A3B UD-IQ2_XXS (9.65 GiB weights, qwen3moe, 48 layers,
# 4 KV heads x 128 dim -> 96 KiB/token f16, i.e. ~1.7x the 7B's 56 KiB).
# f16: 0.75/1.50/3.00/6.00/12.00 GiB at 8k..128k
$PEAK_Q30B = @{
  "8192|f16"     = 10.5; "8192|q8_0"    = 10.2; "8192|q4_0"    = 9.9
  "16384|f16"    = 11.3; "16384|q8_0"   = 10.7; "16384|q4_0"   = 10.1
  "32768|f16"    = 12.9; "32768|q8_0"   = 11.5; "32768|q4_0"   = 10.4
  "65536|f16"    = 16.1; "65536|q8_0"   = 13.1; "65536|q4_0"   = 11.0
  "131072|f16"   = 22.2; "131072|q8_0"  = 16.3; "131072|q4_0"  = 12.6
}
# Ornith-1.0/1.5-9B Q4_K_M (5.12 GiB weights; IQ4_XS was 4.83 GiB and 1.9x
# slower at prefill on AVX2 -- see agent-ornith9b.cmd).  qwen35 arch, 32 layers
# of which only 8 are full attention -- the other 24 are linear_attention/SSM
# CONSTANT state, so only the 8 full-attn layers pay KV.
#   8 layers x 4 KV heads x 256 dim x 2 (K+V) x 2 B = 32 KiB/token f16
#   262144 is the model's native max_position_embeddings.
# f16: 0.19/0.38/0.75/1.50/3.00/6.00 GiB at 8k..256k
$PEAK_ORNITH9B = @{
  "8192|f16"     = 5.5;  "8192|q8_0"    = 5.3;  "8192|q4_0"    = 5.2
  "16384|f16"    = 5.7;  "16384|q8_0"   = 5.4;  "16384|q4_0"   = 5.2
  "32768|f16"    = 6.1;  "32768|q8_0"   = 5.5;  "32768|q4_0"   = 5.2
  "65536|f16"    = 6.8;  "65536|q8_0"   = 5.9;  "65536|q4_0"   = 5.4
  "131072|f16"   = 8.3;  "131072|q8_0"  = 6.7;  "131072|q4_0"  = 5.7
  "262144|f16"   = 11.3; "262144|q8_0"  = 8.7;  "262144|q4_0"  = 6.7
}
# Qwen3.5-4B Q4_K_M (3.01 GB file = 2.81 GiB weights, qwen35 arch) --
# SAME KV profile as the 9B above, verified against the HF config.json:
#   text_config: 32 layers, full_attention_interval 4 -> only 8 full-attn
#   layers (4 KV heads x 256 dim); the other 24 are linear_attention, whose
#   recurrent state is CONSTANT (~50 MiB total) and never grows with -c.
#   => 8 x 4 x 256 x 2 x 2 B = 32 KiB/token f16, native max_position 262144
# So this table is ORNITH9B shifted down by the 2.31 GiB weight difference
# (5.12 - 2.81).  ponytail: DERIVED, not measured on this box -- every value
# carries ~0.1 GiB of slack, same as the hand-written ones above.  Replace it
# with real numbers after one `./bench_decode.py` run (see agent-qwen35-4b.cmd).
$PEAK_Q35_4B = @{
  "8192|f16"     = 3.2;  "8192|q8_0"    = 3.0;  "8192|q4_0"    = 2.9
  "16384|f16"    = 3.4;  "16384|q8_0"   = 3.1;  "16384|q4_0"   = 2.9
  "32768|f16"    = 3.8;  "32768|q8_0"   = 3.2;  "32768|q4_0"   = 2.9
  "65536|f16"    = 4.5;  "65536|q8_0"   = 3.6;  "65536|q4_0"   = 3.1
  "131072|f16"   = 6.0;  "131072|q8_0"  = 4.4;  "131072|q4_0"  = 3.4
  "262144|f16"   = 9.0;  "262144|q8_0"  = 6.4;  "262144|q4_0"  = 4.4
}
# Ornith-1.0-35B MoE IQ2_M (11.66 GB file,
# 2 KV heads x 256 dim -> 20 KiB/token f16 = 0.38x the 7B's 56).  Kept for
# reference: at ~11.7 GiB peak it does NOT fit alongside Windows on this box.
$PEAK_ORNITH35B = @{
  "8192|f16"     = 12.4; "8192|q8_0"    = 12.1; "8192|q4_0"    = 11.9
  "16384|f16"    = 12.7; "16384|q8_0"   = 12.2; "16384|q4_0"   = 11.9
  "32768|f16"    = 13.3; "32768|q8_0"   = 12.5; "32768|q4_0"   = 12.0
  "65536|f16"    = 14.6; "65536|q8_0"   = 13.2; "65536|q4_0"   = 12.2
  "131072|f16"   = 17.4; "131072|q8_0"  = 14.8; "131072|q4_0"  = 12.8
}
if ($Size -eq "17b")   { $PEAK = $PEAK_17B;   $WEIGHTS = 0.5;  $MMAP = 0.43 }
elseif ($Size -eq "q7b")  { $PEAK = $PEAK_Q7B;  $WEIGHTS = 4.36 }
elseif ($Size -eq "q30b") { $PEAK = $PEAK_Q30B; $WEIGHTS = 9.65 }
elseif ($Size -eq "q35_4b") { $PEAK = $PEAK_Q35_4B; $WEIGHTS = 2.81 }
elseif ($Size -eq "ornith9b")  { $PEAK = $PEAK_ORNITH9B;  $WEIGHTS = 5.12 }
elseif ($Size -eq "ornith35b") { $PEAK = $PEAK_ORNITH35B; $WEIGHTS = 11.15 }
elseif ($Size -eq "lfm25") { $PEAK = $PEAK_LFM25; $WEIGHTS = 1.7 }
elseif ($Size -eq "lfm12") { $PEAK = $PEAK_LFM12; $WEIGHTS = 0.75 }
elseif ($Size -eq "mcpm5") { $PEAK = $PEAK_MCPM5; $WEIGHTS = 0.64 }
elseif ($Size -eq "lfm258b") { $PEAK = $PEAK_LFM25_8B; $WEIGHTS = 4.95; $MMAP = 4.76 }
else                   { $PEAK = $PEAK_8B;    $WEIGHTS = 2.1;  $MMAP = 0 }
if (-not $MMAP) { $MMAP = 0 }
# Degrade order: cheapest quality loss first.
$KV_ORDER   = @("f16", "q8_0", "q4_0")
$CTX_ORDER  = @(131072, 65536, 32768, 16384, 8192)

$os   = Get-CimInstance Win32_OperatingSystem
$totG = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
$useG = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 2)
$freeG= [math]::Round($os.FreePhysicalMemory / 1MB, 2)
$cores= (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors

$notes = @()
$keep  = $HeadroomPct / 100.0
$sysBudget = [math]::Floor($totG * (1 - $keep) - $useG)   # RAM the model may still take
$maxT  = [math]::Floor($cores * (1 - $keep))                # CPU threads allowed by the box
if ($maxT -gt $MaxThreads) { $maxT = $MaxThreads }           # ... and by the user
if ($maxT -lt 1) { $maxT = 1 }
$budget = $MaxRam
if ($sysBudget -lt $budget) {
  $budget = $sysBudget
  $notes += "ram cap ${MaxRam} to ${sysBudget} GiB - system already uses ${useG} GiB, keep ${HeadroomPct} pct free"
}

if ($Threads -gt $maxT) { $notes += "threads ${Threads} to ${maxT} (cap ${MaxThreads} of ${cores} cores, keep ${HeadroomPct} pct free)"; $Threads = $maxT }
if ($Threads -ge $cores) { $notes += "warning: -t ${Threads} on ${cores} cores will max out the CPU; -t 4 measured fastest here" }

function Get-Est([int]$c, [string]$k) {
  $key = "$c|$k"
  if ($PEAK.ContainsKey($key)) { return $PEAK[$key] }
  # Unmeasured combo: interpolate from the nearest measured ctx with same KV,
  # assuming flat weight size + linear KV growth; round up 0.5 GiB to stay safe.
  $nearest = $CTX_ORDER | Where-Object { $PEAK.ContainsKey("$_|$k") } | Sort-Object { [math]::Abs($_ - $c) } | Select-Object -First 1
  if (-not $nearest) { return 99 }
  $base = $PEAK["$nearest|$k"]
  $kv   = $base - $WEIGHTS
  [math]::Ceiling($WEIGHTS + $kv * ($c / [double]$nearest) * 2) / 2
}

# Degrade until it fits the RAM budget.
$est = Get-Est $Ctx $CtK
# -CountMmap: subtract the reclaimable file-backed weight pages from the
# estimate before comparing against free RAM.  The estimate is still
# REPORTED as-is (that is what Task Manager will show) but the fit test
# uses the honest private-commit number.
$fit = $est
if ($CountMmap -and $MMAP -gt 0) {
  $fit = [math]::Max($WEIGHTS, $est - $MMAP)
  $notes += "mmap accounted: $est GiB WorkingSet - $MMAP GiB reclaimable = $fit GiB private"
}
$guard = 0
while ($fit -gt $budget -and $guard -lt 12) {
  $guard++
  $old = "$Ctx|$CtK"
  $ki  = [array]::IndexOf($KV_ORDER, $CtK)
  if ($ki -ge 0 -and $ki -lt $KV_ORDER.Count - 1) { $CtK = $KV_ORDER[$ki + 1] }        # f16 -> q8_0 -> q4_0
  else {
    $ci = [array]::IndexOf($CTX_ORDER, $Ctx)
    if ($ci -ge 0 -and $ci -lt $CTX_ORDER.Count - 1) { $Ctx = $CTX_ORDER[$ci + 1] }    # halve the context
    else { break }
  }
  $est = Get-Est $Ctx $CtK
  $fit = $est
  if ($CountMmap -and $MMAP -gt 0) { $fit = [math]::Max($WEIGHTS, $est - $MMAP) }
  if ("$old" -ne "$Ctx|$CtK") { $notes += "downgraded $old to $Ctx+$CtK (needs less RAM)" }
}

$out = [ordered]@{
  ERROR = 0
  CTX   = $Ctx
  CTK   = $CtK
  THREADS = $Threads
  EST   = $est
  BUDGET  = $budget
  TOTRAM  = $totG
  USERAM  = $useG
  FREERAM = $freeG
  CORES   = $cores
  MAXTHREADS = $maxT
  MAXRAM     = $MaxRam
  NOTE   = ($notes -join "; ")
}
if ($fit -gt $budget) {
  $out.ERROR = 1
  $out.NOTE  = "not enough RAM headroom: needs $fit GiB, only $budget GiB allowed"
}
if ($Only) {
  if ($Only -eq "NOTE") { Write-Output $out.NOTE }
  else { Write-Output $out[$Only] }
  if ($out.ERROR -eq 1) { exit 1 } else { exit 0 }
}
$out.GetEnumerator() | ForEach-Object { Write-Output ("{0}={1}" -f $_.Key, $_.Value) }
if ($out.ERROR -eq 1) { exit 1 }
