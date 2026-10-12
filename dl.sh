#!/bin/bash
# =============================================================================
#  dl.sh - GGUF 模型下载（国内镜像优先 + 8 路并行分片 + 严格校验）
#
#  用法:
#     ./dl.sh <owner/repo> <file.gguf> <dest_dir> [并发数]
#
#  例:
#     ./dl.sh LiquidAI/LFM2.5-2.6B-GGUF LFM2.5-2.6B-Q4_K_M.gguf models
#     ./dl.sh LiquidAI/LFM2.5-2.6B-GGUF LFM2.5-2.6B-Q4_K_M.gguf models 16
#
#  镜像策略（按实测稳定性排序，2026-09 于本机测得）:
#     1. ModelScope 魔搭  https://modelscope.cn/models/<repo>/resolve/master/<f>
#        国内直连，5.1 MB/s 且不抽风 —— 首选
#     2. hf-mirror.com   https://hf-mirror.com/<repo>/resolve/main/<f>
#        有时 4 MB/s，有时掉到 70 KB/s —— 只作备用
#     两者都支持 Range（实测返回 206），所以分片下载可行。
#     哪个源先成功用哪个，失败自动切下一个。
#
#  特性:
#     - 8 路（可调）并行 range 下载，单连接限速时仍能跑满带宽
#     - 每片最多重试 5 次，并带 --speed-limit/--speed-time 掐掉卡死的连接
#     - 严格按总字节数校验，对不上就报错，绝不留下半个坏 GGUF
#     - 分片落在 <dest>/.dlparts_<repo>_<file>/ ，重跑可复用已下好的片
#
#  !! 分片合并必须按数字序，不能用 cat part.* ——
#     bash glob 是字典序（part.0 part.1 part.10 part.11 ... part.2），
#     字节块会错位，大小校验照样通过但内容全是垃圾。
#     症状：模型能加载、速度正常，但输出是 ////// 或乱码。
# =============================================================================
set -u

REPO="${1:?用法: dl.sh <owner/repo> <file.gguf> <dest_dir> [nparts]}"
FILE="${2:?用法: dl.sh <owner/repo> <file.gguf> <dest_dir> [nparts]}"
DEST="${3:?用法: dl.sh <owner/repo> <file.gguf> <dest_dir> [nparts]}"
N="${4:-8}"

# ---- 候选镜像，按优先级排列 -------------------------------------------------
MIRRORS=(
  "https://modelscope.cn/models/${REPO}/resolve/master/${FILE}"
  "https://hf-mirror.com/${REPO}/resolve/main/${FILE}"
  "https://huggingface.co/${REPO}/resolve/main/${FILE}"
)
NAMES=("ModelScope(魔搭)" "hf-mirror" "huggingface 直连")

# ---- 分片目录（重跑可续） ---------------------------------------------------
SAFE=$(echo "${REPO}_${FILE}" | tr '/' '_')
TMP="${DEST}/.dlparts_${SAFE}"
mkdir -p "$DEST" "$TMP"

# ---- 逐个镜像尝试，先拿到文件大小的那个胜出 -------------------------------
pick_mirror() {
  for i in "${!MIRRORS[@]}"; do
    local url="${MIRRORS[$i]}"
    printf '[%d/%d] 探测 %-20s ... ' "$((i+1))" "${#MIRRORS[@]}" "${NAMES[$i]}"
    local hdr code
    # ① 先用 HEAD 拿 content-length / x-linked-size
    hdr=$(curl -sSIL --ssl-no-revoke --max-time 30 "$url" 2>/dev/null)
    code=$(echo "$hdr" | grep -iE "^HTTP/" | tail -1 | awk '{print $2}')
    TOTAL=$(echo "$hdr" | grep -iE "^(x-linked-size|content-length)" | tail -1 | tr -d '\r' | awk '{print $2}')
    # ② HEAD 拿不到就退化成 1 字节 range GET，从 Content-Range 的 "/总长" 里取
    #    （ModelScope 的 HEAD 是 chunked 的 200，没有 content-length，只有这条路拿得到）
    if [ -z "${TOTAL:-}" ]; then
      TOTAL=$(curl -sSL --ssl-no-revoke --max-time 30 -r 0-0 -D - -o /dev/null "$url" 2>/dev/null \
              | grep -i "^content-range" | tail -1 | tr -d '\r' | sed 's#.*/##')
    fi
    if [ "${code:-0}" -ge 200 ] && [ "${code:-0}" -lt 400 ] && [ -n "${TOTAL:-}" ] && [ "$TOTAL" -gt 1000000 ]; then
      echo "OK  size=$TOTAL"
      URL="$url"; PICKED=$i
      return 0
    fi
    echo "不可用 (http=${code:-?} size=${TOTAL:-?})"
  done
  return 1
}

pick_mirror || { echo "[ERR] 所有镜像都拿不到文件大小: $REPO/$FILE"; exit 1; }

# 已下好且大小正确 -> 直接收工
if [ -f "$DEST/$FILE" ] && [ "$(wc -c < "$DEST/$FILE")" -eq "$TOTAL" ]; then
  echo "[SKIP] 已存在且大小正确: $DEST/$FILE ($TOTAL bytes)"; exit 0
fi

CHUNK=$(( (TOTAL + N - 1) / N ))
echo "镜像   : ${NAMES[$PICKED]}"
echo "目标   : $DEST/$FILE"
echo "大小   : $TOTAL bytes ($(( TOTAL / 1000000 )) MB)"
echo "分片   : $N 片，每片 $(( CHUNK / 1000000 )) MB"
echo

# ---- 8 路并行 ---------------------------------------------------------------
for i in $(seq 0 $((N-1))); do
  START=$(( i * CHUNK ))
  END=$(( START + CHUNK - 1 ))
  [ $END -ge $TOTAL ] && END=$(( TOTAL - 1 ))
  [ $START -ge $TOTAL ] && continue
  (
    # 已下好且长度正确的片直接跳过（断点续片）
    if [ -f "$TMP/part.$i" ] && [ "$(wc -c < "$TMP/part.$i")" -eq $(( END - START + 1 )) ]; then
      exit 0
    fi
    for a in 1 2 3 4 5; do
      if curl -sSL --ssl-no-revoke \
              --retry 3 --retry-delay 2 \
              --connect-timeout 20 --max-time 3600 \
              --speed-limit 20480 --speed-time 60 \
              -r "${START}-${END}" -o "$TMP/part.$i.tmp" "$URL"; then
        mv -f "$TMP/part.$i.tmp" "$TMP/part.$i"
        exit 0
      fi
      echo "  片 $i 第 $a 次重试"
      sleep 3
    done
    echo "  [ERR] 片 $i 彻底失败"
    exit 1
  ) &
done
wait

# ---- 合并 + 严格校验 -------------------------------------------------------
# 拼接顺序必须是「数字序」而不是 bash glob 的「字典序」。
# 16 片时 glob 会给出 part.0 part.1 part.10 part.11 ... part.2 —— 字节块全错位，
# 文件大小仍然正确、内容全是垃圾（症状：模型能加载但输出 ////// 或乱码）。
PARTS=()
for i in $(seq 0 $((N-1))); do [ -f "$TMP/part.$i" ] && PARTS+=("$TMP/part.$i"); done
if [ "${#PARTS[@]}" -ne "$N" ]; then
    echo "[ERR] 只找到 ${#PARTS[@]}/$N 个分片: $TMP"
    exit 1
fi

ACT=$(cat "${PARTS[@]}" 2>/dev/null | wc -c)
printf '\n下载完成: %s / %s bytes\n' "$ACT" "$TOTAL"
if [ "$ACT" -ne "$TOTAL" ]; then
  echo "[ERR] 大小不符，保留分片以便重跑: $TMP"
  exit 1
fi

cat "${PARTS[@]}" > "$DEST/$FILE"
FINAL=$(wc -c < "$DEST/$FILE")
rm -rf "$TMP"
if [ "$FINAL" -eq "$TOTAL" ]; then
  echo "[OK] $DEST/$FILE  ($(( FINAL / 1000000 )) MB)"
else
  echo "[ERR] 合并后大小异常: $FINAL != $TOTAL"; rm -f "$DEST/$FILE"; exit 1
fi
