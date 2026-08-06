#!/usr/bin/env bash
while true; do
  clear
  n=$(ls results/curve_kd*.json 2>/dev/null | wc -l)
  echo "══ прогонов готово: ${n}/15 ══"
  echo

  for f in $(ls -1t logs/curve_kd*.log 2>/dev/null | head -4); do
    tag=$(basename "$f" .log)
    last=$(grep -E "^epoch" "$f" 2>/dev/null | tail -1)
    ep=$(echo "$last" | awk '{print $2}')
    ep=${ep:-0}
    pct=$(( ep * 100 / 3000 ))
    filled=$(( pct / 5 ))

    printf "%-16s " "$tag"
    printf "["
    printf "%${filled}s" | tr ' ' '='
    printf "%$(( 20 - filled ))s" | tr ' ' ' '
    printf "] %3d%%  %s\n" "$pct" "$(echo "$last" | cut -d' ' -f3-6)"
  done

  echo
  nvidia-smi --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader
  sleep 10
done
