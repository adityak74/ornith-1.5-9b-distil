#!/bin/zsh
set -e
cd /Users/aditya/Projects/adityak74/ornith-1.5-9b-distil
M=/Volumes/SATECHI/.omlx/prism-ml/Ternary-Bonsai-27B-mlx-2bit
E="uv run odistil --distill-config configs/distill.yaml eval --benchmark humaneval --max-tokens 2048"
eval $E --model $M --tag p25-bonsai-plain
eval $E --model $M --think-bias 1024:0.02 --tag p26-bonsai-bias1024-0.02
echo PROBES_DONE
