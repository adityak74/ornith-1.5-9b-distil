#!/bin/zsh
# Ramp sweep + generality (DECISIONS.md 50). HumanEval @ 2048.
set -e
cd /Users/aditya/Projects/adityak74/ornith-1.5-9b-distil
V1=/Volumes/SATECHI/.omlx/Ornith-1.5-9B-MLX-distil-oQ4
ST=/Volumes/SATECHI/.omlx/Ornith-1.5-9B-MLX-oQ4
E="uv run odistil --distill-config configs/distill.yaml eval --benchmark humaneval --max-tokens 2048"
eval $E --model $V1 --think-bias 1024:0.01 --tag p5-bias1024-0.01
eval $E --model $V1 --think-bias 1024:0.04 --tag p6-bias1024-0.04
eval $E --model $V1 --think-bias 768:0.02  --tag p7-bias768-0.02
eval $E --model $V1 --think-bias 1280:0.02 --tag p8-bias1280-0.02
eval $E --model $ST --tag p9-stock-plain
eval $E --model $ST --think-bias 1024:0.02 --tag p10-stock-bias1024-0.02
echo PROBES_DONE
