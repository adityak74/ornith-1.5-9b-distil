#!/bin/zsh
# Decode-time probes on v1 @ oQ4, HumanEval @ 2048 (DECISIONS.md 50).
set -e
cd /Users/aditya/Projects/adityak74/ornith-1.5-9b-distil
M=/Volumes/SATECHI/.omlx/Ornith-1.5-9B-MLX-distil-oQ4
E="uv run odistil --distill-config configs/distill.yaml eval --model $M --benchmark humaneval --max-tokens 2048"
eval $E --tag p0-plain
eval $E --rep-penalty 1.1 --tag p1-rep1.1
eval $E --temp 0.6 --tag p2-temp0.6
eval $E --think-bias 1024:0.005 --tag p3-bias1024-0.005
eval $E --think-bias 1024:0.02 --tag p4-bias1024-0.02
echo PROBES_DONE
