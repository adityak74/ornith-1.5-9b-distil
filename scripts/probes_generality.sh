#!/bin/zsh
# Soft HALT generality: bf16 parent, oQ3, oQ8, the 35B teacher (DECISIONS.md 53). HumanEval @ 2048.
set -e
cd /Users/aditya/Projects/adityak74/ornith-1.5-9b-distil
O=/Volumes/SATECHI/.omlx
E="uv run odistil --distill-config configs/distill.yaml eval --benchmark humaneval --max-tokens 2048"
eval $E --model $O/ornith-ai/Ornith-1.5-9B-MLX --think-bias 1024:0.02 --tag p12-bf16-bias1024-0.02
eval $E --model $O/Ornith-1.5-9B-MLX-oQ3 --tag p13-oq3-plain
eval $E --model $O/Ornith-1.5-9B-MLX-oQ3 --think-bias 1024:0.02 --tag p14-oq3-bias1024-0.02
eval $E --model $O/Ornith-1.5-9B-MLX-oQ8 --tag p15-oq8-plain
eval $E --model $O/Ornith-1.5-9B-MLX-oQ8 --think-bias 1024:0.02 --tag p16-oq8-bias1024-0.02
eval $E --model $O/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit --tag p17-35b-plain
eval $E --model $O/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit --think-bias 1024:0.02 --tag p18-35b-bias1024-0.02
echo PROBES_DONE
