#!/bin/zsh
# Soft HALT on other model families + TruthfulQA (DECISIONS.md 54).
set -e
cd /Users/aditya/Projects/adityak74/ornith-1.5-9b-distil
O=/Volumes/SATECHI/.omlx
E="uv run odistil --distill-config configs/distill.yaml eval --max-tokens 2048"
eval $E --model $O/Ornith-1.5-9B-MLX-distil-oQ4 --benchmark truthfulqa --tag p19-v1-tqa-plain
eval $E --model $O/Ornith-1.5-9B-MLX-distil-oQ4 --benchmark truthfulqa --think-bias 1024:0.02 --tag p20-v1-tqa-bias1024-0.02
eval $E --model $O/lmstudio-community/Qwen3.8-27B-MLX-4bit --benchmark humaneval --tag p21-qwen27b-plain
eval $E --model $O/lmstudio-community/Qwen3.8-27B-MLX-4bit --benchmark humaneval --think-bias 1024:0.02 --tag p22-qwen27b-bias1024-0.02
eval $E --model $O/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit --benchmark humaneval --tag p23-qwen35b-plain
eval $E --model $O/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit --benchmark humaneval --think-bias 1024:0.02 --tag p24-qwen35b-bias1024-0.02
echo PROBES_DONE
