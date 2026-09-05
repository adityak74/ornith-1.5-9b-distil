.PHONY: setup check smoke prompts teach dataset train fuse eval report test lint clean

setup:            ## install the env
	uv sync --extra dev

check:            ## preflight (add DOWNLOAD=1 to fetch models)
	uv run odistil check $(if $(DOWNLOAD),--download,)

prompts:; uv run odistil prompts
teach:;   uv run odistil teach
dataset:; uv run odistil dataset
train:;   uv run odistil train
fuse:;    uv run odistil fuse

quant-q4:; uv run odistil quantize --variant q4
quant-q8:; uv run odistil quantize --variant q8

eval-baseline:    ## re-measure the shipped oQ4 student
	uv run odistil eval --model student:oq4 --tag baseline-oq4

eval-fused:       ## measure the distilled bf16 student
	uv run odistil eval --model runs/v1/fused --tag distilled-bf16

eval-quant:       ## measure the distilled + quantized student
	uv run odistil eval --model runs/v1/quant/q4 --tag distilled-q4

report:;  uv run odistil report --time

smoke:            ## tiny end-to-end pass on 24 items / 20 iters
	uv run odistil --distill-config configs/smoke.yaml prompts
	uv run odistil --distill-config configs/smoke.yaml teach
	uv run odistil --distill-config configs/smoke.yaml dataset --skip-decontam
	uv run odistil --distill-config configs/smoke.yaml train
	uv run odistil --distill-config configs/smoke.yaml eval --model student --benchmark humaneval --limit 5

test:;    uv run pytest -q
lint:;    uv run ruff check src tests
clean:;   rm -rf runs/smoke .pytest_cache .ruff_cache
