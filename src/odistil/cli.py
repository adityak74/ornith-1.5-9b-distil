"""odistil -- two-teacher distillation pipeline for Ornith-1.5-9B.

    odistil check                      environment, models, tokenizer compatibility
    odistil status                     progress of every stage
    odistil prompts [--domain code]    stage 1: build the prompt pool
    odistil teach   [--teacher code]   stage 2: generate teacher traces
    odistil dataset                    stage 3: verify + decontaminate + mix
    odistil train   [--resume]         stage 4: LoRA distillation
    odistil fuse                       stage 5: fuse adapters -> bf16 checkpoint
    odistil quantize --variant q4      stage 6: quantize (or run an oQ recipe)
    odistil eval --model student       stage 7: MMLU / TruthfulQA / HumanEval
    odistil report                     comparison table vs the baselines
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config


def _cfg(args) -> Config:
    return Config.load(args.models_config, args.distill_config)


def cmd_check(args) -> int:
    from .check import run

    return run(_cfg(args), download=args.download)


def cmd_status(args) -> int:
    from .status import run

    return run(_cfg(args))


def cmd_prompts(args) -> int:
    from .pipeline.prompts import build

    build(_cfg(args), domains=args.domain, limit=args.limit)
    return 0


def cmd_teach(args) -> int:
    from .pipeline.teach import run

    run(_cfg(args), teachers=args.teacher, limit=args.limit)
    return 0


def cmd_dataset(args) -> int:
    from .pipeline.dataset import build

    build(_cfg(args), skip_decontam=args.skip_decontam)
    return 0


def cmd_train(args) -> int:
    from .pipeline.train import train

    train(_cfg(args), resume=args.resume, extra=args.extra)
    return 0


def cmd_logits(args) -> int:
    from .logits import extract

    cfg = _cfg(args)
    for split in ("train", "valid"):
        extract(cfg, split=split, top_k=args.top_k)
    return 0


def cmd_distill(args) -> int:
    from .pipeline.distill_train import sanity_check, train

    cfg = _cfg(args)
    d = cfg.distill.get("distill", {})
    alpha = args.alpha if args.alpha is not None else d.get("alpha", 0.3)
    top_k = d.get("top_k", 64)
    if args.check:
        sanity_check(cfg, alpha=alpha, top_k=top_k)
        return 0
    train(cfg, alpha=alpha, top_k=top_k)
    return 0


def cmd_fuse(args) -> int:
    from .pipeline.train import fuse

    print(fuse(_cfg(args), dequantize=args.dequantize))
    return 0


def cmd_quantize(args) -> int:
    from .pipeline.train import quantize

    print(quantize(_cfg(args), args.variant))
    return 0


def cmd_package(args) -> int:
    from .pipeline.package import package

    cfg = _cfg(args)
    package(cfg, Path(args.src), args.name, install=args.install)
    return 0


def cmd_eval(args) -> int:
    from .eval.runner import run

    run(_cfg(args), args.model, benchmarks=args.benchmark, limit=args.limit,
        sample=args.sample, tag=args.tag)
    return 0


def cmd_report(args) -> int:
    from .eval.report import table, write

    cfg = _cfg(args)
    print(table(cfg, show_time=args.time))
    print(f"\n-> {write(cfg)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="odistil", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-config", default=None)
    p.add_argument("--distill-config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check", help="preflight: env, models, tokenizer compatibility")
    s.add_argument("--download", action="store_true", help="fetch missing models from the Hub")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("status", help="where the pipeline is")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("prompts", help="stage 1: build the prompt pool")
    s.add_argument("--domain", action="append", help="restrict to a domain (repeatable)")
    s.add_argument("--limit", type=int, help="cap items per source (smoke tests)")
    s.set_defaults(fn=cmd_prompts)

    s = sub.add_parser("teach", help="stage 2: generate teacher traces")
    s.add_argument("--teacher", action="append", help="knowledge | code (repeatable)")
    s.add_argument("--limit", type=int)
    s.set_defaults(fn=cmd_teach)

    s = sub.add_parser("dataset", help="stage 3: verify, decontaminate, mix")
    s.add_argument("--skip-decontam", action="store_true", help="smoke tests only")
    s.set_defaults(fn=cmd_dataset)

    s = sub.add_parser("train", help="stage 4: LoRA distillation")
    s.add_argument("--resume", action="store_true")
    s.add_argument("extra", nargs="*", help="extra flags passed through to mlx_lm lora")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("logits", help="stage 3b: teacher top-k logprobs for logit distillation")
    s.add_argument("--top-k", type=int, default=64)
    s.set_defaults(fn=cmd_logits)

    s = sub.add_parser("distill", help="stage 4b: train against the teacher's distribution")
    s.add_argument("--alpha", type=float, help="weight on hard-target CE (default from config)")
    s.add_argument("--check", action="store_true", help="one batch through the loss, then stop")
    s.set_defaults(fn=cmd_distill)

    s = sub.add_parser("fuse", help="stage 5: fuse adapters into the trained-on base")
    s.add_argument("--dequantize", action="store_true",
                   help="emit bf16 instead of keeping a quantized base quantized")
    s.set_defaults(fn=cmd_fuse)

    s = sub.add_parser("quantize", help="stage 6: quantize the fused checkpoint")
    s.add_argument("--variant", required=True, help="q4 | q8 | oq4 | oq8 | oq3")
    s.set_defaults(fn=cmd_quantize)

    s = sub.add_parser("package", help="make a checkpoint loadable by oMLX")
    s.add_argument("--src", required=True, help="checkpoint directory to package")
    s.add_argument("--name", required=True, help="model name, e.g. Ornith-1.5-9B-MLX-distil-oQ4")
    s.add_argument("--install", action="store_true", help="copy into the oMLX model dir")
    s.set_defaults(fn=cmd_package)

    s = sub.add_parser("eval", help="stage 7: run the benchmarks")
    s.add_argument("--model", required=True,
                   help="student | student:oq4 | teacher:code | repo id | path")
    s.add_argument("--benchmark", action="append", help="mmlu | truthfulqa | humaneval")
    s.add_argument("--limit", type=int, help="first N items per benchmark (debugging)")
    s.add_argument("--sample", type=int,
                   help="seeded random subset per benchmark -- use this for iteration runs")
    s.add_argument("--tag", help="name for the results directory")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("report", help="comparison table vs the baselines")
    s.add_argument("--time", action="store_true", help="include wall-clock hours")
    s.set_defaults(fn=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
