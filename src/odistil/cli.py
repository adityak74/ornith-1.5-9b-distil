"""odistil -- two-teacher distillation pipeline for Ornith-1.5-9B.

    odistil check                      environment, models, tokenizer compatibility
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

from .config import Config


def _cfg(args) -> Config:
    return Config.load(args.models_config, args.distill_config)


def cmd_check(args) -> int:
    from .check import run

    return run(_cfg(args), download=args.download)


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


def cmd_fuse(args) -> int:
    from .pipeline.train import fuse

    print(fuse(_cfg(args)))
    return 0


def cmd_quantize(args) -> int:
    from .pipeline.train import quantize

    print(quantize(_cfg(args), args.variant))
    return 0


def cmd_eval(args) -> int:
    from .eval.runner import run

    run(_cfg(args), args.model, benchmarks=args.benchmark, limit=args.limit, tag=args.tag)
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

    s = sub.add_parser("fuse", help="stage 5: fuse adapters into a bf16 checkpoint")
    s.set_defaults(fn=cmd_fuse)

    s = sub.add_parser("quantize", help="stage 6: quantize the fused checkpoint")
    s.add_argument("--variant", required=True, help="q4 | q8 | oq4 | oq8 | oq3")
    s.set_defaults(fn=cmd_quantize)

    s = sub.add_parser("eval", help="stage 7: run the benchmarks")
    s.add_argument("--model", required=True,
                   help="student | student:oq4 | teacher:code | repo id | path")
    s.add_argument("--benchmark", action="append", help="mmlu | truthfulqa | humaneval")
    s.add_argument("--limit", type=int, help="cap items per benchmark")
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
