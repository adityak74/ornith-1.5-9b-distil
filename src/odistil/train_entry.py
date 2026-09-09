"""Training entry point that installs the chunkwise gated-delta path first.

`mlx_lm lora` runs in its own process, so the patch has to be applied inside it
before the model is built. This wrapper does that and then hands over to
mlx-lm's own CLI unchanged, so every flag and config key behaves as before.
"""

from __future__ import annotations

import sys


def main() -> None:
    chunk = 64
    argv = []
    it = iter(sys.argv[1:])
    for arg in it:
        if arg == "--gdn-chunk":
            chunk = int(next(it))
        else:
            argv.append(arg)

    if chunk > 0:
        from .gdn_chunkwise import install

        install(chunk=chunk)
        print(f"[odistil] chunkwise gated-delta training path active (C={chunk})", flush=True)

    from mlx_lm.lora import main as lora_main

    sys.argv = ["mlx_lm.lora", *argv]
    lora_main()


if __name__ == "__main__":
    main()
