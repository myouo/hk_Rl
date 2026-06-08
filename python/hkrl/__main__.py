"""Console entry point (``hkrl-train``). Thin wrapper over scripts/train.py logic.

TODO(phase-2): parse args (--config, --smoke), build env+model+algo from config,
run training. Kept minimal so the package imports without heavy deps.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    # TODO(phase-2): delegate to a training entrypoint built from TrainConfig.
    raise NotImplementedError("training entry point not implemented yet")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
