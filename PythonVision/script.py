#!/usr/bin/env python3
"""Run one PythonVision algorithm over the captures in its own input folder.

Each algorithm lives in its own folder with the same shape:

    PythonVision/
        common/         shared geometry, image io and drawing
        script.py       this file
        run.sh          thin wrapper that finds the venv's python
        aruco/
            input/      FieldBW.png + capture_*.png
            out/        written here
            aruco.py    the algorithm

This harness does only what every algorithm needs and would otherwise repeat:
resolve the algorithm's folders, collect the captures (an explicit list, else
every `capture_*.png` in `input/`), create the output folder, and time the run.
Anything specific to an algorithm - its thresholds, its report, the images it
writes - stays in that algorithm's `main`.

Usage:
    python script.py aruco                  # every capture in aruco/input
    python script.py features --steps       # algorithm-specific flags pass through
    python script.py gradient capture_7_x0.0_y0.0_yaw0.0_pitch45.0.png

An algorithm module exposes:

    build_parser(parser)  -> None      add its own flags
    run(captures, out_dir, args) -> int   solve every capture, write its output
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

# Run as a script (`../.venv/bin/python script.py aruco`) rather than as part of
# a package, so the workspace root has to be importable before `common` is. This
# also makes `python -m script` work from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import paths  # noqa: E402  (needs the sys.path insert above)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("algorithm",
                        help="Algorithm folder to run, e.g. 'aruco'.")
    parser.add_argument("captures", nargs="*",
                        help="Capture name(s) in the algorithm's input folder. "
                             "Default: every capture_*.png there.")
    parser.add_argument("--list", action="store_true",
                        help="List the algorithm's captures and exit.")
    parser.add_argument("--no-output", action="store_true",
                        help="Solve but write nothing into out/.")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print the final summary line.")
    return parser


def load_algorithm(name: str):
    """Import `<name>/<name>.py` as a module, or explain what is missing.

    The algorithm is imported by path rather than as a package because the
    folder name and the module name are the same, and a folder named `aruco`
    holding `aruco.py` cannot be imported as `aruco` without the package's
    `__init__` shadowing it.
    """
    folder = paths.algorithm_dir(name)
    module_path = os.path.join(folder, f"{name}.py")
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"no algorithm folder at {folder}")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"no algorithm module at {module_path}")

    if folder not in sys.path:
        sys.path.insert(0, folder)
    return importlib.import_module(name)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    try:
        algorithm = load_algorithm(args.algorithm)
    except (FileNotFoundError, ImportError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    if not hasattr(algorithm, "run"):
        print(f"[error] {args.algorithm}.py defines no run()", file=sys.stderr)
        return 2

    # Give the algorithm a second pass at the same argv so its own flags are
    # declared and help text stays accurate; `extra` is what this parser did not
    # recognise. `--no-output` is shared, so it is re-presented to whichever
    # side declares it.
    algo_parser = argparse.ArgumentParser(
        prog=f"script.py {args.algorithm}",
        description=(algorithm.__doc__ or "").splitlines()[0])
    algo_parser.add_argument("captures", nargs="*")
    algo_parser.add_argument("--no-output", action="store_true")
    if hasattr(algorithm, "build_parser"):
        algorithm.build_parser(algo_parser)
    algo_args = algo_parser.parse_args(extra + args.captures)

    captures = algo_args.captures or []
    resolved = [paths.resolve_capture(args.algorithm, name) for name in captures]
    if not resolved:
        resolved = paths.captures(args.algorithm)
    if args.list:
        for path in resolved:
            print(path)
        return 0
    if not resolved:
        print(f"[error] no captures in {paths.input_dir(args.algorithm)}",
              file=sys.stderr)
        return 2

    out_dir = paths.output_dir(args.algorithm)
    if not args.no_output:
        os.makedirs(out_dir, exist_ok=True)

    print(f"algorithm : {args.algorithm}")
    print(f"input     : {paths.input_dir(args.algorithm)}")
    print(f"output    : {out_dir}")
    print(f"captures  : {len(resolved)}\n")

    # `--no-output` is declared on both parsers, so overwrite rather than
    # duplicate it, and carry `--quiet` across from the harness's own parser.
    run_args = argparse.Namespace(**vars(algo_args))
    run_args.no_output = args.no_output
    run_args.quiet = args.quiet

    start = time.perf_counter()
    status = algorithm.run(resolved, out_dir, run_args)
    elapsed = (time.perf_counter() - start) * 1000.0
    print(f"\n{args.algorithm}: exit {status}, {elapsed:.0f} ms total")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
