"""Export the checked fixed-shape (1, 512) float32 Core ML package.

The model wrapper, reference checks, and Core ML conversion live in
export_512_common.py so the F32 and F16 entrypoints use the same workflow.
"""

import argparse
from pathlib import Path

import export_512_common as exporter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=exporter.ROOT / "artifacts/coreml/f32-512")
    args = parser.parse_args()
    exporter.main(args.output_dir, precision="float32", entrypoint=Path(__file__))


if __name__ == "__main__":
    main()
