"""Export the checked fixed-shape (1, 512) mixed-FP16 Core ML package.

Only gather and linear operations use FP16. Attention, softmax, normalization,
residuals, and pooling remain float32 for numerical stability. The common
export workflow is in export_512_common.py.
"""

import argparse
from pathlib import Path

import export_512_common as exporter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=exporter.ROOT / "artifacts/coreml/f16-512-mixed")
    args = parser.parse_args()
    exporter.main(args.output_dir, precision="float16", entrypoint=Path(__file__))
