"""Export the checked fixed-shape (1, 512) F32 package with an iOS 18 target.

This is the source artifact for Core ML's 4-bit weight compression. It remains
an independent F32 export and must pass saved-package assessment before it is
used as a compression source.
"""

import argparse
from pathlib import Path

import coremltools as ct

import export_512_common as exporter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=exporter.ROOT / "artifacts/coreml/f32-512-ios18")
    args = parser.parse_args()
    exporter.main(
        args.output_dir,
        precision="float32",
        entrypoint=Path(__file__),
        minimum_deployment_target=ct.target.iOS18,
        deployment_target_name="iOS18",
    )


if __name__ == "__main__":
    main()
