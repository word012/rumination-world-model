"""Generate manuscript Figures 2–4 from their model parameters and random seeds."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New directory for the three figure PDFs.")
    output = parser.parse_args().output_dir.absolute()
    if output.exists() or output.is_symlink():
        parser.error(f"Choose a new output directory: {output}")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from paper_distributions import generate_distributions
    from paper_ensemble import generate_ensemble
    from paper_plot import render_figures

    print("Simulating trajectories and the parameter grid (Figures 2, 3 and 4a)...", flush=True)
    ensemble, grid = generate_ensemble()
    print("Computing readout distributions (Figure 4c,e)...", flush=True)
    positive, negative = generate_distributions()
    output.mkdir(parents=True)
    try:
        files = render_figures(ensemble, grid, positive, negative, output)
    except BaseException:
        # The renderer removes its own incomplete files. Preserve any others.
        if not any(output.iterdir()):
            output.rmdir()
        raise
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
