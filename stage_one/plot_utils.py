"""Shared publication-figure style and guarded vector export."""

from pathlib import Path
import os

from .storage import output_path


BASELINE_COLOR = "#7F7F7F"
PROGRAM_COLOR = "#0072B2"
SKIP_COLOR = "#D55E00"
LOOP_COLOR = "#0072B2"
KEEP_COLOR = "#BDBDBD"


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def save_vector(fig, stem):
    """Save matching SVG/PDF files, both constrained to Polar_data."""
    stem = Path(stem)
    outputs = []
    for suffix in (".svg", ".pdf"):
        target = output_path(stem.with_suffix(suffix))
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = output_path(target.with_suffix(target.suffix + ".pending"))
        if pending.exists():
            raise ValueError(f"Unrecovered interrupted figure write: {pending}")
        fig.savefig(pending, format=suffix.removeprefix("."))
        with pending.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(pending, target)
        if os.name == "posix":
            descriptor = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        outputs.append(str(target))
    return outputs
