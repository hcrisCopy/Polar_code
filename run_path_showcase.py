"""Run from project root: python -B ./Polar_code/run_path_showcase.py ..."""

import sys

sys.dont_write_bytecode = True

from path_showcase.cli import main


if __name__ == "__main__":
    main()

