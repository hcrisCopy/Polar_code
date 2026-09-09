"""Launch from the project root: python -B Polar_code/run_stage_one.py ..."""

import sys

sys.dont_write_bytecode = True

from stage_one.cli import main


if __name__ == "__main__":
    main()
