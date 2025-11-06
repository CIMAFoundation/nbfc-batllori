import argparse
from pathlib import Path

from .simulation import (
    DEFAULT_DATA_PATH,
    DEFAULT_FIRE_PATH,
    DEFAULT_RESULTS_PATH,
    run_simulation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Batllori vegetation-fire simulation.")
    parser.add_argument("--start-year", type=int, default=1978, help="First year to simulate (default: 1978)")
    parser.add_argument("--end-year", type=int, default=2020, help="Year after the last simulated year (default: 2020)")
    parser.add_argument(
        "--land-cover-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Directory containing prpYYYY_mktx.asc rasters (default: ./data/prp_mktx)",
    )
    parser.add_argument(
        "--fire-path",
        type=Path,
        default=DEFAULT_FIRE_PATH,
        help="Directory containing i_YYYY_utm32.asc rasters (default: ./data/inc_utm32)",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Directory where encoded outputs will be written (default: ./results)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_simulation(
        start_year=args.start_year,
        end_year=args.end_year,
        data_path=args.land_cover_path,
        fire_path=args.fire_path,
        results_path=args.results_path,
    )


if __name__ == "__main__":
    main()
