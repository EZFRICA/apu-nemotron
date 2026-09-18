#!/usr/bin/env python3
"""
Prepare the data for a live demo: reset the device state, build and import the course
registry locally (no Google credentials), seed example escalations with their clusters, and a sample student notebook.

    uv run python scripts/prepare_demo.py
    uv run python scripts/prepare_demo.py --no-reset        # keep existing memory and events
    uv run python scripts/prepare_demo.py --skip-courses    # escalations only (fast)

The first run downloads the local embedding model (~240 MB) if it is not in ./models yet.
The same actions are available from the "Demo setup" page of the interface.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apu.demo.seed import prepare_demo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-reset", action="store_true", help="do not wipe data/ first")
    parser.add_argument("--skip-courses", action="store_true", help="do not build and import courses")
    args = parser.parse_args()

    report = prepare_demo(reset=not args.no_reset, load_courses=not args.skip_courses)
    print("\nDemo ready.")
    print(f"  courses     : {report.courses or 'not loaded'}")
    print(f"  escalations : {report.escalations_by_class}")
    print(f"  clusters    : {report.clusters_by_class}")
    print(f"  notebook    : {report.notebook_entries} entries")
    print("\nLaunch the interface:  uv run streamlit run apu/ui/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
