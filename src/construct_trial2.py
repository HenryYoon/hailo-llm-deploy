"""
Trial 2 RAFT Dataset Construction Pipeline — Main Entry Point.

Usage:
    python src/construct_trial2.py                    # Run all steps
    python src/construct_trial2.py --step sample      # Step 1: Stratified sampling
    python src/construct_trial2.py --step extract     # Step 2: Reference extraction
    python src/construct_trial2.py --step collect     # Step 3: External data collection
    python src/construct_trial2.py --step chunk       # Step 4: Document chunking
    python src/construct_trial2.py --step build       # Step 5: RAFT assembly
"""
import argparse
import sys

import sampler
import extractor
import collector
import chunker
import raft_builder


STEPS = {
    'sample': sampler.run,
    'extract': extractor.run,
    'collect': collector.run,
    'chunk': chunker.run,
    'build': raft_builder.run,
}

ALL_STEPS = ['sample', 'extract', 'collect', 'chunk', 'build']


def main():
    parser = argparse.ArgumentParser(description="Trial 2 RAFT Dataset Construction")
    parser.add_argument(
        '--step',
        choices=list(STEPS.keys()) + ['all'],
        default='all',
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    if args.step == 'all':
        steps_to_run = ALL_STEPS
    else:
        steps_to_run = [args.step]

    for step_name in steps_to_run:
        print(f"\n{'#' * 60}")
        print(f"# Running: {step_name}")
        print(f"{'#' * 60}\n")
        STEPS[step_name]()

    print(f"\n{'#' * 60}")
    print("# Pipeline complete!")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
