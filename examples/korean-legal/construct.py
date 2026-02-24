"""Korean Legal QA RAFT Dataset Construction Pipeline.

Orchestrates the 5-step pipeline using class-based modules.

Usage:
    python examples/korean-legal/construct.py                    # Run all steps
    python examples/korean-legal/construct.py --step sample      # Step 1
    python examples/korean-legal/construct.py --step extract     # Step 2
    python examples/korean-legal/construct.py --step collect     # Step 3
    python examples/korean-legal/construct.py --step chunk       # Step 4
    python examples/korean-legal/construct.py --step build       # Step 5
"""

import argparse
import logging
import sys

# Allow running as a script from project root
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from config import (
    RAW_CSV, PROCESSED_DATA, METADATA,
    SAMPLED_TRAIN, SAMPLED_VAL, SAMPLED_TEST,
    REFERENCE_EXTRACTION, CHUNK_DATABASE,
    TRAIN_DATASET, VAL_DATASET, TEST_DATASET,
    EXTERNAL_STATUTES, EXTERNAL_CASES,
    VALID_MAJOR_LABELS, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, RANDOM_SEED,
    CHUNK_SIZE, CHUNK_OVERLAP,
    DOCS_PER_SAMPLE, MAX_ORACLE_DOCS, DISTRACTOR_SAME_LABEL_RATIO,
    API_DELAY,
)
from sampler import Sampler
from extractor import ReferenceExtractor
from collector import LawApiCollector
from chunker import DocumentChunker
from raft_builder import RaftBuilder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SAMPLED_PATHS = {"train": SAMPLED_TRAIN, "val": SAMPLED_VAL, "test": SAMPLED_TEST}
DATASET_PATHS = {"train": TRAIN_DATASET, "val": VAL_DATASET, "test": TEST_DATASET}


def step_sample():
    sampler = Sampler(
        raw_csv=RAW_CSV, output_dir=PROCESSED_DATA, metadata_dir=METADATA,
        valid_labels=VALID_MAJOR_LABELS,
        train_size=TRAIN_SIZE, val_size=VAL_SIZE, test_size=TEST_SIZE, seed=RANDOM_SEED,
    )
    sampler.run()


def step_extract():
    extractor = ReferenceExtractor(
        sampled_paths=SAMPLED_PATHS,
        reference_extraction=REFERENCE_EXTRACTION, metadata_dir=METADATA,
    )
    extractor.run()


def step_collect():
    collector = LawApiCollector(
        reference_extraction=REFERENCE_EXTRACTION,
        external_statutes=EXTERNAL_STATUTES, external_cases=EXTERNAL_CASES,
        api_delay=API_DELAY,
    )
    collector.run()


def step_chunk():
    chunker = DocumentChunker(
        external_statutes=EXTERNAL_STATUTES, external_cases=EXTERNAL_CASES,
        reference_extraction=REFERENCE_EXTRACTION, chunk_database=CHUNK_DATABASE,
        metadata_dir=METADATA, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    )
    chunker.run()


def step_build():
    builder = RaftBuilder(
        sampled_paths=SAMPLED_PATHS, reference_extraction=REFERENCE_EXTRACTION,
        chunk_database=CHUNK_DATABASE, dataset_paths=DATASET_PATHS,
        processed_data=PROCESSED_DATA,
        docs_per_sample=DOCS_PER_SAMPLE, max_oracle_docs=MAX_ORACLE_DOCS,
        distractor_same_label_ratio=DISTRACTOR_SAME_LABEL_RATIO, random_seed=RANDOM_SEED,
    )
    builder.run()


STEPS = {
    'sample': step_sample,
    'extract': step_extract,
    'collect': step_collect,
    'chunk': step_chunk,
    'build': step_build,
}

ALL_STEPS = ['sample', 'extract', 'collect', 'chunk', 'build']


def main():
    parser = argparse.ArgumentParser(description="Korean Legal QA RAFT Dataset Construction")
    parser.add_argument(
        '--step', choices=list(STEPS.keys()) + ['all'], default='all',
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    steps_to_run = ALL_STEPS if args.step == 'all' else [args.step]

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
