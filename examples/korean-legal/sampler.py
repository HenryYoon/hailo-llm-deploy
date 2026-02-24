"""Step 1: Stratified sampling from raw CSV into train/val/test splits."""

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class Sampler:
    """Stratified sampling to produce balanced train/val/test splits."""

    def __init__(
        self,
        raw_csv: Path,
        output_dir: Path,
        metadata_dir: Path,
        valid_labels: list[str],
        train_size: int = 1000,
        val_size: int = 100,
        test_size: int = 100,
        seed: int = 42,
    ):
        self.raw_csv = raw_csv
        self.output_dir = output_dir
        self.metadata_dir = metadata_dir
        self.valid_labels = valid_labels
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed

    def load_and_clean(self) -> pd.DataFrame:
        """Load raw CSV and filter to valid major_label rows."""
        df = pd.read_csv(self.raw_csv, dtype=str)
        original_count = len(df)
        df = df[df['major_label'].isin(self.valid_labels)].copy()
        df = df.dropna(subset=['question', 'answer'])
        logger.info("Loaded %d rows, kept %d after cleaning", original_count, len(df))
        return df

    def stratified_sample(self, df: pd.DataFrame) -> tuple:
        """Two-stage stratified split into train/val/test."""
        total_size = self.train_size + self.val_size + self.test_size

        sampled, _ = train_test_split(
            df, train_size=total_size, stratify=df['major_label'],
            random_state=self.seed,
        )
        train_df, temp_df = train_test_split(
            sampled, train_size=self.train_size, stratify=sampled['major_label'],
            random_state=self.seed,
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['major_label'],
            random_state=self.seed,
        )

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def save_stats(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save stratification statistics to JSON."""
        stats = {
            "total": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
            "per_label": {},
        }
        for label in self.valid_labels:
            stats["per_label"][label] = {
                "train": int((train_df['major_label'] == label).sum()),
                "val": int((val_df['major_label'] == label).sum()),
                "test": int((test_df['major_label'] == label).sum()),
            }

        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        stats_path = self.metadata_dir / "sampling_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info("Saved sampling stats to %s", stats_path)

    def run(self) -> tuple:
        """Execute stratified sampling pipeline."""
        df = self.load_and_clean()
        train_df, val_df, test_df = self.stratified_sample(df)

        logger.info("Split sizes: train=%d, val=%d, test=%d",
                     len(train_df), len(val_df), len(test_df))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_path = self.output_dir / "sampled_train.csv"
        val_path = self.output_dir / "sampled_val.csv"
        test_path = self.output_dir / "sampled_test.csv"

        train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
        val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
        test_df.to_csv(test_path, index=False, encoding='utf-8-sig')

        self.save_stats(train_df, val_df, test_df)
        return train_df, val_df, test_df
