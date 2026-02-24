"""
Step 1: Stratified sampling from law_qa_v1.csv
Produces train (1000), val (100), test (100) splits stratified by major_label.
"""
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_CSV, PROCESSED_DATA, METADATA,
    SAMPLED_TRAIN, SAMPLED_VAL, SAMPLED_TEST, SAMPLING_STATS,
    VALID_MAJOR_LABELS, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, RANDOM_SEED,
)


def load_and_clean(csv_path):
    """Load raw CSV and filter to valid major_label rows only."""
    df = pd.read_csv(csv_path, dtype=str)
    original_count = len(df)
    df = df[df['major_label'].isin(VALID_MAJOR_LABELS)].copy()
    df = df.dropna(subset=['question', 'answer'])
    cleaned_count = len(df)
    print(f"Loaded {original_count} rows, kept {cleaned_count} after cleaning")
    return df


def stratified_sample(df):
    """
    Two-stage stratified split:
    1. Sample 1200 from full dataset (stratified)
    2. Split 1200 -> train 1000 + temp 200 (stratified)
    3. Split temp 200 -> val 100 + test 100 (stratified)
    """
    total_size = TRAIN_SIZE + VAL_SIZE + TEST_SIZE  # 1200

    sampled, _ = train_test_split(
        df,
        train_size=total_size,
        stratify=df['major_label'],
        random_state=RANDOM_SEED,
    )

    train_df, temp_df = train_test_split(
        sampled,
        train_size=TRAIN_SIZE,
        stratify=sampled['major_label'],
        random_state=RANDOM_SEED,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['major_label'],
        random_state=RANDOM_SEED,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def save_stats(train_df, val_df, test_df):
    """Save stratification statistics."""
    stats = {
        "total": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "per_label": {}
    }
    for label in VALID_MAJOR_LABELS:
        stats["per_label"][label] = {
            "train": int((train_df['major_label'] == label).sum()),
            "val": int((val_df['major_label'] == label).sum()),
            "test": int((test_df['major_label'] == label).sum()),
        }

    METADATA.mkdir(parents=True, exist_ok=True)
    with open(SAMPLING_STATS, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Saved sampling stats to {SAMPLING_STATS}")


def run():
    """Execute Step 1: stratified sampling."""
    print("=" * 50)
    print("Step 1: Stratified Sampling")
    print("=" * 50)

    df = load_and_clean(RAW_CSV)
    print(f"\nLabel distribution (full dataset):")
    print(df['major_label'].value_counts().to_string())

    train_df, val_df, test_df = stratified_sample(df)

    print(f"\nSplit sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"\nTrain label distribution:")
    print(train_df['major_label'].value_counts().to_string())

    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SAMPLED_TRAIN, index=False, encoding='utf-8-sig')
    val_df.to_csv(SAMPLED_VAL, index=False, encoding='utf-8-sig')
    test_df.to_csv(SAMPLED_TEST, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {SAMPLED_TRAIN}, {SAMPLED_VAL}, {SAMPLED_TEST}")

    save_stats(train_df, val_df, test_df)
    return train_df, val_df, test_df


if __name__ == "__main__":
    run()
