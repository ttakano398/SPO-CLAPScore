import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime


def apply_screening(df, threshold):
    """Remove outliers based on the threshold."""
    print(f"INFO: Screening with threshold={threshold}...")
    indices_to_keep = []

    # Iterate over each wav file group
    for _, group in df.groupby("wav_file_name"):
        for idx_a, row_a in group.iterrows():
            score_a = row_a["score"]
            # Get scores of other listeners in the group
            scores_b = group.drop(idx_a)["score"]

            # Keep if any other score falls within the range [score_a - th, score_a + th]
            if (
                (scores_b >= score_a - threshold) & (scores_b <= score_a + threshold)
            ).any():
                indices_to_keep.append(idx_a)

    screened_df = df.loc[indices_to_keep].sort_index()
    print(
        f"  -> Original: {len(df)} | Kept: {len(screened_df)} | Removed: {len(df) - len(screened_df)}"
    )
    return screened_df


def get_stats(df):
    """Calculate listener statistics (mean, std) and global statistics from training data."""
    stats = df.groupby("listener_id")["score"].agg(["mean", "std"])
    stats["std"] = stats["std"].replace(0, 1).fillna(1)  # Avoid division by zero

    global_mean = stats["mean"].mean()
    global_std = np.sqrt(np.mean(stats["std"] ** 2))

    return stats, {"mean": global_mean, "std": global_std}


def standardize(df, listener_stats, global_stats, is_train=True):
    """Apply Z-score standardization."""
    df_out = df.copy()

    if is_train:
        # Train: Standardize using listener-specific stats
        df_merged = df_out.merge(
            listener_stats, left_on="listener_id", right_index=True, how="left"
        )

        # Fill missing listener stats with global stats
        df_merged["mean"] = df_merged["mean"].fillna(global_stats["mean"])
        df_merged["std"] = df_merged["std"].fillna(global_stats["std"])

        df_out["standard_score"] = (df_merged["score"] - df_merged["mean"]) / df_merged[
            "std"
        ]
    else:
        # Validation: Standardize using global stats
        df_out["standard_score"] = (
            df_out["score"] - global_stats["mean"]
        ) / global_stats["std"]

    # Add metadata columns
    df_out["global_mean"] = global_stats["mean"]
    df_out["global_std"] = global_stats["std"]
    df_out["wav_std"] = (
        df_out.groupby("wav_file_name")["standard_score"].transform("std").fillna(0)
    )

    return df_out


def aggregate_average(df):
    """Aggregate scores by wav file."""
    # Create separate columns for individual scores (score1, score2, ...)
    score_lists = df.groupby("wav_file_name")["standard_score"].apply(list)
    scores_expanded = pd.DataFrame(score_lists.tolist(), index=score_lists.index)
    scores_expanded.columns = [f"score{i+1}" for i in range(scores_expanded.shape[1])]

    # Aggregate metrics
    avg_df = df.groupby("wav_file_name").agg(
        text=("text", "first"),
        average_score=("standard_score", "mean"),
        global_mean=("global_mean", "first"),
        global_std=("global_std", "first"),
        wav_std=("wav_std", "first"),
    )

    # Concatenate: [text, average_score, stats..., score1, score2...]
    final_df = pd.concat([avg_df, scores_expanded], axis=1).reset_index()
    return final_df


def update_and_save_config(
    base_config_path, train_path, val_path, global_stats, date_str
):
    """Load existing config, update paths and stats, and save as new file in ./cfg."""

    # Load base config
    try:
        with open(base_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {base_config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {base_config_path}", file=sys.stderr)
        sys.exit(1)

    # Update paths
    config["train_list"] = train_path
    config["validation_list"] = val_path

    # Update stats
    if "model" not in config:
        config["model"] = {}

    config["model"]["train_standard_stats"] = {
        "mean": global_stats["mean"],
        "std": global_stats["std"],
    }

    # Save to ./cfg directory
    save_dir = "./cfg"
    os.makedirs(save_dir, exist_ok=True)

    save_filename = f"config_{date_str}.json"
    save_path = os.path.join(save_dir, save_filename)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"\nConfiguration file generated: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess: Screening and Standardization"
    )
    parser.add_argument(
        "data_dir", type=str, help="Directory containing train.csv and validation.csv"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base config.json file (Required)",
    )
    parser.add_argument(
        "--screening_threshold",
        type=int,
        default=None,
        help="Threshold for outlier screening (optional)",
    )
    args = parser.parse_args()

    # Generate date string for filenames (e.g., 251231)
    date_str = datetime.now().strftime("%y%m%d")

    listener_stats, global_stats = None, None

    # Store paths for config generation
    config_paths = {"train": None, "validation": None}

    for split in ["train", "validation"]:
        file_path = os.path.join(args.data_dir, f"{split}.csv")
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found.", file=sys.stderr)
            continue

        print(f"\nProcessing {split}...")
        df = pd.read_csv(file_path)

        # 1. Screening
        suffix = ""
        if args.screening_threshold is not None:
            df = apply_screening(df, args.screening_threshold)
            suffix = f"_screened_{args.screening_threshold}"

        # 2. Calculate Stats (Train only)
        if split == "train":
            listener_stats, global_stats = get_stats(df)
            print(f"Global Stats: {global_stats}")

        # 3. Standardization
        df_std = standardize(
            df, listener_stats, global_stats, is_train=(split == "train")
        )

        # 4. Save Standardized (Row-based)
        cols = list(df.columns) + [c for c in df_std.columns if c not in df.columns]
        df_std = df_std[cols]

        # Append date to filename
        save_path_std = os.path.join(
            args.data_dir, f"{split}{suffix}_standard_zscore_{date_str}.csv"
        )
        df_std.to_csv(save_path_std, index=False)
        print(f"Saved standard: {save_path_std}")

        # 5. Save Averaged (Aggregated)
        df_avg = aggregate_average(df_std)
        # Append date to filename
        save_path_avg = os.path.join(
            args.data_dir, f"{split}{suffix}_standard_average_zscore_{date_str}.csv"
        )
        df_avg.to_csv(save_path_avg, index=False)
        print(f"Saved average:  {save_path_avg}")

        # Store path for config
        config_paths[split] = save_path_avg

    # Generate Config if both files were processed successfully
    if config_paths["train"] and config_paths["validation"] and global_stats:
        update_and_save_config(
            args.config,
            config_paths["train"],
            config_paths["validation"],
            global_stats,
            date_str,
        )
    else:
        print("\nWARNING: Config generation skipped (missing train/val data or stats).")


if __name__ == "__main__":
    main()
