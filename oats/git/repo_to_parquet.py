#!/usr/bin/env python3
"""
Simple script to convert a single Git repository to Parquet format.

Extracts all commits from the specified repository, builds a pandas DataFrame,
and saves the result as a Parquet file. Defaults to ``/opt/ds/oats`` as the
source repository and ``/tmp/repo-commits.pq`` as the output path.
"""

import argparse
import traceback
import pandas as pd
from oats.git.git_to_df_converter import extract_git_commits
from oats.git.git_to_df_converter import save_dataframe_to_parquet
from oats.log import cl

log = cl("convert-git-to-pq")


def get_arg_vals() -> argparse.Namespace:
    """Parse and return CLI arguments for the repo-to-Parquet converter.

    Returns:
        Parsed namespace with ``repo_path`` and ``out_file`` attributes.
    """
    parser = argparse.ArgumentParser(description="Convert Git repository to Parquet format")
    parser.add_argument(
        "-r",
        "--repo-path",
        default="/opt/ds/oats",
        help="Path to Git repository (default: /opt/ds/oats)",
    )
    parser.add_argument(
        "-o",
        "--out-file",
        default="/tmp/repo-commits.pq",
        help="Output Parquet file path (default: /tmp/repo-commits.pq)",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    """CLI entry point — parse args and convert a Git repository to Parquet format."""
    # Define paths
    out_file = "/tmp/repo-commits.pq"
    repo_path = "/opt/ds/oats"
    args = get_arg_vals()
    repo_path = args.repo_path
    if args.out_file is None:
        out_file = args.out_file
    df = convert_repo_to_parquet(repo_path=repo_path, out_file=out_file)
    if df is None:
        log.error(f'### Sorry!! Failed to convert repo: {repo_path} to df')
        return
    log.debug('most-recent-git-commit:')
    print(df.iloc[0])
    log.debug('2nd-most-recent-git-commit:')
    print(df.iloc[1])


def convert_repo_to_parquet(repo_path: str, out_file: str) -> pd.DataFrame | None:
    """Convert a Git repository's commit history to a Parquet file.

    Extracts all commits from the given repository, builds a pandas DataFrame,
    and saves it to the specified Parquet output path.

    Args:
        repo_path: Path to the Git repository.
        out_file: Path where the Parquet file will be written.

    Returns:
        The DataFrame of commits on success, or ``None`` on failure.
    """
    log.info(f"Converting Git repository at '{repo_path}' to Parquet format...")
    log.info(f"Output will be saved to: {out_file}")
    df = None
    try:
        # Extract commits from the repository
        df = extract_git_commits(repo_path)
        # Save to Parquet file
        save_dataframe_to_parquet(df, out_file)
        log.info(f"Successfully converted {len(df)} commits to Parquet format!")
        log.info(f"File saved to: {out_file}")
        # Display some information about the data
        if len(df) > 0:
            log.info("\nSample data:")
            log.info(f"First commit: {df.iloc[0]['message'][:100]}...")
            log.info(f"Latest commit: {df.iloc[-1]['message'][:100]}...")
            log.info(f"Author: {df.iloc[0]['author_name']}")
    except Exception:
        log.error(f"### Sorry!! Failed to convert repo: {repo_path} out_file: {out_file} with error:\n\n{traceback.format_exc()}")
    return df


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error(f"### Sorry!! Failed to process git repo to parquet with error:\n\n{traceback.format_exc()}")
